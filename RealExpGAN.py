
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import itertools
from model import ENCODER,DECODER, DisenFace

def KLD(mu, log_sigma):
    return -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - (log_sigma).exp(), -1)

class Discriminator(nn.Module):
    """docstring for Discriminator"""
    def __init__(self, num_pts, num_frame, label_size):
        super(Discriminator, self).__init__()
        self.input_embedding = nn.Linear(num_pts, 256)
        
        self.lstm_1 = nn.LSTM(256, 256, batch_first=True, bidirectional=True)

        self.fc_1 = nn.Linear(512, 256)

        self.cls_2 = nn.Linear(256, 1)

    def forward(self, x, one_hot_y):

        bs = x.shape[0]

        x = F.relu(self.input_embedding(x))
        
        x, (hidden, cell) = self.lstm_1(x)
        

        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)

        hidden = hidden.view(bs, -1)

        x  = F.relu(self.fc_1(hidden))
    #    x = torch.cat((one_hot_y, x), dim = -1)
        logits  = self.cls_2(x)


        return torch.sigmoid(logits)

class Classifier(nn.Module):
    """docstring for Discriminator"""
    def __init__(self, num_pts, num_frame, label_size):
        super(Classifier, self).__init__()
        self.input_embedding = nn.Linear(num_pts, 256)
        
        self.lstm_1 = nn.LSTM(256, 256, batch_first=True, bidirectional=True)
        self.cls_1 = nn.Linear(512, 256)
        self.cls_2 = nn.Linear(256, label_size)

    def forward(self, x):

        bs = x.shape[0]

        x = F.relu(self.input_embedding(x))
        x, (hidden, cell) = self.lstm_1(x)
        

        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)

        hidden = hidden.view(bs, -1)

        f  = self.cls_1(hidden)
        logits = self.cls_2(F.relu(f))
        return logits, f


class expGAN(nn.Module):
    """docstring for ClassName"""
    def __init__(self, num_pts, num_frame, label_size, hidden_unit,  latent_size, num_head, num_layers=3, dropout = 0.1):
        super(expGAN, self).__init__()

        # self.E = ENCODER( num_pts, hidden_unit, latent_size, num_layers, num_head, label_size, dropout = 0.1)
        # self.G = DECODER(num_pts, hidden_unit, latent_size, num_layers, num_head, label_size, dropout = 0.1)
        self.G = DisenFace(num_pts, num_frame, label_size, hidden_unit,  latent_size, num_head)
        self.D = Discriminator(num_pts, num_frame, label_size)
        self.C = Classifier(num_pts, num_frame, label_size)
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.G.parameters()), lr=0.000001, betas=(0.99, 0.999))
       # self.optimizer_E = torch.optim.Adam(, lr=0.0001, betas=(0.99, 0.999))
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=0.000001, betas=(0.99, 0.999))
        self.optimizer_C = torch.optim.Adam(self.C.parameters(), lr=0.000001, betas=(0.99, 0.999))
        self.valid_label_scale = 1
        self.criterionGAN = torch.nn.MSELoss()
        self.criterionCycle = torch.nn.L1Loss()
        self.crossentropy = nn.CrossEntropyLoss()
        self.feature = torch.zeros((label_size, hidden_unit)).cuda()
        self.sample_feature = torch.zeros((label_size, hidden_unit), device="cuda", requires_grad=True)
        
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def forward(self, x, y, mask):
        EPS = 1e-6
      #  self.results = {}
        self.y = y
        self.x = x
        self.results = self.G(x, y, mask)
        num_frame = x.shape[1]
        batch_size = x.shape[0]
        self.device = x.device

        self.recon_x = self.results['x']

        y_one_hot = torch.zeros(self.y.shape[0], 6)
        y_one_hot.scatter_(1, self.y.type(torch.LongTensor).unsqueeze(1), 1)
        self.y_one_hot = torch.clamp(y_one_hot, EPS, 1-EPS).to(self.device)


        self.mean = self.results['mean']
        self.sigma = self.results['log_sigma']
        z = self.reparameterize(self.mean, self.sigma)

       # self.recon_x = self.G(z, num_frame, y)

        #sample
        z_sample = torch.randn(z.shape[0], z.shape[1]).to(self.device)
        self.y_sample = torch.randint(6, size = (z.shape[0],)).to(self.device)
    #    z_sample = torch.cat((z_sample, self.y_sample), dim = 1)

        self.one_hot_sample = torch.zeros(self.y_sample.shape[0], 6)
        self.one_hot_sample.scatter_(1,self.y_sample.type(torch.LongTensor).unsqueeze(1), 1)
        self.one_hot_sample = torch.clamp(self.one_hot_sample, EPS, 1-EPS).to(self.device)

        self.x_sample = self.G.sample(z_sample, self.y_sample, num_frame)


        return self.results['mean']

    def reparameterize(self, mu, logvar):

        batch_size = mu.shape[0]
        dim = logvar.shape[-1]

        std = torch.exp(logvar * 0.5)
  
        z = torch.normal(mean = 0, std = 1, size=(batch_size, dim)).to(mu.device)
        z = z*std + mu


        return z


    def loss_C(self):

        logits, self.real_f = self.C(self.x)

        loss = self.crossentropy(logits, self.y.long())
        return loss


    def loss_D(self):

        batch_size = self.x.shape[0]

        valid_label = torch.ones((batch_size, 1)).to(self.device) * ( self.valid_label_scale )

        fake_label = torch.zeros((batch_size, 1)).to(self.device) 
          
        pred_real = self.D(self.x, self.y_one_hot )
        loss_real = self.criterionGAN(pred_real, valid_label)
        pred_fake = self.D(self.recon_x.detach(), self.y_one_hot.detach())
        loss_fake = self.criterionGAN(pred_fake, fake_label)


        pred_sample = self.D(self.x_sample.detach(), self.one_hot_sample.detach())
        loss_sample = self.criterionGAN(pred_fake, fake_label)

        loss = (loss_fake + loss_real + loss_sample)/3
        return loss

    def kld_loss(self):

        kld_loss = KLD(self.results['mean'] , self.results['log_sigma'] )
        return kld_loss.mean()

    def loss_G(self):
        batch_size = self.x.shape[0]
        valid_label = torch.ones((batch_size, 1)).to(self.device) 
       # pred_fake = self.D(self.recon_x, self.y_one_hot)
      #  pred_sample = self.D(self.x_sample, self.one_hot_sample)
     #   loss_G = self.criterionGAN(pred_sample, valid_label)
        loss_recon = self.criterionCycle(self.results["x"], self.x)  


        self.results["loss_recon"] = loss_recon
        return loss_recon


    def loss_CR(self):
        logits, recon_f = self.C(self.recon_x)
        return self.criterionCycle(self.real_f, recon_f)  

    def loss_GC(self):
        # moving average https://discuss.pytorch.org/t/how-to-apply-exponential-moving-average-decay-for-variables/10856

        _, real_f = self.C(self.x)
        bs = real_f.shape[0]
       # self.feature = self.feature.to(self.device)
        for i in range(bs):
            label = self.y[i]
            self.feature[label] = 0.9*self.feature[label].detach() + 0.1*real_f[i].detach()

       
        _, sample_f = self.C(self.x_sample)
        #self.sample_feature = self.sample_feature.to(self.device).grad.zero_()
       # self.sample_feature.grad.zero_()
        self.sample_feature = self.sample_feature.detach()
        for i in range(bs):
            label = self.y_sample[i]


            self.sample_feature[label] = 0.9*self.sample_feature[label].detach() + 0.1*sample_f[i]
            

        loss_gc = F.mse_loss(self.sample_feature, self.feature)
        return loss_gc





    def optimize_parameters(self, optim = True):
       

       # optimize C:
        if optim:
            self.set_requires_grad([self.C, self.D], True)
            
            self.optimizer_C.zero_grad()
            loss_C = self.loss_C()
        
            loss_C.backward()
            self.optimizer_C.step()


        #optimize D:

        self.optimizer_D.zero_grad()
        loss_D = self.loss_D()
        if optim:
            loss_D.backward()
            self.optimizer_D.step()
        self.results["loss_D"] = loss_D
#        self.results["loss_D"] = torch.tensor(0)
    #     #optimize E:

        self.set_requires_grad([self.C, self.D], False)
        self.optimizer_G.zero_grad()

        loss_kld = self.kld_loss()
        loss_G = self.loss_G()
        loss_E = loss_G + 0.0001*loss_kld

        self.results["loss_G"] = loss_G
        self.results["kld"] = loss_kld
        # optimize G:
        
        loss_GC = self.loss_GC()
        loss =  loss_G  + 0.0001*loss_kld + loss_GC
        if optim:
            loss.backward()
            self.optimizer_G.step()


        return self.results




    def sample(self, z, y, num_frame):
        return self.G.sample(z, y, num_frame)

#z + c :
#timequries: zeros
# results: static
