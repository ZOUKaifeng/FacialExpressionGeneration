
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


from model import DisenFace

def KLD(mu, log_sigma):
    return -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - (log_sigma).exp(), -1)

class Discriminator(nn.Module):
    """docstring for Discriminator"""
    def __init__(self, num_pts, num_frame, label_size):
        super(Discriminator, self).__init__()
        self.input_embedding = nn.Linear(num_pts, 256)
        
        self.lstm_1 = nn.LSTM(256, 256, batch_first=True, bidirectional=False)

        self.fc_1 = nn.Linear(256, 256)

        self.cls_2 = nn.Linear(256+label_size, 1)

    def forward(self, x, one_hot_y):

        bs = x.shape[0]

        x = F.relu(self.input_embedding(x))
        
        x, (hidden, cell) = self.lstm_1(x)
        
        x = x[:, -1, :]

        x  = F.relu(self.fc_1(x))
        x = torch.cat((one_hot_y, x), dim = -1)
        logits  = self.cls_2(x)


        return torch.sigmoid(logits)

class Discriminator_without_label(nn.Module):
    """docstring for Discriminator"""
    def __init__(self, num_pts, num_frame, label_size):
        super(Discriminator_without_label, self).__init__()
        self.input_embedding = nn.Linear(num_pts, 256)
        
        self.lstm_1 = nn.LSTM(256, 256, batch_first=True, bidirectional=False)
        self.cls_1 = nn.Linear(256, 256)
        self.cls_2 = nn.Linear(256, 1+label_size)

    def forward(self, x, one_hot_y):

        bs = x.shape[0]

        x = F.relu(self.input_embedding(x))
        
        x, (hidden, cell) = self.lstm_1(x)
        

       # hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)

        # out = lstm_out * alpha 
        # out = F.relu(torch.sum(out, 1))
      #  x = F.relu(self.linear(hidden))
        x = x[:, -1, :]
        x = torch.cat((one_hot_y, x), dim = -1)
        x  = self.cls_1(x)
        
        logits  = self.cls_2(x)


        return torch.sigmoid(logits[0]), torch.softmax(logits[1:])

class expGAN(nn.Module):
    """docstring for ClassName"""
    def __init__(self, num_pts, num_frame, label_size, hidden_unit,  latent_size, num_head, num_layers=3, dropout = 0.1):
        super(expGAN, self).__init__()

        self.G = DisenFace(num_pts, num_frame, label_size, hidden_unit,  latent_size, num_head)
        self.D = Discriminator(num_pts, num_frame, label_size)
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=0.0001, betas=(0.99, 0.999))
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=0.0001, betas=(0.99, 0.999))
        self.valid_label_scale = 1
        self.criterionGAN = torch.nn.MSELoss()
        self.criterionCycle = torch.nn.L1Loss()

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

    def forward(self, x, one_hot_y, mask, train = True):
        results = {}
        batch_size = x.shape[0]
        self.device = x.device
        results_tvae = self.G(x, one_hot_y, mask, train)


        z = results_tvae["mean"].detach()
        self.sample_y = torch.zeros_like(one_hot_y).to(self.device)
        index = np.random.randint(6, size = batch_size)

        self.sample_y[:, index] = 1
        self.sample_x = self.sample(x, z, self.sample_y, mask)


        self.recon_x = results_tvae["x"]
        self.x = x
        self.label = one_hot_y
        self.optimizer_D.zero_grad()
        loss_D = self.optimize_discriminator()
        if train:
            loss_D.backward()
            self.optimizer_D.step()  

        self.optimizer_G.zero_grad() 
        loss_G, loss_recon, loss_kld = self.optimize_generator(results_tvae)
        loss = loss_G + loss_recon + 0.0001*loss_kld.mean()
        if train:
            loss.backward()
            self.optimizer_G.step()  

        results["fake_x"] = self.recon_x 
        results["loss_D"] = loss_D
        results["loss_G"] = loss_G 
        results["loss_recon"] = loss_recon 
        results["kld"] = 0.0001*loss_kld.mean()
        results["z"] = results_tvae["mean"]



        return results

    def optimize_discriminator(self):

        batch_size = self.x.shape[0]

        valid_label = torch.ones((batch_size, 1)).to(self.device) * ( self.valid_label_scale )

        fake_label = torch.zeros((batch_size, 1)).to(self.device) 
          
        pred_real = self.D(self.x, self.label)
        loss_real = self.criterionGAN(pred_real, valid_label)
        pred_fake = self.D(self.recon_x.detach(), self.label)
        loss_fake = self.criterionGAN(pred_fake, fake_label)
        pred_sample = self.D(self.sample_x.detach(), self.sample_y )
        loss_sample = self.criterionGAN(pred_sample, fake_label)


        loss = (loss_fake + loss_real+loss_sample)/3

        return loss


    def optimize_generator(self, vae_results):
        batch_size = self.x.shape[0]
        valid_label = torch.ones((batch_size, 1)).to(self.device) 
        loss_recon = self.criterionCycle(self.recon_x, self.x)
        pred_fake = self.D(self.recon_x, self.label)
        loss_G = self.criterionGAN(pred_fake, valid_label)
        loss_kld = KLD(vae_results["mean"], vae_results["log_sigma"])


        pred_sample = self.D(self.sample_x, self.sample_y )
        loss_sample = self.criterionGAN(pred_sample, valid_label)


        return (loss_G+loss_sample)/2, loss_recon, loss_kld


    def sample(self, inputs, z, one_hot_y, mask):
        x = self.G.sample( inputs, z, one_hot_y, mask)
        return x
#z + c :
#timequries: zeros
# results: static