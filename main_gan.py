
#3.334363660844155, acc 0.9754098360655737, loss 0.6908055394887924

import os

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
import matplotlib.pyplot as plt

from RealExpGAN import expGAN

import numpy as np
import scipy.sparse as sp

from scipy.linalg import orthogonal_procrustes
from evaluate import calculate_frechet_distance, LSTMClassifier
import math


data_list = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise"]
index2label = {}
label2index = {}
for i in range(6):
    index2label[i] = data_list[i]
    label2index[data_list[i]] = i


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPS = 1e-6


C = - 0.5 * math.log( 2 * math.pi)

class FaceData(Dataset):
    def __init__(self, data, label, mask): 
        self.pts = data
        self.labels = label 
        self.length = len(data)
        self.mask = mask

        
        
    def __len__(self):

        return self.length

    def __getitem__(self, idx):
        pc = self.pts[idx]
     #   pc = np.reshape(self.pts[idx], (100, 83,3))  # 100x249 => 100x83x3
        # pc, coeff = on_unit_cube(pc)

        mask = np.expand_dims(self.mask[idx], -1)
        pc = pc * mask


        pc = torch.FloatTensor(pc)

        # s = torch.FloatTensor([coeff['scale']])
        # m = torch.FloatTensor(coeff['min'])
        y = self.labels[idx]
        return pc, y, mask





def train_step(net, dataloader, optimizer):
    net.train()
    total_loss_G = []
    total_loss_D = []
    kld = []
    total_data = 0
    total_error = 0
    correct = 0
    train_results = {}
    for data in dataloader:
        pts, label, mask = data
        pts, label, mask = pts.to(device).float(), label.to(device), mask.to(device)
        bs = pts.shape[0]
        total_data += bs
        

        net(pts, label, mask)
        results = net.optimize_parameters()
        

        total_error += results["loss_recon"].item()
       
        total_loss_G.append(results["loss_G"].item())
        total_loss_D.append(results["loss_D"].item())
        kld.append(results["kld"].item())
        
    train_results["recon_loss"] = total_error/len(dataloader)
    train_results["loss_G"] = np.mean(total_loss_G)
    train_results["loss_D"] = np.mean(total_loss_D)
    train_results["kld"] = np.mean(kld)
    return train_results

def feature_extract(net, classifier, feature):
    with torch.no_grad():
        for i, l in enumerate(net.results['y']):
            l = l.item()
            _, f = classifier(net.results['x'][i].unsqueeze(0))
            feature[l].append(f.squeeze().detach().cpu().numpy())

         #for i, l in enumerate(net.y):

        #     l = l.item()
        #     _, f = classifier(net.recon_x[i].unsqueeze(0))
        #     feature[l].append(f.squeeze().detach().cpu().numpy())

    return feature
    
def valid_step(net, dataloader, classifier, cal_fid = False):
    net.eval()
    total_loss_G = []
    total_loss_D = []
    kld = []
    total_data = 0
    total_error = 0
    correct = 0
    valid_results = {}
    feature = {}
    for i in range(6):
        feature[i] = []

    with torch.no_grad():
        for data in dataloader:
            pts, label, mask = data
            pts, label, mask = pts.to(device).float(), label.to(device), mask.to(device)
            bs = pts.shape[0]
            total_data += bs
            
            net(pts, label, mask)

            results = net.optimize_parameters(False)

            total_error += results["loss_recon"].item()
            
            total_loss_G.append(results["loss_G"].item())
            total_loss_D.append(results["loss_D"].item())
            kld.append(results["kld"].item())
            
            if cal_fid == True:
                feature = feature_extract(net, classifier, feature)


        valid_results["recon_loss"] = total_error/len(dataloader)
        valid_results["loss_G"] = np.mean(total_loss_G)
        valid_results["loss_D"] = np.mean(total_loss_D)
        valid_results["kld"] = np.mean(kld)

    fid = 0
    

    if cal_fid == True:
        for i in range(6):
            fid += calculate_frechet_distance(np.array(feature[i]), np.array(gt_feature[i]))
    return valid_results, fid/6

# def find_zeros(gt, pred):
#     #bs, 100, 249
#     gt_sum =
def visulization( displacement, filename, first_frame = None):
    #first frame: , 1, 249
    #displacement: , 99, 249
 

    

    data = np.zeros((num_frame, num_pts*3))
    if first_frame:
        data[0] = first_frame
        for i, f in enumerate(displacement):
            
            data[i+1] = data[i] + f
    else:
        for i, f in enumerate(displacement):
            data[i] = f
    np.save( filename, data)


def test_step(net, dataloader, classifier):
    net.eval()
    total_loss_G = []
    total_loss_D = []
    kld = []
    total_data = 0
    total_error = 0
    correct = 0
    test_results = {}
    recon = []

    feature = {}
    for i in range(6):
        feature[i] = []

    sample_count = 0
    recon_feature = {}    
    for i in range(6):
        recon_feature[i] = []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            pts, label, mask = data
            pts, label, mask = pts.to(device).float(), label.to(device), mask.to(device)
            bs = pts.shape[0]
            total_data += bs


            z = net(pts, label, mask)

            feature = feature_extract(net, classifier, feature)
            
            for i, sample in enumerate(net.x_sample):
                recon_sample = (sample*std).cpu().numpy()+m_data
                label_sample = net.y_sample[i].item()
                sample_count += 1
                visulization( recon_sample, checkpoint+data_list[label_sample]+'_sample'+str(sample_count)+'.npy')
            
            
          #  print("disentangle data")
            for a, p in enumerate(pts):
            #    p = (p).cpu().numpy()
                p =( p * mask[a]*std ).cpu().numpy() + m_data
                visulization( p, checkpoint+'/sub_'+str(a)+data_list[label[a]]+'_gt.npy')

            
            for i in range(6):
                y = torch.ones_like(label).to(device) * i 
                
                disenResults = net.sample(z, y, num_frame)
                _, f = classifier(disenResults)
                for sub_f in f:
                    recon_feature[i].append(sub_f.squeeze().detach().cpu().numpy())

                disenResults = (disenResults*std).cpu().numpy()+m_data
                disenResults = disenResults * mask.cpu().numpy()
                for n, sub in enumerate(disenResults):
                    visulization( sub, checkpoint+data_list[i]+'_'+str(n)+'.npy')
    fid = 0
    recon_fid = 0
    print(recon_feature)
    for i in range(6):
        
        fid += calculate_frechet_distance(np.array(feature[i]), np.array(gt_feature[i]))
        recon_fid += calculate_frechet_distance(np.array(recon_feature[i]), np.array(gt_feature[i]))
    return test_results, fid/6, recon_fid/6


def Displacement(subject):
    # subject (606, 100, 249) 

    masks = []


    data = []
    displace = []
    for sub in subject:

        mask = np.ones(100)
        disp = np.zeros((100, 249))
        for i, frame in enumerate(sub):




            if frame.max() - frame.min() != 0:


                
                disp[i] = frame
                data.append(frame)

            else:
                mask[i] = 0
                disp[i] = frame
        displace.append(disp)

        masks.append(mask)
    return np.array(displace), np.array(masks), np.array(data)


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass
 


lam = 1e-4
num_frame = 100
num_pts = 83
z = 50
c = 6
epochs = 5000
lr = 0.000001
weight_decay = 0.0001
label_size = 6
latent_size = 64
batch_size = 8
hidden_unit = 256
num_head = 4
dropout = 0.2
num_layers = 4
checkpoint = './results/RealGAN/'

data_path = 'dataset.pkl'
dataset = np.load(data_path, allow_pickle=True)
subject = np.array(dataset["data"])
labels =  np.array(dataset["labels"])
subject = np.reshape(subject, (606, 100 , 249))
displace, masks, data = Displacement(subject)
m_data = data.mean(0)
std = data.std(0)

trainsformed_data = (subject-m_data)/std
std = torch.tensor(std).to(device)


random_seeds = 2021
test_size = 0.2
best_loss = 1000000
#net = expGAN(3*num_pts, num_frame, label_size, hidden_unit, latent_size, num_head,num_layers, dropout).to(device)
import pickle5 as pickle

#gt_feature = np.load("training_feature.pkl", allow_pickle = True)
with open( 'training_feature.pkl', 'rb') as f:
    gt_feature = pickle.load(f)
if not os.path.exists(checkpoint):
    os.makedirs(checkpoint)
skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state = random_seeds) 
n = 0
sys.stdout = Logger(checkpoint + 'training.txt')
for train_index, test_index in skf.split(subject, labels):
    net = expGAN(3*num_pts, num_frame, label_size, hidden_unit, latent_size, num_head,num_layers, dropout).to(device)


    n += 1

    classifier = torch.load("./evaluate/classfier_"+str(n)+'.pt').to(device)
    
    train, valid= train_test_split(train_index, test_size=test_size, random_state = random_seeds)
    
    train_data = trainsformed_data[train]
    train_labels = labels[train]
    train_mask = masks[train]

    print("number of train data:", train_data.shape[0])
    #mean, std, train_data = normalize(train_data)
    train_dataset = FaceData(train_data, train_labels, train_mask)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    valid_data = trainsformed_data[valid]
    valid_labels = labels[valid]
    vallid_mask = masks[valid]
    print("number of valid data:", valid_data.shape[0])
   # _, _, valid_data = normalize(valid_data, mean = mean, std = std)
    valid_dataset = FaceData(valid_data, valid_labels, vallid_mask)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
#    std = torch.FloatTensor(std).unsqueeze(0).unsqueeze(0).to(device)
    best_fid = 1000

    for ep in range(epochs):

        if ep > 100:
            for p in zip(net.optimizer_G.param_groups, net.optimizer_D.param_groups, net.optimizer_C.param_groups):
                p[0]['lr'] = 0.00001
                p[1]['lr'] = 0.00001
                p[2]['lr'] = 0.00001
        elif ep > 500:
            for p in zip(net.optimizer_G.param_groups, net.optimizer_D.param_groups, net.optimizer_C.param_groups):
                p[0]['lr'] = 0.00005
                p[1]['lr'] = 0.00005
                p[2]['lr'] = 0.00005

        elif ep > 1000:
            for p in zip(net.optimizer_G.param_groups, net.optimizer_D.param_groups, net.optimizer_C.param_groups):
                p[0]['lr'] = 0.000001
                p[1]['lr'] = 0.000001
                p[2]['lr'] = 0.000001

        cal_fid = False
        if ep % 10 == 0:
            cal_fid = True
        train_results = train_step(net,  train_loader, optimizer)
        valid_results, fid = valid_step(net,  valid_loader, classifier, cal_fid)
    
        print("epoch {}, training: recon loss {},loss_D {}, loss_G {}, loss kld {}".format(ep,  train_results["recon_loss"], train_results["loss_D"],train_results["loss_G"], train_results["kld"]))
        print(" validation: recon loss {}, loss_D {},loss_G {}, loss kld {}, fid {}".format(valid_results["recon_loss"], valid_results["loss_D"],valid_results["loss_G"], valid_results["kld"], fid))
        if cal_fid == True and fid < best_fid:
            best_fid = fid 
            torch.save(net,checkpoint+ 'GAN_'+str(n)+'.pt')
    
    
    test_data = trainsformed_data[test_index]
    test_labels = labels[test_index]
    test_mask = masks[test_index]
    print("number of test data:", test_data.shape[0])
    test_dataset = FaceData(test_data, test_labels, test_mask)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=4)
    net = torch.load(checkpoint + '/GAN_'+str(n)+'.pt')
    test_results, fid, recon_fid = test_step(net,  test_loader, classifier)
    print("Finish! ")
    print("fid", fid)
    print("recon_fid", recon_fid)


