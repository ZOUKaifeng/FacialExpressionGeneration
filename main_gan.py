
#3.334363660844155, acc 0.9754098360655737, loss 0.6908055394887924

import os


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
import matplotlib.pyplot as plt

from ExpGAN import expGAN

import numpy as np
import scipy.sparse as sp
import  open3d as o3d
from scipy.linalg import orthogonal_procrustes

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
        
        

        y_one_hot = torch.zeros(label.shape[0], 6)

        y_one_hot.scatter_(1, label.type(torch.LongTensor).unsqueeze(1), 1)

        y_one_hot = torch.clamp(y_one_hot, EPS, 1-EPS).to(device)

        results = net(pts, y_one_hot, mask)
        

        total_error += results["loss_recon"].item()
       
        total_loss_G.append(results["loss_G"].item())
        total_loss_D.append(results["loss_D"].item())
        kld.append(results["kld"].item())
        
    train_results["recon_loss"] = total_error/len(dataloader)
    train_results["loss_G"] = np.mean(total_loss_G)
    train_results["loss_D"] = np.mean(total_loss_D)
    train_results["kld"] = np.mean(kld)
    return train_results
    
def valid_step(net, dataloader):
    net.eval()
    total_loss_G = []
    total_loss_D = []
    kld = []
    total_data = 0
    total_error = 0
    correct = 0
    valid_results = {}
    with torch.no_grad():
        for data in dataloader:
            pts, label, mask = data
            pts, label, mask = pts.to(device).float(), label.to(device), mask.to(device)
            bs = pts.shape[0]
            total_data += bs
            
            

            y_one_hot = torch.zeros(label.shape[0], 6)

            y_one_hot.scatter_(1, label.type(torch.LongTensor).unsqueeze(1), 1)

            y_one_hot = torch.clamp(y_one_hot, EPS, 1-EPS).to(device)

            results = net(pts, y_one_hot, mask, train = False)
            

            total_error += results["loss_recon"].item()
            
            total_loss_G.append(results["loss_G"].item())
            total_loss_D.append(results["loss_D"].item())
            kld.append(results["kld"].item())
        
        valid_results["recon_loss"] = total_error/len(dataloader)
        valid_results["loss_G"] = np.mean(total_loss_G)
        valid_results["loss_D"] = np.mean(total_loss_D)
        valid_results["kld"] = np.mean(kld)
    
    return valid_results

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


def test_step(net, dataloader):
    net.eval()
    total_loss_G = []
    total_loss_D = []
    kld = []
    total_data = 0
    total_error = 0
    correct = 0
    test_results = {}
    recon = []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            pts, label, mask = data
            pts, label, mask = pts.to(device).float(), label.to(device), mask.to(device)
            bs = pts.shape[0]
            total_data += bs


            y_one_hot = torch.zeros(label.shape[0], 6)

            y_one_hot.scatter_(1, label.type(torch.LongTensor).unsqueeze(1), 1)

            y_one_hot = torch.clamp(y_one_hot, EPS, 1-EPS).to(device)

            results = net(pts, y_one_hot, mask, train = False)


            total_error += results["loss_recon"].item()
            
            total_loss_G.append(results["loss_G"].item())
            total_loss_D.append(results["loss_D"].item())
            kld.append(results["kld"].item())
        


            if i == len(dataloader)-1:
                print("disentangle data")
                for a, p in enumerate(pts):
                #    p = (p).cpu().numpy()
                    p =( p * mask[a]*std ).cpu().numpy() + m_data
                    print(label[a])
                    visulization( p, checkpoint+'/sub_'+str(a)+data_list[label[a]]+'_gt.npy')

                
                z = results['z']
                for i in range(6):
                    y = torch.ones_like(label).to(device) * i
                    y_one_hot = torch.zeros(y.shape[0], 6)

                    y_one_hot.scatter_(1, y.type(torch.LongTensor).unsqueeze(1), 1)

                    y_one_hot = torch.clamp(y_one_hot, EPS, 1-EPS).to(device)*2
                
                
                    disenResults = net.sample(pts, z, y_one_hot, mask)
                    disenResults = (disenResults*std).cpu().numpy()+m_data
                    disenResults = disenResults * mask.cpu().numpy()
                    for n, sub in enumerate(disenResults):
                        print(np.sum(sub))
               
                        visulization( sub, checkpoint+data_list[i]+'_'+str(n)+'.npy')



    test_results["recon_loss"] = total_error/len(dataloader)
    test_results["loss_G"] = np.mean(total_loss_G)
    test_results["loss_D"] = np.mean(total_loss_D)
    test_results["kld"] = np.mean(kld)

    
    return test_results

def Displacement(subject):
    # subject (606, 100, 249) 

    masks = []


    data = []
    displace = []
    for sub in subject:

        mask = np.ones(num_frame)
        disp = np.zeros((num_frame, num_pts*3))
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

lam = 1e-4
num_frame = 80
num_pts = 68
z = 50
c = 6
epochs = 10000
lr = 0.0001
weight_decay = 0.0001
label_size = 6
latent_size = 64
batch_size = 8
hidden_unit = 256
num_head = 4
dropout = 0.2
num_layers = 4
checkpoint = './results/GAN/'

data_path = 'BUFaceData.pkl'
MUG_datapath = 'MUGFaceData.pkl'
dataset = np.load(data_path, allow_pickle=True)
subject_ = []
labels_ = []
for sub in dataset:
    for exp in dataset[sub]:
        if dataset[sub][exp]["maker"].shape[0] < 40:
            continue
        labels_.append(label2index[exp])
        subject_.append(dataset[sub][exp]["maker"][:num_frame])
# subject_ = dataset["marker"]
# labels_ =  dataset["label"]
data = np.load(MUG_datapath, allow_pickle=True)
for sub in data.keys():
    for exp in data[sub].keys():
        for act in data[sub][exp].keys():
            if data[sub][exp][act]["marker"].shape[0] < 40:
                continue
            labels_.append(label2index[exp])
            subject_.append(data[sub][exp][act]["marker"][:num_frame])

labels = np.array(labels_)
subject = np.array(subject_)
print(subject.shape)
print(labels.shape)
subject = np.reshape(subject, (-1, num_frame , num_pts*3))
displace, masks, data = Displacement(subject)
m_data = data.mean(0)
std = data.std(0)

trainsformed_data = (displace-m_data)/std
std = torch.tensor(std).to(device)


random_seeds = 2021
test_size = 0.2
best_loss = 1000000
net = expGAN(3*num_pts, num_frame, label_size, hidden_unit, latent_size, num_head,num_layers, dropout).to(device)
if not os.path.exists(checkpoint):
    os.makedirs(checkpoint)
skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state = random_seeds) 
for train_index, test_index in skf.split(subject, labels):
#     train, valid= train_test_split(train_index, test_size=test_size, random_state = random_seeds)
    
#     train_data = trainsformed_data[train]
#     train_labels = labels[train]
#     train_mask = masks[train]

#     print("number of train data:", train_data.shape[0])
#     #mean, std, train_data = normalize(train_data)
#     train_dataset = FaceData(train_data, train_labels, train_mask)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
#     valid_data = trainsformed_data[valid]
#     valid_labels = labels[valid]
#     vallid_mask = masks[valid]
#     print("number of valid data:", valid_data.shape[0])
#    # _, _, valid_data = normalize(valid_data, mean = mean, std = std)
#     valid_dataset = FaceData(valid_data, valid_labels, vallid_mask)
#     valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
#     optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
# #    std = torch.FloatTensor(std).unsqueeze(0).unsqueeze(0).to(device)

#     for ep in range(epochs):
#         train_results = train_step(net,  train_loader, optimizer)
#         valid_results = valid_step(net,  valid_loader)
    
#         print("epoch {}, training: recon loss {},loss_D {}, loss_G {}, loss kld {}".format(ep,  train_results["recon_loss"], train_results["loss_D"],train_results["loss_G"], train_results["kld"]))
#         print(" validation: recon loss {}, loss_D {},loss_G {}, loss kld {}".format(valid_results["recon_loss"], valid_results["loss_D"],valid_results["loss_G"], valid_results["kld"]))

#         torch.save(net,checkpoint+ 'GAN.pt')


    test_data = trainsformed_data[test_index]
    test_labels = labels[test_index]
    test_mask = masks[test_index]
    print("number of test data:", test_data.shape[0])
    test_dataset = FaceData(test_data, test_labels, test_mask)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=4)
    net = torch.load(checkpoint + '/GAN.pt')
    test_results = test_step(net,  test_loader)
    print(" validation: recon loss {}, loss_D {},loss_G {}, loss kld {}".format(test_results["recon_loss"], test_results["loss_D"],test_results["loss_G"], test_results["kld"]))
    break