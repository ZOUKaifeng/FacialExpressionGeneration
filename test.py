import os


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
import matplotlib.pyplot as plt

from model import DisenFace

import numpy as np
import scipy.sparse as sp
import  open3d as o3d
from scipy.linalg import orthogonal_procrustes
from evaluate import  LSTMClassifier, calculate_frechet_distance
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





def KLD(mu, log_sigma):
    return -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp())
def gaussian_nll(mu, log_sigma, x):
    return 0.5 * torch.pow((x - mu) / log_sigma.exp(), 2) + log_sigma + 0.5 * np.log(2 * np.pi)

def loss_funtion(results, pts, mask):
   
    log_sigma = torch.FloatTensor([1]).to(device)


    gaussian_loss = F.mse_loss(results["x"], pts, reduction = 'mean')


  #  gaussian_loss = gaussian_nll( results["x"], log_sigma ,pts).sum(1).sum(1)
    kld_loss = KLD(results["mean"], results["log_sigma"]).mean()
    return  gaussian_loss, lam*kld_loss



# def find_zeros(gt, pred):
#     #bs, 100, 249
#     gt_sum =
def visulization( displacement, filename, first_frame = None):
    #first frame: , 1, 249
    #displacement: , 99, 249
 

    

    data = np.zeros((num_frame, num_pts*3))
    if first_frame is not None:
        data[0] = first_frame
        for i, f in enumerate(displacement):
            
            data[i+1] = data[i] + f
    else:
        for i, f in enumerate(displacement):
            data[i] = f
    np.save( filename, data)


def test_step(net, dataloader):
    net.eval()
    total_loss = []
    total_data = 0
    total_error = 0
    correct = 0
    test_results = {}
    recon = []
    recon_results = {}
    ground_truth = {}
    for i in range(6):
        recon_results[i] = []
        ground_truth[i] = []

    with torch.no_grad():
        for t, data in enumerate(dataloader):

            pts, label, mask = data
            pts, label, mask = pts.to(device).float(), label.to(device), mask.to(device)
            bs = pts.shape[0]
            total_data += bs

            for ii, tt in enumerate(label):
               # print(pts[ii].shape)
              
                ground_truth[tt.item()].append(pts[ii])



            y_one_hot = torch.zeros(label.shape[0], 6)

            y_one_hot.scatter_(1, label.type(torch.LongTensor).unsqueeze(1), 1)

            y_one_hot = torch.clamp(y_one_hot, EPS, 1-EPS).to(device)

            results = net(pts, y_one_hot, mask, False)

            recon_loss, kld_loss = loss_funtion(results, pts, mask)
            loss = recon_loss+kld_loss

            error = ((torch.abs(pts-results["x"])*mask)*std).mean().item()
            total_error += error


            total_loss.append(loss.item())


            # reconstruction = (results["x"]*mask).cpu().numpy()+m

            # for s in reconstruction:
            #     recon.append(s)



            #if i == len(dataloader)-1:
            
            for a, p in enumerate(pts):
            #    p = (p).cpu().numpy()
                p =( p * mask[a]*std ).cpu().numpy() + m_data
                
                visulization( p, checkpoint+'/sub_'+str(a)+data_list[label[a]]+'_gt.npy')

            
            z = results['mean']
            for i in range(6):
                
                y = torch.ones_like(label).to(device) * i
                y_one_hot = torch.zeros(y.shape[0], 6)

                y_one_hot.scatter_(1, y.type(torch.LongTensor).unsqueeze(1), 1)

                y_one_hot = torch.clamp(y_one_hot, EPS, 1-EPS).to(device)*2
                

            
                disenRe = net.sample(pts, z, y_one_hot, mask)
                disenResults = (disenRe*std).cpu().numpy()+m_data
                disenResults = disenResults * mask.cpu().numpy()
                
                for n, sub in enumerate(disenResults):
                   
                    recon_results[i].append(disenRe[n])   #Calculate fid

                    if i == len(dataloader)-1:
                        visulization( sub, checkpoint+data_list[i]+'_'+str(n)+'.npy')



    test_results["error"] = total_error/len(dataloader)
    test_results["loss"] = np.mean(total_loss)
    test_results["reconstruction"] = np.array(recon)
    
    return test_results, recon_results, ground_truth

def Displacement(subject):
    # subject (606, 100, 249) 

    masks = []


    data = []
    displace = []
    for sub in subject:

        mask = np.ones(num_frame)
        disp = np.zeros((num_frame, num_pts*3))
        for i, frame in enumerate(sub):

            if i == num_frame:
                break


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
num_frame = 100
num_pts = 83
z = 50
c = 6
epochs = 10000
lr = 1e-6
weight_decay = 0.0001
label_size = 6
latent_size = 64
batch_size = 8
hidden_unit = 256
num_head = 4
dropout = 0.2
num_layers = 4
'''
checkpoint = './results/MUG_BU/'
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
'''
data_path = 'dataset.pkl'
dataset = np.load(data_path, allow_pickle=True)
subject = np.array(dataset["data"])
labels =  np.array(dataset["labels"])
#subject[:, 0, :] = np.zeros((606,249))
subject_ = np.reshape(subject, (606*100 , 249))
m = subject_.mean(0)
std = subject_.std(0)

subject = (subject_-m)/std
subject = np.reshape(subject, (606, 100 , 249))

labels = np.array(labels)
subject = np.array(subject)
print(subject.shape)
print(labels.shape)
# data_path = 'dataset.pkl'
# dataset = np.load(data_path, allow_pickle=True)
# subject = np.array(dataset["data"])
# labels =  np.array(dataset["labels"])
# subject = np.reshape(subject, (606, 100 , 249))


subject = np.reshape(subject, (-1, num_frame , num_pts*3))
displace, masks, data = Displacement(subject)
m_data = data.mean(0)
std = data.std(0)

trainsformed_data = (displace-m_data)/std
std = torch.tensor(std).to(device)


random_seeds = 2021
test_size = 0.2
best_loss = 1000000
net = DisenFace(3*num_pts, num_frame, label_size, hidden_unit, latent_size, num_head,num_layers, dropout).to(device)

skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state = random_seeds) 
n = 0
checkpoint = './results/test/'

fid_record = np.zeros((6, 5))
for train_index, test_index in skf.split(subject, labels):
    n += 1
    test_data = trainsformed_data[test_index]
    test_labels = labels[test_index]
    test_mask = masks[test_index]
    print("number of test data:", test_data.shape[0])
    test_dataset = FaceData(test_data, test_labels, test_mask)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=4)
    net = torch.load(checkpoint + '/model_'+str(n)+'.pt')
    test_results, recon, gt = test_step(net,  test_loader)
    print(" test: error {},  loss {}".format( test_results["error"], test_results["loss"]))


    #calculate FID for each class:
    classifier = torch.load("./evaluate/classfier_"+str(n)+'.pt').to(device)

    for i in range(6):
        data = recon[i]
        ground_truth = gt[i]

        feature = []
        for d in data:
            d = d.unsqueeze(0)
            _, f = classifier(d)
            feature.append(f.squeeze().detach().cpu().numpy())

        gt_feature = []
        for g in ground_truth:
            g = g.unsqueeze(0)
            
            _, f = classifier(g)
            gt_feature.append(f.squeeze().detach().cpu().numpy())

        fid = calculate_frechet_distance(np.array(feature), np.array(gt_feature))

        fid_record[i, n-1] = fid


print("fid score")
print("======={}===={}===={}===={}===={}===={}==".format(str(1), str(2), str(3), str(4), str(5), str(6)))
print("mean==={}===={}===={}===={}===={}===={}==".format(round(np.mean(fid_record[0]), 3), round(np.mean(fid_record[1]),3),
                                                     round(np.mean(fid_record[2]), 3), round(np.mean(fid_record[3]), 3), 
                                                     round(np.mean(fid_record[4]),3), round(np.mean(fid_record[5]),3)))
print("std===={}===={}===={}===={}===={}===={}==".format(round(np.std(fid_record[0]), 3), round(np.std(fid_record[1]),3),
                                                     round(np.std(fid_record[2]), 3), round(np.std(fid_record[3]), 3), 
                                                     round(np.std(fid_record[4]),3), round(np.std(fid_record[5]),3)))

