# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 00:17:57 2020

@author: Dian
"""
import time
import datetime
import scipy.io as sio
import tifffile
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from scipy import misc
import cv2
from scipy.io import loadmat
import sys
from matplotlib.pyplot import *
from numpy import *
from torch.nn import functional 
from torch import nn
import torch
import os
import torch.utils.data as data
import metrics
import h5py
class HSI_MSI_Data(data.Dataset):
    def __init__(self,train_hrhs_all,train_hrms_all,train_lrhs_all):
        self.train_hrhs_all  = train_hrhs_all
        self.train_hrms_all  = train_hrms_all
        self.train_lrhs_all  = train_lrhs_all
    def __getitem__(self, index):
        train_hrhs = self.train_hrhs_all[index, :, :, :]
        train_hrms= self.train_hrms_all[index, :, :, :]
        train_lrhs = self.train_lrhs_all[index, :, :, :]
        return train_hrhs, train_hrms, train_lrhs

    def __len__(self):
        return self.train_hrhs_all.shape[0]
    
def warm_lr_scheduler(optimizer, init_lr1,init_lr2, warm_iter,iteraion, lr_decay_iter, max_iter, power):
    if iteraion % lr_decay_iter or iteraion > max_iter:
        return optimizer
    if iteraion < warm_iter:
        lr=init_lr1+iteraion/warm_iter*(init_lr2-init_lr1)
    else:
      lr = init_lr2*(1 - (iteraion-warm_iter)/(max_iter-warm_iter))**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def fspecial(func_name,kernel_size,sigma):
    if func_name=='gaussian':
        m=n=(kernel_size-1.)/2.
        y,x=ogrid[-m:m+1,-n:n+1]
        h=exp( -(x*x + y*y) / (2.*sigma*sigma) )
        h[ h < finfo(h.dtype).eps*h.max() ] = 0
        sumh=h.sum()
        if sumh!=0:
            h/=sumh
        return h

def Gaussian_downsample(x,psf,s):
    y=np.zeros((x.shape[0],int(x.shape[1]/s),int(x.shape[2]/s)))
    if x.ndim==2:
        x=np.expand_dims(x,axis=0)
    for i in range(x.shape[0]):
        x1=x[i,:,:]
        x2=signal.convolve2d(x1,psf, boundary='symm',mode='same')
        y[i,:,:]=x2[0::s,0::s]
    return y
        
def poly_lr_scheduler(optimizer, init_lr, iteraion, lr_decay_iter=1, max_iter=100, power=0.9):
    if iteraion % lr_decay_iter or iteraion > max_iter:
        return optimizer

    lr = init_lr*(1 - iteraion/max_iter)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
        
def adjust_learning_rate(optimizer, epoch, milestones=None):
    def to(epoch):
        if epoch <= args.warmup:
            return 1
        elif args.warmup < epoch <= milestones[0]:
            return 0
        for i in range(1, len(milestones)):
            if milestones[i - 1] < epoch <= milestones[i]:
                return i
        return len(milestones)
 
    n = to(epoch)
 
    global lr
    lr = args.base_lr * (0.2 ** n)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def create_F():
     F =np.array([[2.0,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  6, 11, 17, 21, 22, 21, 20, 20, 19, 19, 18, 18, 17, 17],
         [1,  1,  1,  1,  1,  1,  2,  4,  6,  8, 11, 16, 19, 21, 20, 18, 16, 14, 11,  7,  5,  3,  2, 2,  1,  1,  2,  2,  2,  2,  2],
         [7, 10, 15, 19, 25, 29, 30, 29, 27, 22, 16,  9,  2,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]])
     for band in range(3):
        div = np.sum(F[band][:])
        for i in range(31):
            F[band][i] = F[band][i]/div;
     return F


def  dataset_input(data_put,downsample_factor):
  if data_put=='pavia': 
        F=loadmat('.\data\R.mat')
        F=F['R']
        F=F[:,0:-10]
        for band in range(F.shape[0]):
            div = np.sum(F[band][:])
            for i in range(F.shape[1]):
                F[band][i] = F[band][i]/div;
        R=F
        HRHSI=tifffile.imread('.\data\original_rosis.tif')
        HRHSI=HRHSI[0:-10,0:downsample_factor**2*int(HRHSI.shape[1]/downsample_factor**2),0:downsample_factor**2*int(HRHSI.shape[2]/downsample_factor**2)]
        HRHSI=HRHSI/np.max(HRHSI)
  elif data_put=='Chikusei':
        mat=h5py.File('.\data\Chikusei.mat')
        HRHSI=mat['chikusei']
        mat1=sio.loadmat('.\data\Chikusei_data.mat')
        R=mat1['R']
        R=R[0:8:2,:]
        HRHSI=HRHSI[:,100:900,100:900]
        HRHSI=np.transpose(HRHSI,(0,2,1))
        x1=np.max(HRHSI)
        x2=np.min(HRHSI)
        x3=-x2/(x1-x2)
        HRHSI=HRHSI/(x1-x2)+x3
  elif data_put=='houston':
        mat=sio.loadmat('.\data\Houston.mat')
        HRHSI=mat['Houston']
        HRHSI=np.transpose(HRHSI,(2,0,1))
        HRHSI=HRHSI[:,0:336,100:900]
        x1=np.max(HRHSI)
        x2=np.min(HRHSI)
        x3=-x2/(x1-x2)
        HRHSI=HRHSI/(x1-x2)+x3
        R=np.zeros((4,HRHSI.shape[0]));
        for i in range(R.shape[0]):
          R[i,36*i:36*(i+1)]=1/36.0 
  else:
        sys.exit(0)
  return HRHSI,R

def  savedata(dataset,R,training_size,stride,downsample_factor,PSF,num):
         if dataset=='CAVE':
              path='D:\我的代码\高光谱集数据\CAVE\\'
         elif dataset=='Harvard':
              path='D:\我的代码\高光谱集数据\Harvard\\'
         imglist=os.listdir(path)
         train_hrhs=[]
         train_hrms=[]
         train_lrhs=[]
         for i in range(num):
            img=loadmat(path+imglist[i])
            img1=img["b"]
            HRHSI=np.transpose(img1,(2,0,1))
            HSI_LR=Gaussian_downsample(HRHSI,PSF,downsample_factor)
            MSI=np.tensordot(R,  HRHSI, axes=([1], [0]))
            for j in range(0, HRHSI.shape[1]-training_size+1, stride):
                for k in range(0, HRHSI.shape[2]-training_size+1, stride):
                    temp_hrhs = HRHSI[:,j:j+training_size, k:k+training_size]
                    temp_hrms = MSI[:,j:j+training_size, k:k+training_size]
                    temp_lrhs = HSI_LR[:,int(j/downsample_factor):int((j+training_size)/downsample_factor), int(k/downsample_factor):int((k+training_size)/downsample_factor)]
                    train_hrhs.append(temp_hrhs)
                    train_hrms.append(temp_hrms)
                    train_lrhs.append(temp_lrhs)
            sio.savemat(dataset+'.mat',{'hrhs':train_hrhs,'ms':train_hrms,'lrhs':train_lrhs})
            print(train_hrhs.shape, train_hrms.shape,train_lrhs.shape)
    



class HSI_MSI_Data1(data.Dataset):
    def __init__(self,dataset):
         mat=h5py.File(dataset+'.mat')
         self.train_hrhs_all  = mat['hrhs']
         self.train_hrms_all  = mat['ms']
         self.train_lrms_all  = mat['lrhs']
    def __getitem__(self, index):
        train_hrhs = torch.Tensor(self.train_hrhs_all[index, :, :, :])
        train_hrms= torch.Tensor(self.train_hrms_all[index, :, :, :])
        train_lrhs= torch.Tensor(self.train_lrhs_all[index, :, :, :])
        return train_hrhs, train_hrms,train_lrhs
    def __len__(self):
        return self.train_hrhs_all.shape[0]
class CNN(nn.Module):
    def __init__(self,a,b):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(        
            nn.Conv2d(a+b, 64-b, 3, 1, 1),     
            nn.LeakyReLU(negative_slope=0.2, inplace=False), 
            )
        self.conv2 = nn.Sequential(        
        nn.Conv2d(64, 64-b, 3, 1, 1),     
        nn.LeakyReLU(negative_slope=0.2, inplace=False), 
        )
        self.conv3 = nn.Sequential(        
        nn.Conv2d(64, 64-b, 3, 1, 1),     
        nn.LeakyReLU(negative_slope=0.2, inplace=False), 
        )
        self.conv4 = nn.Sequential(        
        nn.Conv2d(64, a, 3, 1, 1),     
        )
#        self.squeeze = nn.AdaptiveAvgPool2d(1)
#        self.excitation = nn.Sequential(
#         nn.Linear(10, 2),
#         nn.ReLU(inplace=True),
#         nn.Linear(2, 10),
#         nn.Sigmoid()
#         )
        basecoeff = torch.Tensor([[-4.63495665e-03, -3.63442646e-03,  3.84904063e-18,
                     5.76678319e-03,  1.08358664e-02,  1.01980790e-02,
                    -9.31747402e-18, -1.75033181e-02, -3.17660068e-02,
                    -2.84531643e-02,  1.85181518e-17,  4.42450253e-02,
                     7.71733386e-02,  6.70554910e-02, -2.85299239e-17,
                    -1.01548683e-01, -1.78708388e-01, -1.60004642e-01,
                     3.61741232e-17,  2.87940558e-01,  6.25431459e-01,
                     8.97067600e-01,  1.00107877e+00,  8.97067600e-01,
                     6.25431459e-01,  2.87940558e-01,  3.61741232e-17,
                    -1.60004642e-01, -1.78708388e-01, -1.01548683e-01,
                    -2.85299239e-17,  6.70554910e-02,  7.71733386e-02,
                     4.42450253e-02,  1.85181518e-17, -2.84531643e-02,
                    -3.17660068e-02, -1.75033181e-02, -9.31747402e-18,
                     1.01980790e-02,  1.08358664e-02,  5.76678319e-03,
                     3.84904063e-18, -3.63442646e-03, -4.63495665e-03]])
        coeff = torch.mm(basecoeff.T, basecoeff)
        coeff = torch.Tensor(coeff)
        coeff = torch.unsqueeze(coeff, 0)
        coeff = torch.unsqueeze(coeff, 0)
        self.coeff = torch.repeat_interleave(coeff, a,0)
        psf=fspecial('gaussian', 7, 3)
        psf = torch.Tensor(psf)
        psf = torch.unsqueeze(psf, 0)
        psf = torch.unsqueeze(psf, 0)
        self.psf = torch.repeat_interleave(psf, a,0)
    def forward(self, x,y):
        def Upsample_4(coeff,inputs):     
                    _,c, h, w= inputs.shape
                    outs = functional.conv_transpose2d(inputs, coeff.cuda(), bias=None, stride=4, padding=21, output_padding=1, groups=c, dilation=1)
                    return outs

        x1=Upsample_4(self.coeff,x)    
#        x1=functional.interpolate(x, size=None, scale_factor=4, mode='bilinear', align_corners=None)   
        x2 = torch.cat((x1,y),1)
        x2 = torch.cat((self.conv1(x2),y),1)
        x2 = torch.cat((self.conv2(x2),y),1)
        x2 = torch.cat((self.conv3(x2),y),1)
        x3 = self.conv4(x2)
        return x3+x1
class CNNF(nn.Module):
    def __init__(self,a,b):
        super(CNNF, self).__init__()
        self.conv3 = nn.Sequential(        
            nn.Conv2d(a+b, 64, 3, 1, 1),     
            nn.LeakyReLU(negative_slope=0.2, inplace=False), 
            nn.Conv2d(64, 64, 3, 1, 1), 
            nn.LeakyReLU(negative_slope=0.2, inplace=False), 
            nn.Conv2d(64, 64, 3, 1, 1),     
            nn.LeakyReLU(negative_slope=0.2, inplace=False), 
            nn.Conv2d(64, a, 3, 1, 1),
         )
         
        basecoeff = torch.Tensor([[-4.63495665e-03, -3.63442646e-03,  3.84904063e-18,
                     5.76678319e-03,  1.08358664e-02,  1.01980790e-02,
                    -9.31747402e-18, -1.75033181e-02, -3.17660068e-02,
                    -2.84531643e-02,  1.85181518e-17,  4.42450253e-02,
                     7.71733386e-02,  6.70554910e-02, -2.85299239e-17,
                    -1.01548683e-01, -1.78708388e-01, -1.60004642e-01,
                     3.61741232e-17,  2.87940558e-01,  6.25431459e-01,
                     8.97067600e-01,  1.00107877e+00,  8.97067600e-01,
                     6.25431459e-01,  2.87940558e-01,  3.61741232e-17,
                    -1.60004642e-01, -1.78708388e-01, -1.01548683e-01,
                    -2.85299239e-17,  6.70554910e-02,  7.71733386e-02,
                     4.42450253e-02,  1.85181518e-17, -2.84531643e-02,
                    -3.17660068e-02, -1.75033181e-02, -9.31747402e-18,
                     1.01980790e-02,  1.08358664e-02,  5.76678319e-03,
                     3.84904063e-18, -3.63442646e-03, -4.63495665e-03]])
        coeff = torch.mm(basecoeff.T, basecoeff)
        coeff = torch.Tensor(coeff)
        coeff = torch.unsqueeze(coeff, 0)
        coeff = torch.unsqueeze(coeff, 0)
        self.coeff = torch.repeat_interleave(coeff, 10,0)
        psf=fspecial('gaussian', 7, 3)
        psf = torch.Tensor(psf)
        psf = torch.unsqueeze(psf, 0)
        psf = torch.unsqueeze(psf, 0)
        self.psf = torch.repeat_interleave(psf, 10,0)
    def forward(self, x,y):
        def Upsample_4(coeff,inputs):     
                    _,c, h, w= inputs.shape
                    outs = functional.conv_transpose2d(inputs, coeff.cuda(), bias=None, stride=4, padding=21, output_padding=1, groups=c, dilation=1)
                    return outs
        x1=Upsample_4(self.coeff,x)    
#        x1=functional.interpolate(x, size=None, scale_factor=4, mode='bilinear', align_corners=None)   
        x2 = torch.cat((x1,y),1)
        x2 = self.conv3(x2)
        return x2+x1