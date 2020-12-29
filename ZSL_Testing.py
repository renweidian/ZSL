# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 21:47:13 2020

@author: Dian
"""
from model1 import *
#model1='nosubsapce'
#data2='houston'
data2='pavia'
PSF = fspecial('gaussian', 7, 3)
downsample_factor=4
[HRHSI,R]=dataset_input(data2,downsample_factor)
HSI0=Gaussian_downsample(HRHSI,PSF,downsample_factor)
SNRh=30;
sigma = np.sqrt(np.sum(HSI0**2)/(10**(SNRh/10))/(HSI0.shape[0]*HSI0.shape[1]*HSI0.shape[2]));
HSI0=HSI0+ 0*np.random.randn(HSI0.shape[0],HSI0.shape[1],HSI0.shape[2])
MSI0=np.tensordot(R,  HRHSI, axes=([1], [0]))
sigma = np.sqrt(np.sum(MSI0**2)/(10**(SNRh/10))/(MSI0.shape[0]*MSI0.shape[1]*MSI0.shape[2]));
MSI0=MSI0+ 0*np.random.randn(MSI0.shape[0],MSI0.shape[1],MSI0.shape[2])

p=10
HSI3=HSI0.reshape(HSI0.shape[0],-1)
U0,S,V=np.linalg.svd(np.dot(HSI3,HSI3.T))
U0=U0[:,0:int(p)]
HSI0_Abun=np.tensordot(U0.T,  HSI0, axes=([1], [0]))

net2=CNN(p,MSI0.shape[0]).cuda() 
net2.load_state_dict(torch.load(data2+'.pkl'))
MSI_1=torch.Tensor(np.expand_dims(MSI0,axis=0))
HSI_1=torch.Tensor(np.expand_dims(HSI0_Abun,axis=0))
starttime = time.time()
abudance = net2(HSI_1.cuda(),MSI_1.cuda())
abudance=abudance.cpu().detach().numpy()
abudance1=np.squeeze(abudance)
Fuse1=np.tensordot(U0, abudance1, axes=([1], [0]))
endtime = time.time()
print (endtime - starttime)

  