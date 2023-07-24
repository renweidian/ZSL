# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 21:47:13 2020

@author: Dian
"""
from model1 import *
data2='pavia'
PSF = fspecial('gaussian', 7, 3)
downsample_factor=4
[HRHSI,R]=dataset_input(data2,downsample_factor)
MSI=np.tensordot(R,  HRHSI, axes=([1], [0]))
HSI=Gaussian_downsample(HRHSI,PSF,downsample_factor)

p=10
HSI3=HSI.reshape(HSI.shape[0],-1)
U0,S,V=np.linalg.svd(np.dot(HSI3,HSI3.T))
U0=U0[:,0:int(p)]
HSI_Abun=np.tensordot(U0.T,  HSI, axes=([1], [0]))

net2=CNN(p,MSI.shape[0]).cuda() 
net2.load_state_dict(torch.load(data2+'1.pkl'))
MSI_1=torch.Tensor(np.expand_dims(MSI,axis=0))
HSI_1=torch.Tensor(np.expand_dims(HSI_Abun,axis=0))
starttime = time.time()
abudance = net2(HSI_1.cuda(),MSI_1.cuda())
abudance=abudance.cpu().detach().numpy()
abudance1=np.squeeze(abudance)
Fuse1=np.tensordot(U0, abudance1, axes=([1], [0]))
endtime = time.time()
print (endtime - starttime)
a,b=metrics.rmse1(np.clip(Fuse1,0,1),HRHSI)
print(a,b)
Fuse1=np.clip(Fuse1,0,1)
Fused_HSI=np.transpose(Fuse1,(1,2,0))

  
