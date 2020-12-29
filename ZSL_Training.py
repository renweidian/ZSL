# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 12:56:22 2020

@author: Dian
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 13:20:42 2020

@author: Dian
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 23 22:06:42 2020

@author: Dian
"""
from model1 import *

 
if __name__ == '__main__': 
    p=10
    stride=1
    training_size=32
    downsample_factor=4
    LR=1e-3
    EPOCH=400
    BATCH_SIZE=64 
    loss_optimal=1.75
    init_lr1=1e-4
    init_lr2=5e-4
    decay_power=1.5
    data2='pavia'
    [HSI0,MSI0,HRHSI]=dataset_input(data2,downsample_factor)
    maxiteration=2*math.ceil(((HRHSI.shape[1]/downsample_factor-training_size)//stride+1)*((HRHSI.shape[2]/downsample_factor-training_size)//stride+1)/BATCH_SIZE)*EPOCH
    print(maxiteration)
    warm_iter=math.floor(maxiteration/40)
    print(maxiteration)
    
    HSI3=HSI0.reshape(HSI0.shape[0],-1)
    U0,S,V=np.linalg.svd(np.dot(HSI3,HSI3.T))
    U0=U0[:,0:int(p)]
    HSI0_Abun=np.tensordot(U0.T,  HSI0, axes=([1], [0]))
    augument=[0]
    HSI_aug=[]
    HSI_aug.append(HSI0)
    MSI_aug=[]
    MSI_aug.append(MSI0)
    U=U0
    train_hrhs = []
    train_hrms = []
    train_lrhs= []
    Response=h5py.File('Estimated_Responses.mat')
    PSF_Estimated= np.transpose(Response['B'])
    R_Estimated= np.transpose(Response['R'])
    for j in augument:       
        HSI = cv2.flip(HSI0, j)
#        MSI_aug.append(MSI0)
        HSI_aug.append(HSI)
    for j in range(len(HSI_aug)):
        HSI = HSI_aug[j]
#        MSI = MSI_aug[j]
        HSI_Abun=np.tensordot(U.T,  HSI, axes=([1], [0]))
        HSI_LR_Abun=Gaussian_downsample(HSI_Abun,PSF_Estimated,downsample_factor)
        MSI_LR=np.tensordot(R_Estimated,  HSI, axes=([1], [0])) 
#        MSI_LR=Gaussian_downsample(MSI,PSF,downsample_factor) 
        for j in range(0, HSI_Abun.shape[1]-training_size+1, stride):
            for k in range(0, HSI_Abun.shape[2]-training_size+1, stride):
                
                temp_hrhs = HSI[:,j:j+training_size, k:k+training_size]
                temp_hrms = MSI_LR[:,j:j+training_size, k:k+training_size]
                temp_lrhs = HSI_LR_Abun[:,int(j/downsample_factor):int((j+training_size)/downsample_factor), int(k/downsample_factor):int((k+training_size)/downsample_factor)]
                
                train_hrhs.append(temp_hrhs)
                train_hrms.append(temp_hrms)
                train_lrhs.append(temp_lrhs)
               

    train_hrhs=torch.Tensor(train_hrhs)
    train_lrhs=torch.Tensor(train_lrhs)
    train_hrms=torch.Tensor(train_hrms)
    print(train_hrhs.shape, train_hrms.shape, train_lrhs.shape)
    train_data=HSI_MSI_Data(train_hrhs,train_hrms,train_lrhs)
    train_loader = data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    
    cnn=CNN(p,MSI0.shape[0]).cuda() 
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
#    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
    loss_func = nn.L1Loss(reduction='mean') 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.2, last_epoch=-1)
    for m in cnn.modules():
      if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)    
    MSI_1=torch.Tensor(np.expand_dims(MSI0,axis=0))
    HSI_1=torch.Tensor(np.expand_dims(HSI0_Abun,axis=0))
    step=0
    loss_list=[]
    U22=torch.Tensor(U0)
    for epoch in range(EPOCH): 
        for step1, (a1, a2,a3) in enumerate(train_loader): 
            cnn.train()
            lr=warm_lr_scheduler(optimizer, init_lr1,init_lr2, warm_iter,step, lr_decay_iter=1,  max_iter=maxiteration, power=decay_power)
            step=step+1
            output = cnn(a3.cuda(),a2.cuda()) 
            Fuse1=torch.tensordot(U22.cuda(), output, dims=([1], [1])) 
#            Fuse1=torch.clamp(Fuse1,0,1)
            Fuse1=torch.Tensor.permute(Fuse1,(1,0,2,3))
#            a1=torch.tensordot(U22.cuda(), a1.cuda(), dims=([1], [1]))  
            loss = loss_func(Fuse1, a1.cuda())
#            loss = loss_func(output, a1.cuda())
            optimizer.zero_grad()           
            loss.backward()               
            optimizer.step() 
#        scheduler.step()
        cnn.eval()
        with torch.no_grad():
            abudance = cnn(HSI_1.cuda(),MSI_1.cuda())
            abudance=abudance.cpu().detach().numpy()
            abudance1=np.squeeze(abudance)
        Fuse2=np.tensordot(U0, abudance1, axes=([1], [0]))
        sum_loss,psnr_=metrics.rmse1(np.clip(Fuse2,0,1),HRHSI)
        if sum_loss<loss_optimal:
           loss_optimal=sum_loss
#        torch.save(cnn.state_dict(), data2+'.pkl') 
        loss_list.append(sum_loss)
        print(epoch,lr,sum_loss,psnr_)
        

    torch.save(cnn, data2+'.pkl') 
