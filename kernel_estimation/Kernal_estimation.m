function   [R,B]    =   Kernal_estimation(HSI, MSI,size_B)
[M,N,L]=size(MSI);
R=ones(size(MSI,3),size(HSI,3));
R=R/size(HSI,3);
% MSI_2D=hyperConvert2D(MSI);
HSI_2D=hyperConvert2D(HSI);
s0=1;
sf=size(MSI,1)/size(HSI,1);
iter=20;
% size_B=7;
 B=  abs(randn(size_B,size_B));
mu=0;
% B=ones(7,7);
B=B/sum(B(:));
for i=1:iter
     fft_B     =    psf2otf(B,[M N]);
  MSI_BS= Gaussian_downsample(MSI, fft_B, sf,s0);
  MSI_BS=hyperConvert2D(MSI_BS);
 R=R_update2(MSI_BS,HSI_2D,R,mu);
 
 R_HSI=R*HSI_2D;
 R_HSI=hyperConvert3D(R_HSI,M/sf,N/sf);
 R_HSI_up=zeros(M,N,L);
R_HSI_up(s0:sf:end,s0:sf:end,:)=R_HSI;
  B=B_update2(R_HSI_up, MSI,size_B,sf,B,mu);
end
end

% function   R    =   R_update(MSI_BS, HSI_2D,R,mu)
% R=(MSI_BS*HSI_2D'+mu*R)/(HSI_2D*HSI_2D'+mu*eye(size(HSI_2D,1)));
% end
% function   B    =   B_update(R_HSI_up, MSI,size_B,sf,B,mu)
% ymymt = zeros(size_B*size_B, size_B*size_B);
% rtyhymt = zeros(size_B*size_B, 1);
% for i= 2*sf+1:sf:size(MSI,1)- floor((size_B-1)/2)-1 
%     for j=2*sf+1:sf:size(MSI,2)- floor((size_B-1)/2)-1
%         for k=1:size(MSI,3)
%             d_size=floor((size_B-1)/2);
%       image_patch=MSI(i-d_size:i+d_size,j-d_size:j+d_size,k);
%        ymymt = ymymt + image_patch(:)*image_patch(:)';
%         rtyhymt = rtyhymt + image_patch(:)*R_HSI_up(i, j, k);
%         end
%     end   
% end
% B=(ymymt+mu*eye(size(ymymt,1)))\(rtyhymt+mu*B(:));
% % B=B/sum(B(:));
% B=projsplx(B);
% B=reshape(B,size_B,size_B);
% end
