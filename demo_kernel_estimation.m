
clear

addpath(genpath('kernel_estimation'))
S=imread('data\original_rosis.tif');
F=load('data\R.mat');
downsampling_scale=4;
a=downsampling_scale^2*floor(size(S,1)/downsampling_scale^2);
b=downsampling_scale^2*floor(size(S,2)/downsampling_scale^2);
S=double(S);
S=S(1:a,1:b,1:end-10);
S=S/max(S(:));

F=F.R;
  F=F(:,1:end-10);
 for band = 1:size(F,1)
        div = sum(F(band,:));
        for i = 1:size(F,2)
            F(band,i) = F(band,i)/div;
        end
 end


 sizeb=7;
 sag=3;
[M,N,L] = size(S);

%  simulate LR-HSI
S_bar = hyperConvert2D(S);


  psf        =    fspecial('gaussian',sizeb,sag);
%  psf        =    fspecial('average',sizeb);

par.fft_B      =    psf2otf(psf,[M N]);
par.fft_BT     =    conj(par.fft_B);
s0=1;
par.H          =    @(z)H_z(z, par.fft_B, downsampling_scale, [M N],s0 );
par.HT         =    @(y)HT_y(y, par.fft_BT, downsampling_scale,  [M N],s0);
Y_h_bar=par.H(S_bar);

  
SNRh=30;
sigma = sqrt(sum(Y_h_bar(:).^2)/(10^(SNRh/10))/numel(Y_h_bar));
rng(10,'twister')
   Y_h_bar = Y_h_bar+ 0*randn(size(Y_h_bar));
HSI=hyperConvert3D(Y_h_bar,M/downsampling_scale, N/downsampling_scale );
  %  simulate HR-MSI
rng(10,'twister')
Y = F*S_bar;
SNRm=35;
sigmam = sqrt(sum(Y(:).^2)/(10^(SNRm/10))/numel(Y));
Y = Y+ 0*randn(size(Y));
MSI=hyperConvert3D(Y,M,N);

[R,B]=Kernal_estimation(HSI, MSI,sizeb);
a1=csnr(F,R,0,0)
a2=csnr(psf,B,0,0)
save('Estimated_Responses.mat','R','B') 
% norm(psf-B,'fro')/norm(psf)
% norm(F-R,'fro')/norm(R)