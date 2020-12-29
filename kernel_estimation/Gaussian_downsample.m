function   x    =   Gaussian_downsample(z, fft_B, sf,s0)
[M, N,L]         =    size(z);



    x           =    zeros(round(M/sf),round(N/sf),L);    
    for  i  = 1 : L
        Hz         =    real( ifft2(fft2( z(:,:,i) ).*fft_B) );
        t          =    Hz(s0:sf:end, s0:sf:end);
        x(:,:,i)     =    t;
    end
end


