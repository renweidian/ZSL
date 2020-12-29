function   B    =   B_update2(R_HSI_up, MSI,size_B,sf,B,mu)
ymymt = zeros(size_B*size_B, size_B*size_B);
rtyhymt = zeros(size_B*size_B, 1);
for i= 1*sf+1:sf:size(MSI,1)- floor((size_B-1)/2)-1 
    for j=1*sf+1:sf:size(MSI,2)- floor((size_B-1)/2)-1
        for k=1:size(MSI,3)
            d_size=floor((size_B-1)/2);
      image_patch=MSI(i-d_size:i+d_size,j-d_size:j+d_size,k);
       ymymt = ymymt + image_patch(:)*image_patch(:)';
        rtyhymt = rtyhymt + image_patch(:)*R_HSI_up(i, j, k);
        end
    end   
end
mu=1e-3;
B=B(:);
V=B;
G=zeros(size(B));
for i=1:20
    B=(ymymt+mu*eye(size(ymymt,1)))\(rtyhymt+mu*V-G/2);
    V=B+G/(2*mu);
   V=projsplx(V(:));
    G=G+2*mu*(B-V);
end
B=reshape(V,size_B,size_B);




