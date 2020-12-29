function   B    =   B_update1(R_HSI_up, MSI,size_B,sf,B,mu)
ymymt = zeros(size_B*size_B, size_B*size_B);
rtyhymt = zeros(size_B*size_B, 1);
for i= 2*sf+1:sf:size(MSI,1)- floor((size_B-1)/2)-1 
    for j=2*sf+1:sf:size(MSI,2)- floor((size_B-1)/2)-1
        for k=1:size(MSI,3)
            d_size=floor((size_B-1)/2);
      image_patch=MSI(i-d_size:i+d_size,j-d_size:j+d_size,k);
       ymymt = ymymt + image_patch(:)*image_patch(:)';
        rtyhymt = rtyhymt + image_patch(:)*R_HSI_up(i, j, k);
        end
    end   
end
gama=1/norm(ymymt,'fro');
B=B(:)-gama*(ymymt*B(:)-rtyhymt);
% B=B/sum(B(:));
B=projsplx(B(:));
B=reshape(B,size_B,size_B);
end