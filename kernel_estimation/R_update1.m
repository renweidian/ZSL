function   R    =   R_update1(MSI_BS, HSI_2D,R,mu)
% R=(MSI_BS*HSI_2D'+mu*R)/(HSI_2D*HSI_2D'+mu*eye(size(HSI_2D,1)));
gama=1/norm(HSI_2D*HSI_2D','fro');
% gama=1e-5;
R=R-gama*(R*HSI_2D-MSI_BS)*HSI_2D';
R(R<0)=0;
end