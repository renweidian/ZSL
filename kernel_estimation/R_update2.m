function   R    =   R_update2(MSI_BS, HSI_2D,R,mu)


mu=1e-3;
V=R;
G=zeros(size(R));
G1=HSI_2D*HSI_2D';
G2=MSI_BS*HSI_2D';
for i=1:100
    R=(G2+mu*V-G/2)/(G1+mu*eye(size(HSI_2D,1)));
    V=R+G/(2*mu);
    V(V<0)=0;
    G=G+2*mu*(R-V);
%     mu=mu*1.1;
% norm(R-V)
end
% a1=norm(R1*HSI_2D-MSI_BS,'fro');
% a2=norm(V*HSI_2D-MSI_BS,'fro');
% % norm(R1-V)/norm(R)
%   R;