
function [alpha_hat_sm_zeros,alpha_hat_sm_nonzeros,I_zeros_index,I_nonzeros_index,Bhat_matrix_sm_zeros,Bhat_matrix_sm_nonzeros]=detector(Bhat_matrix_sm,alpha_hat_sm)
[m,n]=size(Bhat_matrix_sm);
I=zeros(m,1);
for i=1:m
 if sum(Bhat_matrix_sm(i,:))==0
   I(i,1)=1;
 end
end 
I_zeros_index=find(I);
I_nonzeros_index=find(~I);
[m_zeros,n_zeros]=size(I_zeros_index);
Bhat_matrix_sm_zeros=Bhat_matrix_sm(I_zeros_index,:);
[m_nonzeros,n_nonzeros]=size(I_nonzeros_index);
Bhat_matrix_sm_nonzeros=Bhat_matrix_sm(I_nonzeros_index,:);
alpha_hat_sm_zeros=alpha_hat_sm(I_zeros_index,:);
alpha_hat_sm_nonzeros=alpha_hat_sm(I_nonzeros_index,:);
