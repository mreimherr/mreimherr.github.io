% Y are coefficients of data in orthonormal basis expansion
% we reccommend that Y be based on PCA. Use FPCA from pace package
% for sparse FDA, or FDA package for high frequency.
% X is the design matrix for scalar covariates
% H_all is a array containing all of the H matrices
% H_all should include the identity (error) as it will not be added
% Returns an array of the coefficients for the estimated covariance.
% Iter are the number of desired iterations.
% Trunc = true truncates negative eigenvalues.

function [C,Herit,SE] = FMINQUE(Y,X,H_all,Iter,Trunc)
I = size(H_all,3);
[N M] = size(Y);
if isempty(X) 
    %J = 0;
    %A = eye(N);
    Yp=Y;
    Np=N;
    Hp = H_all;
else
    J = size(X,2);
    [U,~,~] = svd(X);
    A = U(:,(J+1):end)';

    Yp = A*Y;
    Np = size(Yp,1);
    Hp = zeros(Np,Np,I);
    for i = 1:I,
        %b = 1 + (i-1)*N;
        Hp(:,:,i) = A*H_all(:,:,i)*A';
    end
end


B = zeros(Np,Np,I);



Hp(:,:,I) = diag(ones(Np,1));

%We start the iteration process here%
c_all = zeros(I,I,Iter+1);
c_all(:,:,1) = ones(I);



for i_iter = 1:Iter,
    

c_v = c_all(:,:,i_iter);

B_all = zeros(Np*(I),Np);
[v lamb] = eig(c_v);
R = zeros(size(Hp));
for i = 1:(I),
    for j = 1:(I),
        R(:,:,i) = R(:,:,i) + v(j,i)*Hp(:,:,j);  
    end
end

for i = 1:(I),
    [B_tmp,~] = gmres(@(B)MQSub(B,R,lamb,I),reshape(Hp(:,:,i),[],1));
    %[B_tmp,~] = symmlq(@(B)MQSub(B,R,lamb,I),reshape(Hp(:,:,i),[],1),[],200);
    B(:,:,i) = reshape(B_tmp,Np,Np);
end

for i = 1:(I),
    b = 1 + (i-1)*Np;
    B_all(b:(b+Np-1),:) = B(:,:,i);
end

D = zeros(I,I);
for i = 1:(I),
    for j = 1:(I),
        D(i,j) = trace(Hp(:,:,i)*B(:,:,j));
    end
end

Dinv = inv(D);
Ap = zeros(Np,Np,I);
for i = 1:(I),
    Ap(:,:,i) = kron(Dinv(:,i)',diag(ones(Np,1)))*B_all;
end

C = zeros(M,M,I);
Herit = zeros(I,1);
Ht = zeros(I,M);

for i = 1:(I),
    C(:,:,i) = Yp'*Ap(:,:,i)*Yp;
    if Trunc == true,
        [V_e D_e] = eig(C(:,:,i));
        D_e2 = diag(D_e);
        D_e2(D_e2 < 0) = 0;
        C(:,:,i) = V_e * diag(D_e2) * V_e';
    end
    Herit(i) = sum(diag(C(:,:,i)))*sum(diag(Hp(:,:,i)));
    Ht(i,:) = diag(C(:,:,i))*sum(diag(Hp(:,:,i)));
end

Herit = Herit/sum(Herit);

for i = 1:(I),
    for j = 1:(I),
        %c_all(i,j,i_iter+1) = sum(sum(C(:,:,i) .* C(:,:,j)))/(M^2);
        c_all(i,j,i_iter+1) = sum(sum(C(:,:,i) .* C(:,:,j))); %changed since Y should be from basis
    end
end

c_all(:,:,i_iter+1) = c_all(:,:,i_iter+1)/sum(sum(c_all(:,:,i_iter+1)))*(I)^2;


end

for j =  1:M,
    Ht(:,j) = Ht(:,j) /sum(Ht(:,j));
end


% Here we are computing the variance.
cz = zeros(I,I);
SigZ = zeros(I,I);
Z = zeros(I,1);
grad_h = zeros(I,I);
for i=1:(I),
    Z(i) = trace(C(:,:,i))/M;
    for(j = 1:I)
        cz(i,j) = sum(sum(C(:,:,i) .* C(:,:,j)))/(M^2);
    end
end
[v lamb] = eig(cz);
R = zeros(size(Hp));
for i = 1:(I),
    for j = 1:(I),
        R(:,:,i) = R(:,:,i) + v(j,i)*Hp(:,:,j);  
    end
end
for i=1:(I),
    for j=1:(I),
        
        tmp = 0;
        for l=1:(I)
            for ll = 1:I,
            tmp = tmp + cz(l,ll)*Ap(:,:,i)*Hp(:,:,l)*Ap(:,:,j)*Hp(:,:,ll);
            end
            %tmp = tmp + lamb(l)*Ap(:,:,i)*R(:,:,l)*Ap(:,:,j)*R(:,:,l);
        end
        SigZ(i,j) = 2*trace(tmp);
        grad_h(i,j) = -Z(i)/sum(Z)^2;
    end
    grad_h(i,i) =  (sum(Z)-Z(i))/sum(Z)^2;
end

SE = sqrt(diag(grad_h*SigZ*(grad_h')));

end


function [y] = MQSub(B,R,lamb,I)
    Ntmp = size(R,1);
    B = reshape(B,[],Ntmp);
    y = zeros(Ntmp);
    thresh = mean(abs(diag(lamb)));
    for i = 1:I,
        %mat1_tmp = Hp(:,:,i)*B;
        %for j = 1:I,
        %    y = y + c_v(i,j)*mat1_tmp*Hp(:,:,j);
        %end
        if abs(lamb(i,i)) > (0.0001*thresh)
            y = y+ lamb(i,i)*R(:,:,i)*B*R(:,:,i);
        end
    end
    y = reshape(y,[],1);
end





