% Vector Example of FMINQUE function
% Here we give an example of estimating
% the variance components in a block model.
% We compare to fitlme.

% number of blocks
b = 200;

% observations per block
nb = 2;

% total sample size
n = b*nb;

% points per curve
m = 40;

% to call on FMINQUE we have to give it
% the form of the covariance matrices.
% this gives a great deal of flexibility,
% but is a bit more complicated than 
% just a call to mixed effect model.

% Z denotes mixed effects assignments
Z = repmat(1:b,nb,1);
Z = reshape(Z,n,1);
D = dummyvar(Z);

% We now calculate the two covariance matrices,
% and combine them into an array.
H1 = D*D';
H2 = eye(n,n);
Hall = zeros(n,n,2);
Hall(:,:,1) = H1;
Hall(:,:,2) = H2; 


 
% We then simulate the variance components of the model as multivariate normals
rho = 0.5;
C_theo = [1 rho; rho 1];
beta0 = [1 1];
beta = [2 3];
eps = mvnrnd(beta0,C_theo,n);
alpha = mvnrnd(zeros(2,1),C_theo,b);

x  = normrnd(0,1,n,1);
Y = x*beta + D*alpha + eps;
X_tmp = [ones(n,1) x];

% FMINQUE
result = FMINQUE(Y,X_tmp,Hall,2,false);
sigb2hat_F = result(:,:,1);
sige2hat_F = result(:,:,2);
for j = 1:2;
    W = sigb2hat_F(j,j)*H1 + sige2hat_F(j,j)*H2;
    tmp1 = X_tmp'*linsolve(W,X_tmp);
    tmp2 = X_tmp'*linsolve(W,Y(:,j));
    bhat(:,j) = linsolve(tmp1,tmp2);
end;

bhat(2,:)
beta
