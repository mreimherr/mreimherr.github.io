% Scalar Example of FMINQUE function
% Here we give an example of estimating
% the variance components in a block model.
% We compare to fitlme.

% number of blocks
b = 150;

% observations per block
nb = 2;

% total sample size
n = b*nb;

% within block variance
sigb2 = 2;

% between block variance
sige2 = 1;

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

reps = 10;
beta = 0;
test_stat = zeros(reps,1);
pval = zeros(reps,1);
pval_lme = zeros(reps,1);
sigb2hat_F = zeros(reps,1);
sige2hat_F = zeros(reps,1);
sigb2hat_lme = zeros(reps,1);
sige2hat_lme = zeros(reps,1);

time_F = zeros(reps,1);
time_lme = zeros(reps,1);

for i = 1:reps;    
alpha = normrnd(0,sqrt(sigb2),b,1);
eps = normrnd(0,sqrt(sige2),n,1);
x  = normrnd(0,1,n,1);
Y = x*beta + D*alpha + eps;

% FMINQUE
X_tmp = [ones(n,1) x];
tic;
result = FMINQUE(Y,X_tmp,Hall,2,false);
sigb2hat_F(i) = result(:,:,1);
sige2hat_F(i) = result(:,:,2);
V = sigb2hat_F(i)*H1 + sige2hat_F(i)*H2;
Vinv = inv(V);
B = X_tmp'*Vinv*X_tmp;
bhat = linsolve(B,X_tmp'*Vinv*Y);
SE_mat = inv(B);
test_stat(i) = bhat(2)/sqrt(SE_mat(2,2));
pval(i) = 2*(1-normcdf(abs(test_stat(i)),0,1));
time_F(i) = toc;

% Standard Mixed Effects 
MyData = table(Y,x,Z,'VariableNames',{'Y','x','Z'});
tic;
lme = fitlme(MyData,'Y~x+(1|Z)');
[aa,bb] = covarianceParameters(lme);
sigb2hat_lme(i) = aa{1};
sige2hat_lme(i) = bb;
pval_lme(i) = coefTest(lme);
time_lme(i) = toc;

end;

% Compare variance components
plot(sigb2hat_F,sigb2hat_lme);
plot(sige2hat_F,sige2hat_lme);

% Compare pvalues
plot(pval,pval_lme);

% Compare times
[mean(time_F), mean(time_lme)]
% for smaller samples, FMINQUE is faster
% for larger samples, lme is faster

