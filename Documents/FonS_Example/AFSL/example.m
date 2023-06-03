% This is a simulation to illustrate high-dimensional adaptive function-on-scalar regression in Fan and Reimherr(2016).
% This code was initially written by Rina Foygel Barber, and then expanded and edited by Zhaohu(Jonathan) Fan.
% This simulation calls on our main function.
% We only simulate one data set and call on our main function
% Send any questions to Matthew Reimherr at mreimherr@psu.edu


% We are thrilled to anounce that we propose an efficent computational framework for high dimemsional functional regression in our working paper
% Send any questions to Matthew Reimherr at mreimherr@psu.edu for that paper.



% set a path for implementing the algorithms
addpath(genpath('X:/16_ResearchOneFile/ADMM/testing_2_24'))

%% Data Sets Generation
% processing the data sets.

I =500;  % the number of covariates
N =100;   % the number of observations

% First we set parameters for simulations
nlam_base=100;
BIC_para = 1;

%%%%%%%%%%%%%%%%%
M=50;
I0 = 10;  % we arbitrarily select 10 covariates out of I as our target covariables;
rho = 0.5;
nbasis = min([50 M]);
T_domain = (0:(M-1))/(M-1); 

% Next we set up the covariates parameters for Matern process%

mu_X = zeros(I,1); 
Sig_X = ones(I,I); 
for i = 1:(I-1)
    for j = (i+1):I
        Sig_X(i,j) = rho^(j-i);
        Sig_X(j,i) = Sig_X(i,j);
    end
end 
% Here we set  signals parameters for our model %
nu_alpha = 2.5; 
range = 1/4;
variance = 1;
hyp = [log(range),log(variance)/2]; 
% we set errors parameters for our model%
nu_eps = 1.5;
mu_eps = zeros(M,1);
range1=0.01; 
variance = 1;
hyp1 = [log(range1),log(variance)/2]; 
Sig_eps=covMaterniso(2*nu_eps,hyp1,T_domain');
Sig_eps2 = (Sig_eps + Sig_eps.')./2;
mu_alpha = zeros(M,1);
Sig_alpha = covMaterniso(2*nu_alpha,hyp,T_domain');

X = mvnrnd(mu_X,Sig_X,N);
X= zscore(X);
alpha_1 = mvnrnd(mu_alpha,Sig_alpha,I0); 
eps = mvnrnd(mu_eps,Sig_eps2,N);
I_X = randsample(I,I0,false); 
Y_full = X(:,I_X)*alpha_1+ eps;



subplot(1,3,1)
plot(T_domain,Y_full')
xlabel('time','FontSize',12,'FontWeight','bold','Color','k')
title({'Y(t) functions'},'FontSize',8,'FontWeight','bold')

subplot(1,3,2)
plot(T_domain,eps')
xlabel('time','FontSize',12,'FontWeight','bold','Color','k')
title({'error coefficients'},'FontSize',8,'FontWeight','bold')
subplot(1,3,3)
plot(T_domain,alpha_1')
xlabel('time','FontSize',12,'FontWeight','bold','Color','k')
title({'betas functions'},'FontSize',8,'FontWeight','bold')

M_pc=3;

% If Y(t) is not a functional object, then the following steps are to do the FD conversion
Y_obs = zeros(N,M); 
T_obs = zeros(N,M); 
T_pos = zeros(N,M); 
Y_obs_cell = cell(1,N);
T_obs_cell = cell(1,N);
    for i = 1:N
        T_pos(i,:) = 1:M;
        T_obs(i,:) = T_domain(T_pos(i,:));
        Y_obs(i,:) = Y_full(i,T_pos(i,:)); 
        Y_obs_cell{i} = Y_obs(i,:);
        T_obs_cell{i} = T_obs(i,:);
    end
    Y_obs_vec = reshape(Y_obs',N*M,1);

     bspline_basis = create_bspline_basis([0 1], nbasis, 4);
     Y_f = data2fd(T_domain',Y_obs', bspline_basis,2,0.00001);
    

% set parameters for FSL
 lamratio=0.001;
% Call on FSL and AFSL
  [history]=AFSL(Y_f, X,M, M_pc, nlam_base, BIC_para, lamratio)
 

subplot(1,3,1)
plot(T_domain,alpha_1')
xlabel('time','FontSize',12,'FontWeight','bold','Color','k')
title({'betas functions'},'FontSize',8,'FontWeight','bold')

subplot(1,3,2)
plot(T_domain,history.Predictor_estimation_FSL')
xlabel('time','FontSize',12,'FontWeight','bold','Color','k')
title({'The FSL estimate  plotted '},'FontSize',8,'FontWeight','bold')

subplot(1,3,3)
plot(T_domain,history.Predictor_estimation_AFSL')
xlabel('time','FontSize',12,'FontWeight','bold','Color','k')
title({'AFSL estimate plotted'},'FontSize',8,'FontWeight','bold')
