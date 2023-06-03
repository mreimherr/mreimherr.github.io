
% set a path for implementing the algorithms
%addpath(genpath('X:/16_ResearchOneFile/ADMM/testing_2_24'));
addpath(genpath('/Users/Matthew/Desktop/ForJeff/AFSL'));
addpath(genpath('/Users/Matthew/Desktop/ForJeff'));

%% Data Sets Generation
% processing the data sets.
Y = csvread('accel.csv');
X = csvread('cov_data.csv');


X= zscore(X);
I =size(X,2);  % the number of covariates
N =size(Y,1);   % the number of observations
M =size(Y,2);


nbasis = 50;
T_domain = (0:(M-1))/(M-1);
bspline_basis = create_bspline_basis([0 1], nbasis, 4);
Y_f = data2fd(T_domain',Y', bspline_basis,2,0.000001);



% Call on FSL and AFSL
nlam=50;
BIC_para = 0;
FEV_thresh = 0.95;
M_pc=3;
lamratio = 0.01;
[history]=AFSL(Y_f, X,M, M_pc, nlam, BIC_para, lamratio);



subplot(1,2,1)
plot(T_domain,history.Predictor_estimation_FSL')
xlabel('time','FontSize',12,'FontWeight','bold','Color','k')
title({'The FSL estimate  plotted '},'FontSize',8,'FontWeight','bold')

subplot(1,2,2)
plot(T_domain,history.Predictor_estimation_AFSL')
xlabel('time','FontSize',12,'FontWeight','bold','Color','k')
title({'AFSL estimate plotted'},'FontSize',8,'FontWeight','bold')
