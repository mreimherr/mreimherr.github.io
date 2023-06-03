
function [xs,lambda_seq, history] = ...
    group_lasso_activeset(A, b, p, rho, alpha, nlam, lamratio, smax)
% Solve group lasso problem via ADMM using a sequence of lambdas,
%           controlling active set size along the path
% modified from Boyd's group_lasso function
%
% solves the following problem via ADMM:
%
%   minimize 1/2*|| Ax - b ||_2^2 + \lambda sum(norm(x_i))
%
% Input:
%   p is a K-element vector giving the block sizes n_i, so that x_i
%       is in R^{n_i}.
%   rho is the augmented Lagrangian parameter.
%   alpha is the over-relaxation parameter (typical values for alpha are
%               between 1.0 and 1.8).
%   nlam = # lambda values
%   lamratio = min(lambda value)/max(lambda value)
%   smax = max # groups allowed in the solution
%
% 
%
% The solution is returned in the vector x.
%
% history is a structure that contains the objective value, the primal and
% dual residual norms, and the tolerances for the primal and dual residual
% norms at each iteration.
%
%
%
%
% More information can be found in the paper linked at:
% http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
%

t_start = tic;

%% Global constants and defaults

QUIET    = 1;
MAX_ITER = 1000;
ABSTOL   = 1e-4;
RELTOL   = 1e-2;
ACTIVESET_ITER = 100;
ACTIVESET_BURNIN = 10;

%% Data preprocessing

[m, n] = size(A);

xs = zeros(n,nlam);

K=length(p);p=reshape(p,1,K);
stopind=cumsum(p);
startind=[1, stopind(1:(K-1))+1];

Anorm=zeros(K,1);
for i=1:K, Anorm(i)=norm(A(:,startind(i):stopind(i)),'fro'); end
bnorm=norm(b);

% save a matrix-vector multiply
Atb = A'*b;
% check that sum(p) = total number of elements in x
if (sum(p) ~= n)
    error('invalid partition');
end

% cumulative partition
cum_part = cumsum(p);

% maximum lambda value
norm_temp = cumsum(Atb.^2); norm_temp = norm_temp(cum_part);
norm_temp = norm_temp - [0 norm_temp(1:(end-1))']';
[lambda_max,gmax] = max(sqrt(norm_temp));
lambda_seq=lambda_max*(1:(lamratio-1)/(nlam-1):lamratio);
gmax_ind=cum_part(gmax)-p(gmax)+(1:p(gmax));


%% ADMM solver

x = zeros(n,1);
z = zeros(n,1);
u = Atb; % rather than zeros(n,1) - this means that at lambda=lambda_max,
%                                   optimality of x=0 is visible
%                                   immediately.


if ~QUIET
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
      'r norm', 'eps pri', 's norm', 'eps dual', 'objective', 'penalty');
end

S_hist=[];
S=gmax;Sind=gmax_ind;
% S=[];Sind=[];
ilam=1;resid=b;
history.resid=norm(b);
while(length(S)<smax && ilam<nlam),
    ilam=ilam+1;lam0=lambda_seq(ilam-1);lambda=lambda_seq(ilam);
    
    Sold=S;
    
    % screening rule: find the next lambda & the next active set
    v1=b/lam0-resid/lam0;
    if(ilam==2),
        v1=A(:,gmax_ind)*A(:,gmax_ind)'*b;
    end
    v2=b/lambda-resid/lam0;
    v2perp=v2-v1*(v1'*v2)/norm(v1)^2;
    for g=setdiff(1:length(p),S), % check each new group for inclusion
        g_ind=cum_part(g)-p(g)+(1:p(g));
           if(norm(A(:,g_ind)'*(resid/lam0-v2perp/2))>=...
                1-norm(v2perp)*norm(A(:,g_ind),'fro')/2)
            S=[S g];
            Sind=[Sind g_ind]; % columns of A that need to be included
        end
    end
   
   
    S_hist_line=[lambda 0 S];
    S_hist=[S_hist, zeros(size(S_hist,1),length(S_hist_line)-size(S_hist,2))];
    S_hist=[S_hist;[S_hist_line,zeros(1,size(S_hist,2)-length(S_hist_line))]];
    % pass to group_lasso_warmstart
        dlmwrite('save_to_find_error.txt',S_hist);
    [xS,zS,uS] = group_lasso_warmstart(A(:,Sind), b, lambda, p(S), ...
                            rho, alpha, x(Sind), z(Sind), u(Sind));
    x(Sind)=xS;z(Sind)=zS;u(Sind)=uS;
                        
    xs(:,ilam)=z; % using z not x b/c z has exact zero groups;
                    % at convergence would have x=z
    resid = b-A*z; % used for screening rule on next step
    
    % update S: some groups that were screened in, are still at zero
    Snew=[];Sind=[];
    for g=S,
        g_ind=cum_part(g)-p(g)+(1:p(g));
        if(norm(z(g_ind))>0)
            Snew=[Snew g];
            Sind=[Sind g_ind];
        end
    end
    S=Snew;
    S_hist_line=[lambda 1 S];
    S_hist=[S_hist, zeros(size(S_hist,1),length(S_hist_line)-size(S_hist,2))];
    S_hist=[S_hist;[S_hist_line,zeros(1,size(S_hist,2)-length(S_hist_line))]];
            dlmwrite('save_to_find_error.txt',S_hist);
            
    history.resid(ilam) =  norm(resid); 
end                
                        

xs=xs(:,1:ilam);
lambda_seq=lambda_seq(1:ilam);


                 


if ~QUIET
    toc(t_start);
end
end
