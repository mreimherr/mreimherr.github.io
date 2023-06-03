% This is a main function to illustrate high-dimensional adaptive function-on-scalar regression in Fan and Reimherr(2016).
% This code was initially written by Rina Foygel Barber, and then expanded and edited by Zhaohu(Jonathan) Fan.
% Our main function includes BIC calculations
% Send any questions to Matthew Reimherr at mreimherr@psu.edu

% We are thrilled to anounce that we propose an efficent computational framework for high dimemsional functional regression in our working paper.
% Send any questions to Matthew Reimherr at mreimherr@psu.edu for that paper.

function    [history]= AFSL(Y_f, X,M, M_pc, nlam_base, BIC_para, lamratio)
   
tic
 % FPCA inside the function.
 [N,I]=size(X);
 Y_pca = pca_fd(Y_f,M_pc);
     Y_scores = getVal(Y_pca,'harmscr');
     FPCA_func_tmp = getVal(Y_pca,'harmfd');
     eval_pts = (0:(M-1))/(M-1);
     FPCA_func = eval_fd(FPCA_func_tmp,eval_pts);
     Y_sm_vec = reshape(Y_scores',N*M_pc,1);
     

 A_sm = kron(X,diag(ones(M_pc,1)));
   b = Y_sm_vec;
Beta_captured_sm = zeros(1, 1);
Beta_captured_sm_ad = zeros(1, 1);
    p_sm = M_pc*ones(I,1);
    nlam = nlam_base;
for j=1:1
    % FSL algorithm
    tic
    [Bhat_sm, lambda_seq, hist] = group_lasso_activeset(A_sm, b, p_sm, 1.0, 1.0,nlam, lamratio, 40);
    nlam = size(lambda_seq,2);
    BIC_sm = zeros(nlam,1);
    act_set= zeros(nlam,1);
     Res_thresh = 1e-10;
    cross_thresh=0;
    for i = 1:nlam
        tmp_bhat = Bhat_sm(:,i);
        tmp_bhat_mat = reshape(tmp_bhat,M_pc,I)';
         active_mat = sqrt(diag(tmp_bhat_mat*tmp_bhat_mat')) > 0;
        act_set(i) = sum(active_mat);
        %BIC calculation
          if(i == 1 || act_set(i) ~= act_set(i-1))
            if(act_set(i)~=0 && cross_thresh==1)
                BIC_sm(i) = M_pc*act_set(i)*log(N*M_pc) + BIC_para*M_pc*act_set(i)*log(I);
            elseif(act_set(i)~=0 && cross_thresh==0)
                active_vec = zeros(M_pc,I);
                active_vec(:,active_mat) = 1;
                active_vec = logical(reshape(active_vec,M_pc*I,1));
                Atmp = A_sm(:,active_vec);
                HatMat = Atmp*((Atmp'*Atmp)\Atmp');
                Res = norm(Y_sm_vec - HatMat*Y_sm_vec)^2;
                BIC_sm(i) = N*M_pc*log(Res/(N*M_pc)) + M_pc*act_set(i)*log(N*M_pc) + BIC_para*M_pc*act_set(i)*log(I);
                if(Res < Res_thresh)
                    cross_thresh = 1; 
                end
            else
                BIC_sm(i) = N*M_pc*log(norm(b)^2/(N*M_pc));
            end 
        else
            BIC_sm(i) = BIC_sm(i-1);
          end
    end
    nlam_opt_sm=find(BIC_sm==min(BIC_sm),1,'Last');
    Bhat_matrix_sm = reshape(Bhat_sm(:,nlam_opt_sm),M_pc,I)';
    captured_coeff_sm = ( sqrt(diag(Bhat_matrix_sm*Bhat_matrix_sm')) ~= 0);
    alpha_hat_sm = Bhat_matrix_sm*FPCA_func'; 
    
    BIC_sm_FSL=min(BIC_sm);
   [m,n]=size(find(captured_coeff_sm));
   Beta_captured_sm(1,j)=m;
   Beta_captured_sm_index=find(captured_coeff_sm);         
   FSLtime=toc;
   tic   
   % AFSL algorithms
   [alpha_hat_sm_zeros,alpha_hat_sm_nonzeros,I_zeros_index,I_nonzeros_index,Bhat_matrix_sm_zeros,Bhat_matrix_sm_nonzeros]=detector(Bhat_matrix_sm,alpha_hat_sm);% choose significant covariate randomly
   [mm,nn]=size(I_nonzeros_index); 
  if mm==0
       warning('no predictors are selected in the FSL step')
       history=0; 
       continue
  else       
    I_tilde = size(I_nonzeros_index,1);
    I_X_tilde = I_nonzeros_index;
    X_tilde = X(:,I_X_tilde);
    ahs_tilde = alpha_hat_sm(I_X_tilde);
    alpha_hat_sm_pre_adp = zeros(1,mm);
    for i=1:mm
      alpha_hat_sm_pre_adp(i)=sqrt((mean(ahs_tilde(i,:).^2)));
    end
    X_tilde_adp=X_tilde*diag(alpha_hat_sm_pre_adp);

    A_sm_adp = kron(X_tilde_adp,diag(ones(M_pc,1)));
  
    b = Y_sm_vec;
    p_sm_adp = M_pc*ones(I_tilde,1);
    nlam = nlam_base;
    [Bhat_sm_ad, lambda_seq_ad, hist] = group_lasso_activeset(A_sm_adp, b, p_sm_adp, 1.0, 1.0,nlam, lamratio, 40);

     nlam = size(lambda_seq_ad,2);
    BIC_sm = zeros(nlam,1);
    act_set= zeros(nlam,1);
     Res_thresh = 1e-10;
   cross_thresh=0;
    for i = 1:nlam   
        tmp_bhat = Bhat_sm_ad(:,i);
        tmp_bhat_mat = reshape(tmp_bhat,M_pc,I_tilde)';
        active_mat = sqrt(diag(tmp_bhat_mat*tmp_bhat_mat')) > 0;
        act_set(i) = sum(active_mat);
        %BIC calculation 
          if(i == 1 || act_set(i) ~= act_set(i-1))
            if(act_set(i)~=0 && cross_thresh==1)
                BIC_sm(i) = M_pc*act_set(i)*log(N*M_pc) + BIC_para*M_pc*act_set(i)*log(I);
            elseif(act_set(i)~=0 && cross_thresh==0)
                active_vec = zeros(M_pc,I_tilde);
                active_vec(:,active_mat) = 1;
                active_vec = logical(reshape(active_vec,M_pc*I_tilde,1));
                Atmp = A_sm_adp(:,active_vec);
                HatMat = Atmp*((Atmp'*Atmp)\Atmp');
                Res = norm(Y_sm_vec - HatMat*Y_sm_vec)^2;
           
                BIC_sm(i) = N*M_pc*log(Res/(N*M_pc)) + M_pc*act_set(i)*log(N*M_pc) + BIC_para*M_pc*act_set(i)*log(I);
                if(Res < Res_thresh)
                    cross_thresh = 1; 
                end
            else
                BIC_sm(i) = N*M_pc*log(norm(b)^2/(N*M_pc));
            end 
        else
            BIC_sm(i) = BIC_sm(i-1);
          end
    end
    nlam_opt_sm=find(BIC_sm==min(BIC_sm),1,'Last');
    
    BIC_sm_AFSL=min(BIC_sm);
    Bhat_matrix_sm_ad_tmp = reshape(Bhat_sm_ad(:,nlam_opt_sm),M_pc,I_tilde)';
    captured_coeff_sm_ad= ( sqrt(diag(Bhat_matrix_sm_ad_tmp*Bhat_matrix_sm_ad_tmp')) ~= 0);
    alpha_hat_sm_ad_tmp= Bhat_matrix_sm_ad_tmp*FPCA_func'; 
    alpha_hat_sm_ad = diag(alpha_hat_sm_pre_adp)*alpha_hat_sm_ad_tmp;
      Beta_captured_sm_ad_id= find(captured_coeff_sm_ad);
      [m,n]=size( Beta_captured_sm_ad_id);
      Beta_captured_sm_ad(1,j)=m;
      Beta_captured_sm_ad_index= Beta_captured_sm_index(Beta_captured_sm_ad_id);
  end
   
    AFSLtime=toc; 
    history.BIC_FSL=BIC_sm_FSL;
    history.BIC_AFSL=BIC_sm_AFSL;
    history.Predictor_selected_FSL=Beta_captured_sm_index;
    history.Predictor_selected_AFSL=Beta_captured_sm_ad_index;
    [~,idx]=ismember(history.Predictor_selected_AFSL,history.Predictor_selected_FSL);
    history.Predictor_estimation_FSL=alpha_hat_sm(Beta_captured_sm_index,:);
    history.Predictor_estimation_AFSL=alpha_hat_sm_ad(idx,:);
    history.FSLtime=FSLtime;
    history.AFSLtime=AFSLtime+FSLtime;
end

