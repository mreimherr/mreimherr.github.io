# X: Functional time series to find the order of, can either be in matrix form or an fd object
# p_all: can either be an integer to test all orders 1:p_all, or a vector of orders the user wants to specifically test
# npc: Number of FPCs to be used when estimating the FAR model
# Author: Matthew Reimherr, mreimherr@psu.edu
Order_FAR<-function(X, p_all = 4, npc = 3){
	require(fda)
	
	nharm<-30
	if(class(X)=="fd"){
		M<-300
		t_eval<-seq(0,1,length=M)
		Xg = t(eval.fd(t_eval,X))
		nb<-dim(X$coefs)[1]
		fbf<-X$basis
	}else if(class(X)=="matrix"){
		M<-dim(X)[2]
		N<-dim(X)[1]
		t_eval<-seq(0,1,length=M)
		nb<-100
		fbf = create.bspline.basis(rangeval=c(0,1), nbasis=nb) 
		Xg = X
	}else{
		error(stop("X needs to be a matrix or functional"))
	}
	Test<-numeric(0)
	pval<-numeric(0)
	if(length(p_all) == 1){
		p_vec = 1:p_all
	}else{p_vec = p_all}
	for(p in p_vec){
		Xgp<-Xg[p:(N-1),]
		if( p > 1){
		for(j in 2:p){Xgp<-cbind(Xgp,Xg[(p-j+1):(N-j),])}
		}
		t_eval2 = seq(0,1,length=p*M)
		Xgfo = Data2fd(t_eval2, t(Xgp),  fbf)
		Ygfo = Data2fd(t_eval, t(Xg[(p+1):N,]),  fbf)
		Xgfo<-center.fd(Xgfo)
		Ygfo<-center.fd(Ygfo)
		
		qy<-npc
		qx<-qy*p
		
		
		X.fd<-pca.fd(Xgfo,qx)
		Y.fd<-pca.fd(Ygfo,qy)
		
		X<-X.fd$scores
		Y<-Y.fd$scores
		lm1<-lm(Y~X)
		Ceps<-(1/(N-p-qx-1))*(t(lm1$res)%*%lm1$res)
		
		M0<-400 #points to approx integral of harmonics over [(p-1)/(p),1] interval
		t.tmp<-seq(from=(p-1)/(p), to = 1, length.out = (M0+1))
		t.tmp<-t.tmp[2:(M0+1)]
		v.tmp<-eval.fd(t.tmp,X.fd$harmonics)
		V<-t(t(v.tmp)%*%v.tmp)/(M*p)
		eg.v<-eigen(V)
		
		qv<-length(which(eg.v$values>0.9))
		alpha<-eg.v$vectors[,(1:qv)]
		lambda<-diag(t(X)%*%X)/N
		Psi<-diag((N*lambda)^(-1))%*%(t(X)%*%Y)
		I_qy<-diag(rep(1,times=qy))
		
		av_tmp<-as.vector(t(alpha)%*%Psi)
		Tst.Stat<-(N-p)*av_tmp%*%solve( (I_qy%x%t(alpha))%*%(Ceps%x%diag(1/lambda))%*%(I_qy%x%alpha) )%*%av_tmp
		
		Test<-c(Test,Tst.Stat)
		pval<-c(pval,1-pchisq(Tst.Stat,df=(qv*qy)))
	}	
	Sig<-rep('',times=length(p_vec))
	Sig[pval < 0.1] = '*'
	Sig[pval < 0.05] = '**'
	Sig[pval < 0.01] = '***'
	Table<-data.frame(p_vec,Test,pval,Sig)
	names(Table) <-c("Order under HA","Test Stat","P-Value","Sig")
	return(Table)
}