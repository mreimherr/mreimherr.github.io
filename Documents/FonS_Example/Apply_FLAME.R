#install.packages("FLAME/flm_0.1.tar.gz",rep=NULL)
library(flm)
library(fda)
load("Activity.RDA")
#write data for AFSL
#write.table(accel,"accel.csv",row.names=FALSE,col.names=FALSE,sep=",")
#write.table(covariate_data,"cov_data.csv",row.names=FALSE,col.names=FALSE,sep=",")

Y<-accel
X <- covariate_data
X <- scale(X)

N<-dim(Y)[1]
M<-dim(Y)[2]
I<-dim(X)[2]

tm<-seq(0,1,length=M)



mybasis<-create.bspline.basis(c(0,1),nbasis=50)
mypar<-fdPar(mybasis,2,.000001) # lowest gcv
Y_sm<-smooth.basis(tm,t(Y),mypar)
Y_fd<-Y_sm$fd
# Data looks periodic

# not a large set, so set kill switch to all predictors
FLAME_fit<-FLAME(Y_fd,X,type_kernel="sobolev", NoI = I)
FLAME_fit$predictors

# 2nd option for flame, you can use a list
Y_list<-list(time_domain=tm,data=accel)
FLAME_fit2<-FLAME(Y_list,X,type_kernel="sobolev", NoI = I)
FLAME_fit2$predictors

plot(FLAME_fit$beta)
length(FLAME_fit$predictors)


# data also looks periodic, so we can swap to periodic kernel
FLAME_fit_p<-FLAME(Y_fd,X,type_kernel="periodic", period_kernel=1, NoI = I)
plot(FLAME_fit_p$beta)
length(FLAME_fit_p$predictors)




