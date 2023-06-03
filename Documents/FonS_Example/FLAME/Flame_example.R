#install.packages("flm_0.1.tar.gz",rep=NULL)
library(flm)
data(simulation)
FLAME_fit<-FLAME(Y_fd,X,type_kernel="sobolev", NoI = 100)
# maybe the most important options are listed above
# you can change the kernel type and the max number of predictors
# allowed in the model.

plot(FLAME_fit$beta)
length(FLAME_fit$predictors