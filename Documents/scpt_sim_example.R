# This is a relativley simple data simulation to illustrate
# the scpt package which implements the change-point procedures
# outline in Gromenko, Kokoskza, and Reimherr (2015)
#
# The package was written by Oleksander Gromenko and Matthew Reimherr
# This example was written by Matthew Reimherr and Panyiotis Constantinou
# send any questions to mreimherr@psu.edu

# begin by removing everything and freeing memory
rm(list=ls()); gc()

# install the scpt package binary.  
# this can be downloaded from www.personal.psu.edu/~mlr36
install.packages("scpt",rep="http://R-Forge.R-project.org")
library(MASS)
library(scpt)

# First we set the various sampling parameters
N=100 # Number of curves per location
M=50 # Number of obseverations per curve
S=11 # Number of spacial locations
T=seq(0,1,length=M) # generates time points
delta<-0 # This is a very simple change-point parameter, half the curves will have mean zero, the other half mean delta (at all space/time locations).  Set delta = 0 to work under the null.  

# Next we set the covariance function parameters
# These parameters refer to the covariance function 
# in Gneiting (2002)
gamma=1 #gamma belongs to (0,1]
alph=1/2  #alph belongs to (0,1] 
sigm=1    # sigm^2 > 0
beta=0 # beta belongs to [0,1], for beta=0 we have separable covariance 
c=1   # nonnegative scaling parameter of space  
a=1   # nonnegative scaling parameter of time
taph=1 # beta*d/2 >= 0

# We will use spatial locations on the square that are an equal 
# distance apart.
L=1+round(sqrt(S))
x=seq(0,1,length=L)
y=seq(0,1,length=L)
coordinates=expand.grid(x,y)
train=sample(1:L^2,size=S)
A<-matrix(0,S,2)
A=coordinates[train,]

# We now define the covariance function
covariance_function=function(s1,s2,t1,t2){((sigm^2)/(a*(abs(t1-t2))^(2*alph)+1)^(taph))*exp(-c*(sqrt(sum((s1-s2)^2)))^(2*gamma)/(a*(abs(t1-t2))^(2*alph)+1)^(beta*gamma))}

# Next we generate a large matrix of all of the spatial/temporal locations.
X_matrix=matrix(0,S*M,3)
X_matrix[,3] = rep(T,each=S)
X_matrix[,1] = rep(A[,1],times=M)
X_matrix[,2] = rep(A[,2],times=M)

# Then we define a large covariance matrix
# At this point, the data spatio/temporal data are being 
# treated in vector form.  We will conver to them to an array later.
Covariance_Matrix=matrix(0,S*M,S*M)
for(i in 1:(M*S)){
for(j in i:(M*S)){
Covariance_Matrix[i,j]=covariance_function(X_matrix[i,1:2],X_matrix[j,1:2],X_matrix[i,3],X_matrix[j,3])
Covariance_Matrix[j,i]=Covariance_Matrix[i,j]
}}

# We now simulate the vectorized data
X_data=mvrnorm(n=N,mu=rep(0,S*M),Sigma=Covariance_Matrix)

# Now they are converted into an array
X_array<-array(0,dim=c(M,S,N))
for(n in 1:N){
	X_array[,,n]=t(matrix(X_data[n,],nrow=S,ncol=M))
	if(n >= N/2){X_array[,,n] = X_array[,,n] + delta}
}

# Finally, carry out the test.
# Be careful with the number of cores argument as 
# the the default will use all alvailable resourse.
# Use the detectCores() function in the parallel package
# to find out the number of available cores.
# WARNING: number of cores must equal 1 on windows due to mclapply
tmp<-spatial_change_test(X_array,A,.85,'all',250)

# Thre p-values are returned for the three different tests.
tmp

