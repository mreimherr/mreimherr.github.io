# Here we simulate under an FAR(2) and and apply "Order_FAR" 
# See Kokoszka and Reimherr (2013) for details
library(fda)

######
### define all data generating parameters
#####

# K: time points per curve
# N: sample size/length of functional time series
# burn: burn in to ensure we are at stationary distribution
K=300
N=100
burn=200

######
### create Gaussian Kernel using ||\Psi||=P
######
# These control the norm of the FAR operators.
# If P1=P2 = 0, then the model is FAR(0), i.e. iid
# If P1 != 0 and P2 = 0, then the model is FAR(1)
# If both P1!= and P2!=0, then the model is FAR(2)
P1=.5
P2=.3

# The kernel of the FAR operators are taken to be gaussian.
# t are the time points we observe.
t=seq(0,1,K)
G1 = matrix(0, K, K)
G2 = matrix(0, K, K)
for(i in 1:K){
	for(j in 1:K){
		G1[i,j]= (P1/.7468241328)*exp(-(t[i]^2+t[j]^2)/2)
		G2[i,j]= (P2/.7468241328)*exp(-(t[i]^2+t[j]^2)/2)
	}
}




#create a K*N matrix of standard normal vectors
S <-matrix(rnorm(K*(N+burn),0,1),nrow=N+burn,ncol=K)
S[,1]<-0

#create Weiner process(Browian Motion) using S
W<-matrix(0,nrow=N+burn,ncol=K)
for(j in 1:(N+burn)){W[j,] = (1/sqrt(K))*cumsum(S[(j),])}

#create Brownian Bridge
B = matrix(0,N+burn,K)
for(i in 1:K){B[,i] = W[,i]-((i-1)/(K-1))*W[,K]}


#######
### generate FAR(1) process using Gaussian Kernel 
#######
Xg= matrix(0, N+burn, K)
Xg[1,] = B[1,]
for( i in 3:(N+burn)){
	#Xg[i,] = (1/K)*G1%*%Xg[i-1,]+ (1/K)*G2%*%Xg[i-2,]+B[i,] #guassian model
	Xg[i,] = P1*Xg[i-1,]+ P2*Xg[i-2,]+B[i,] #simple model
	}

Xg=Xg[(burn+1):(N+burn),]


#Now apply our test
p_max<-2 # max order to test
source("Order_FAR.R")

# Feed the raw data
# Assumes equally spaced on unit inteval
# uses Bspline basis in calculations
Order_FAR(Xg,p_max,npc=3)

# Or feed functional objects
nb<-100 #Number of basis functions for the data
fbf = create.bspline.basis(rangeval=c(0,K)/K, nbasis=nb) #Basis for data
X.f<-Data2fd(t,t(Xg),fbf)
Order_FAR(X.f,p_max,npc=3)

# Should only be a small difference between the two 
# due to slight smoothing when using Data2fd




