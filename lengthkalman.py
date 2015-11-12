import pystan
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

ocode = """
data {
    int<lower=0> N;
    int<lower=1> K;
    vector[N] l;
    
    vector<lower=0>[K] transPprior; // transit prior
}
parameters {
    vector[N] lp;  
    real<lower=0> sigmaL;
    
    simplex[K] mixture[N];
    real muLP0;
    real<lower=0> sigmaLP0;
    
    real<lower=0> sigmaLObs;
        
    real muLTrans[K]; 
    real alphaLTrans[K];
    real<lower=0,upper=1> sigmaLTrans[K];
}
model {
    lp[1] ~ normal(muLP0,sigmaLP0);
    
    for(k in 1:K){
        mixture[1] ~ dirichlet(transPprior);
    }
    
    real ps[K];
    for (n in 1:N) {
        for (k in 1:K) {
            ps[k] <- log(theta[k])+ normal_log(lp[n],alphaLTrans[k]*lp[n-1]+muLTrans[k],sigmaLTrans[k]);
        }
        increment_log_prob(log_sum_exp(ps));
    }

    for (n in 2:N){
        for  (k in 1:K){
        
            ps[k] <- 
            lp[n] ~ normal(muLTrans[mode[n-1]] + alphaLTrans[mode[n-1]] *lp[n-1], sigmaLTrans[n-1]);
            newMode <-  categorical(transP[mode[n-1]]);
            mode[n] ~ newMode;
            
        }
    }
    
    for (n in 1:N){
        l[n] ~ normal(lp[n],sigmaLObs);
    }
}
"""

sequence = pd.read_csv("data/ROI1.txt",sep='\t')
length = sequence['Mem Length'].values
meanlength =sequence['Mem Length'].mean()
stdlength  =sequence['Mem Length'].std()
length = length -meanlength
length = length/stdlength

actin = sequence['A-CatApicalSum'].values
meanActin=sequence['A-CatApicalSum'].mean()
stdActin  =sequence['A-CatApicalSum'].std()

actin = actin -meanActin
actin = actin/stdActin
#print length


sm = pystan.StanModel(model_code=ocode)

op = sm.optimizing(data=dict(l=length, N=len(length),K=2))
plt.plot(op["lp"],label="lp")
plt.plot(length,label="l")

plt.plot(op["ap"],label="ap")
plt.plot(actin,label="a")

plt.legend(loc='upper left')
plt.show()
for key in op:
    print key
    print op[key]
