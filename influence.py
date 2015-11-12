import pystan
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

ocode = """
data {
    int<lower=0> N;
    vector[N] l;
    
    
    vector[N] a;
    
}
parameters {
    vector[N] lp;    
    vector[N] ap;
    real<lower=0> sigmaA;
    real<lower=0> sigmaL;
    
    real muAP0;
    real<lower=0> sigmaAP0;
    
    real muLP0;
    real<lower=0> sigmaLP0;
    
    real<lower=0> sigmaAObs;
    real<lower=0> sigmaLObs;
    
    real muATrans;
    real muLTrans;
    
    real alphaATrans;
    real alphaLTrans;
    real<lower=0> sigmaATrans;
    real<lower=0> sigmaLTrans;
    real betaALTrans;
}
model {
    ap[1] ~ normal(muAP0,sigmaAP0);
    lp[1] ~ normal(muLP0,sigmaLP0);

    for (n in 2:N){
        ap[n] ~ normal(muATrans + alphaATrans *ap[n-1], sigmaATrans);
        lp[n] ~ normal(muLTrans + alphaLTrans *lp[n-1] + betaALTrans*ap[n-1], sigmaLTrans);
    }
    
    for (n in 1:N){
        a[n] ~ normal(ap[n],sigmaAObs);
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

op = sm.optimizing(data=dict(l=length,a=actin, N=len(length)))
plt.plot(op["lp"],label="lp")
plt.plot(length,label="l")

plt.plot(op["ap"],label="ap")
plt.plot(actin,label="a")

plt.legend(loc='upper left')
plt.show()
for key in op:
    print key
    print op[key]
