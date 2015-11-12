import pystan
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

ocode = """
data {
    int<lower=0> N;
    vector[N] a;
}
parameters {    
    vector[N] ap;
    
    real muAP0;
    real<lower=0> sigmaAP0;
    
    real<lower=0> sigmaAObs;
       
    real muATrans;
    
    real alphaATrans;
    real<lower=0> sigmaATrans;
}
model {
    ap[1] ~ normal(muAP0,sigmaAP0);

    for (n in 2:N){
        ap[n] ~ normal(muATrans + alphaATrans *ap[n-1], sigmaATrans);
    }
    
    for (n in 1:N){
        a[n] ~ normal(ap[n],sigmaAObs);
    }
}
"""

sequence = pd.read_csv("data/ROI1.txt",sep='\t')

actin = sequence['A-CatApicalSum'].values
meanActin=sequence['A-CatApicalSum'].mean()
stdActin  =sequence['A-CatApicalSum'].std()

actin = actin -meanActin
actin = actin/stdActin
#print length


sm = pystan.StanModel(model_code=ocode)

op = sm.optimizing(data=dict(a=actin, N=len(actin)))

plt.plot(op["ap"],label="ap")
plt.plot(actin,label="a")

plt.legend(loc='upper left')
plt.show()
for key in op:
    print key
    print op[key]
