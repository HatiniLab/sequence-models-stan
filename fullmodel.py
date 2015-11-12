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
    //vector[N] beta;
    real beta;
    //real<lower=0> sigma;
    //real mu0;
    real mu1;
    real<lower=0> sigmaO;
}
model {
    //sigma ~ exponential(2.5);
    //mu0 ~ normal(0,10);
    sigmaO ~ exponential(2.5);
    beta ~ normal(0,10);
    mu1 ~ normal(0,10);
    //beta ~ normal(mu0,sigma);
    //beta[1] ~ normal(mu0,sigma);
    //for (n in 2:N){
        //beta[n] ~ normal(beta[n-1],sigma);
    //}
    for (n in 1:N){
        //l[n] ~ normal(mu1+beta[n]*a[n],sigmaO);
        l[n] ~ normal(mu1+beta*a[n],sigmaO);
    }
}
"""

sequence = pd.read_csv("data/ROI2.txt",sep='\t')
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

op = sm.optimizing(data=dict(l=length,a=actin, N=len(length)),iter=10000)
plt.plot(length,label="l")
plt.plot(actin,label="a")
plt.plot(op["beta"],label="beta");
plt.legend(loc='upper left')
plt.show()
for key in op:
    print key
    print op[key]
