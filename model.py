import pystan
import numpy as np

import pandas as pd

ocode = """
data {
    int<lower=0> N;
    vector[N] l;
    vector[N] a;
}
parameters {
    real alpha;
    real beta;
    real gamma;
    real psi;
    real psi0;
    real<lower=0> sigmaA;
    real<lower=0> sigmaL;
}
model {


    for (n in 2:N){
        a[n] ~ normal(psi0 + psi *a[n-1], sigmaA);
        l[n] ~ normal(alpha + beta * l[n-1] + gamma*a[n-1], sigmaL);
    }
}
"""

sequence = pd.read_csv("ROI1.txt",sep='\t')
length = sequence['Mem Length'].values
actin = sequence['A-CatApicalSum'].values
#print length


sm = pystan.StanModel(model_code=ocode)

op = sm.optimizing(data=dict(l=length,a=actin, N=len(length)))
print op['alpha']
print op['beta']
print op['gamma']
print op['sigmaL']
print op['sigmaA']
print op['psi0']
print op['psi']
