import pystan
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd


sequence = pd.read_csv("data/ROI1.txt",sep='\t')
length = sequence['Mem Length'].values
meanlength =sequence['Mem Length'].mean()
stdlength  =sequence['Mem Length'].std()
length = length -meanlength
length = length/stdlength


sm = pystan.StanModel(file="./hmm.stan")

op = sm.optimizing(data=dict(y=[length], N=1,T=len(length),K=2))    

for key in op:
    print key
    print op[key]
