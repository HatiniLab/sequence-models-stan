from pymc3 import Model, Normal, HalfNormal
from pymc3 import find_MAP


import matplotlib.pyplot as plt

import pandas as pd


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


basic_model = Model()

with basic_model:

    # Priors for unknown model parameters
    alpha = Normal('alpha', mu=0, sd=10)
    beta = Normal('beta', mu=0, sd=10)
    sigma = HalfNormal('sigma', sd=1)

    # Expected value of outcome
    mu = alpha + beta*actin

    # Likelihood (sampling distribution) of observations
    Y_obs = Normal('Y_obs', mu=mu, sd=sigma, observed=length)

map_estimate = find_MAP(model=basic_model)

print(map_estimate)
