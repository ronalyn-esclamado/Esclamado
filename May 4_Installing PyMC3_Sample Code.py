# -*- coding: utf-8 -*-
"""
Created on Sat May 4 17:15:11 2024

@author: user
"""

import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt

# Generate some data
np.random.seed(0)
N = 100
x = np.random.normal(size=N)
y = 3 * x + np.random.normal(size=N)

# Define the model
with pm.Model() as basic_model:
    # Priors
    sigma = pm.HalfNormal('sigma', sd=1)
    intercept = pm.Normal('Intercept', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=10)

    # Likelihood
    likelihood = pm.Normal('y', mu=intercept + beta * x, sd=sigma, observed=y)

    # Inference
    trace = pm.sample(5000, tune=1000, return_inferencedata=False)

# Posterior analysis
with basic_model:
    az.plot_trace(trace)