# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 18:33:20 2024

@author: Esclamado
"""
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt

# Generating sample data
np.random.seed(42)
X = np.linspace(0, 10, 100)
true_slope = 2
true_intercept = 1
true_noise = 1
Y = true_slope * X + true_intercept + np.random.normal(scale=true_noise, size=len(X))

# Plot the synthetic data
plt.scatter(X, Y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Synthetic Data')
plt.show()

# Reshape X_new to have the shape (10,)
X_new = np.linspace(10, 15, 10)

# Bayesian linear regression model for forecasting
with pm.Model() as model:
    # Priors
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=1)
    
    # Deterministic variable
    mu = alpha + beta * X
    
    # Likelihood
    likelihood = pm.Normal('Y', mu=mu, sigma=sigma, observed=Y)
    
    # Forecasting
    mu_new = alpha + beta * X_new
    Y_pred = pm.Normal('Y_pred', mu=mu_new, sigma=sigma, shape=len(X_new))  # Define shape for Y_pred
    
    # Sampling
    trace = pm.sample(2000, tune=1000, cores=1)  # Adjust tune and cores as needed

# Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, label='Original Data')
plt.plot(X_new, np.mean(trace['Y_pred'], axis=0), color='red', label='Mean Prediction')
plt.fill_between(X_new, 
                 np.percentile(trace['Y_pred'], 2.5, axis=0), 
                 np.percentile(trace['Y_pred'], 97.5, axis=0), 
                 color='blue', alpha=0.2, label='95% Credible Interval')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Bayesian Linear Regression Forecasting')
plt.legend()
plt.show()


