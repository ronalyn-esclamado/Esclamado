# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 20:30:02 2024

@authors: abalora_dellavan_ellorda_esclamado
"""

import scipy.stats as sts
import numpy as np
import matplotlib.pyplot as plt

mu = np.linspace(1.65, 1.8, num = 50)
test = np.linspace(0, 2)
uniform_dist = sts.uniform.pdf(mu) + 1 #sneaky advanced note: I'm using the uniform distribution for clarity,
                                        #but we can also make the beta distribution look completely flat by tweaking alpha and beta!
uniform_dist = uniform_dist/uniform_dist.sum()

def likelihood_func(datum, mu):
 likelihood_out = sts.norm.pdf(datum, mu, scale = 0.1)    
 return likelihood_out/likelihood_out.sum() 

likelihood_out = likelihood_func(1.7, mu)
unnormalized_posterior = likelihood_out * uniform_dist
plt.plot(mu, unnormalized_posterior)
plt.xlabel("$\mu$ in meters")
plt.ylabel("Unnormalized Posterior")
plt.show()
