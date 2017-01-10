import json
import numpy as np
from scipy.integrate import trapz
import matplotlib.pyplot as plt

mz = np.loadtxt('output/equilibrium.out')
mz = mz/max(mz) # account for error in solver
theta = np.arccos(mz)

# Stoner Wohlfarth equation
def sw(t, h):
    return 0.5*np.power(np.sin(t), 2) + h*np.cos(t)

# Analytic formula for the Boltzmann distribution
def boltz(ts, sigma, h):
    ans = np.sin(t)*np.exp(-2*sigma*sw(ts, h))
    t2 = np.linspace(0, np.pi/2, 10000)
    norm = trapz(t2, np.sin(t2)*np.exp(-2*sigma*sw(t2, h)))
    return -ans/norm

with open('output/equilibrium_normalised.json', 'r') as f:
    config = json.load(f)

# plot
t = np.linspace(0, 0.5, 1000)
analytic = boltz(t, config['particle']['stability-ratio'], 0)
plt.hist(theta, bins=50, normed=True, label='moma')
plt.plot(t, analytic, 'r', lw=2, label='analytic')
plt.xlabel('theta')
plt.legend(loc='upper right')
plt.savefig('output/equilibrium.svg')
