import numpy as np
import matplotlib.pyplot as plt
import json

# Load in the data
dt = np.loadtxt('output/convergence.dt')
convergence_data = np.loadtxt('output/convergence.out').reshape(-1, 6, 3)

# Compute the strong errors
diffs = np.linalg.norm(np.diff(convergence_data, 1, axis=1), axis=2)
ensemble_diff = np.mean(diffs, axis=0)

# Plot the convergence
plt.loglog(dt[1:], ensemble_diff)
plt.xlabel('dt')
plt.ylabel('error')
plt.savefig('output/convergence.svg')

# Compute the convergence rate
A = np.vstack([np.ones_like(ensemble_diff), np.log(dt[1:])]).T
b = np.log(ensemble_diff)
least_square_res = np.linalg.lstsq(A, b)
rate = least_square_res[0][1]

# Save results
results = {
    'convergence_plot': 'convergence.svg',
    'convergence_rate': rate
}
with open('output/results.json', 'w') as f:
    json.dump(results, f)
