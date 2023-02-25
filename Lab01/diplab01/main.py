import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 500  # simulation time (in years)
dt = 0.1  # time step (in years)
N = 10  # number of plant species
K = 100  # carrying capacity (same for all species)
r = np.random.normal(0.5, 0.1, N)  # intrinsic growth rate (random for each species)
alpha = np.random.uniform(0, 1, size=(N, N))  # competition/facilitation coefficient (random for each pair of species)
beta = np.random.uniform(0, 1, N)  # drought resistance coefficient (random for each species)

# Initial conditions
N0 = np.random.uniform(0, K, N)

# Simulate the system over time
t = np.arange(0, T, dt)
Nt = np.zeros((N, len(t)))
Nt[:, 0] = N0

for i in range(1, len(t)):
    # Compute the growth rate for each species
    dNdt = r * Nt[:, i-1] * (1 - np.sum(alpha * Nt[:, i-1], axis=1) / K)
    # Apply the drought stress
    dNdt *= np.exp(-beta * np.sin(2 * np.pi * t[i] / 20)**2)
    # Add random noise to the growth rate
    dNdt += np.random.normal(0, 0.05 * np.mean(r), N)
    # Update the population size for each species
    Nt[:, i] = np.maximum(0, Nt[:, i-1] + dNdt * dt)

# Plot the results
plt.figure()
for i in range(N):
    plt.plot(t, Nt[i, :], label='Species {}'.format(i+1))
plt.xlabel('Time (years)')
plt.ylabel('Population size')
plt.legend()
plt.show()
