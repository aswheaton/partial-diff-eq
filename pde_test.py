import sys
import time
import matplotlib.pyplot as plt
import numpy as np

from CH_Lattice import CH_Lattice
from Poisson_Lattice import Poisson_Lattice

l = int(sys.argv[1])
n = int(sys.argv[2])
m = int(sys.argv[3])
phi_0 = float(sys.argv[4])
max_iter=int(sys.argv[5])

# simulation = CH_Lattice(a=0.1, M=0.1, K=0.1, phi_0=phi_0,
#                         dx=1.0, dt=1.0, size=(n,m)
#                         )
# simulation.run(animate=False, max_iter=max_iter)

simulation = Poisson_Lattice(epsilon=1.0, phi_0=phi_0,
                             dx=1.0, dt=1.0, size=(l,n,m)
                             )
simulation.make_wire()
simulation.run(animate=False, max_iter=max_iter, tol=0.01)

plt.plot(range(len(simulation.free_energy)), simulation.free_energy)
plt.title("Free Energy for phi_0={}".format(phi_0))
plt.savefig("plots/free_energy_phi0={}.png".format(phi_0))
np.savetxt("data/free_energy_phi0={}.csv".format(phi_0), simulation.free_energy)
