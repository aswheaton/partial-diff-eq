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

def plot_vector_field_2D(x_comp, y_comp):
    # TODO: Check that arrays have same dimensions.
    x_axis = np.array(range(x_comp.shape[0]))
    y_axis = np.array(range(x_comp.shape[0]))
    plt.quiver(x_axis, y_axis, x_comp, y_comp)#, angles="xy")
    plt.show()
    plt.clf()


def main():
    # simulation = CH_Lattice(a=0.1, M=0.1, K=0.1, phi_0=phi_0,
    #                         dx=1.0, dt=1.0, size=(n,m)
    #                         )
    # simulation.run(animate=False, max_iter=max_iter)

    simulation = Poisson_Lattice(epsilon=1.,phi_0=0.,dx=1.,dt=1., size=(l,n,m))
    simulation.make_wire()
    simulation.run(animate=False, max_iter=max_iter, tol=0.01)

    phi_slice_xy = simulation.phi[:,:,25]
    phi_slice_yz = simulation.phi[25,:,:]
    phi_slice_xz = simulation.phi[:,25,:]
    fig, axes = plt.subplots(1,3)
    axes[0].imshow(phi_slice_xy, origin="lower", cmap="viridis")
    axes[0].set_title("Electric Potential xy-plane")
    axes[1].imshow(phi_slice_yz, origin="lower", cmap="viridis")
    axes[1].set_title("Electric Potential yz-plane")
    axes[2].imshow(phi_slice_xz, origin="lower", cmap="viridis")
    axes[2].set_title("Electric Potential xz-plane")
    plt.show()
    plt.clf()

    efield_x, efield_y, efield_z = simulation.electric_field_comp()
    xy_plane_x = efield_x[:,:,25]
    xy_plane_y = efield_y[:,:,25]
    plot_vector_field_2D(xy_plane_x, xy_plane_y)

    # plt.plot(range(len(simulation.free_energy)), simulation.free_energy)
    # plt.title("Free Energy for phi_0={}".format(phi_0))
    # plt.savefig("plots/free_energy_phi0={}.png".format(phi_0))
    # np.savetxt("data/free_energy_phi0={}.csv".format(phi_0), simulation.free_energy)

main()
