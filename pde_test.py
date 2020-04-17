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

    long_list_size = x_comp.shape[0]*x_comp.shape[1]
    i_vals = np.zeros(long_list_size, dtype=int)
    j_vals = np.zeros(long_list_size, dtype=int)
    e_vals_x = np.zeros(long_list_size, dtype=float)
    e_vals_y = np.zeros(long_list_size, dtype=float)
    counter = 0
    for i in range(x_comp.shape[0]):
        for j in range(x_comp.shape[1]):
            i_vals[counter] = i
            j_vals[counter] = j
            e_vals_x[counter] = x_comp[i,j]
            e_vals_y[counter] = y_comp[i,j]
            counter += 1

    plt.imshow(np.sqrt(x_comp**2+y_comp**2), origin="lower", cmap="viridis")
    plt.colorbar()
    plt.quiver(i_vals, j_vals, e_vals_x, e_vals_y)
    plt.show()
    plt.clf()


def main():
    # simulation = CH_Lattice(a=0.1, M=0.1, K=0.1, phi_0=phi_0,
    #                         dx=1.0, dt=1.0, size=(n,m)
    #                         )
    # simulation.run(animate=False, max_iter=max_iter)

    simulation = Poisson_Lattice(epsilon=1.,phi_0=0.,dx=1.,dt=1., size=(l,n,m))
    simulation.make_monopole()
    simulation.run(animate=False, max_iter=max_iter, tol=0.01)

    slice_index = 25

    phi_slice_xy = simulation.phi[:,:,slice_index]
    phi_slice_yz = simulation.phi[slice_index,:,:]
    phi_slice_xz = simulation.phi[:,slice_index,:]
    # fig, axes = plt.subplots(1,3)
    # axes[0].imshow(phi_slice_xy, origin="lower", cmap="viridis")
    # axes[0].set_title("xy-plane")
    # axes[1].imshow(phi_slice_yz, origin="lower", cmap="viridis")
    # axes[1].set_title("yz-plane")
    # axes[2].imshow(phi_slice_xz, origin="lower", cmap="viridis")
    # axes[2].set_title("xz-plane")
    # pcm = axes.pcolormesh(np.random.random((20, 20)) * (col + 1), cmap=cm[col])
    plt.imshow(phi_slice_xy, origin="lower", cmap="viridis")
    plt.colorbar()
    plt.show()
    plt.clf()

    efield_x, efield_y, efield_z = simulation.electric_field_comp()
    xy_plane_x = efield_x[:,:,slice_index]
    xy_plane_y = efield_y[:,:,slice_index]

    # stacked_matrix = np.stack([i_vals, j_vals, e_vals_x, e_vals_y], axis=1)
    # np.savetxt("data/e_xy_monopole.csv", stacked_matrix)

    plot_vector_field_2D(xy_plane_x, xy_plane_y)

    # plt.plot(range(len(simulation.free_energy)), simulation.free_energy)
    # plt.title("Free Energy for phi_0={}".format(phi_0))
    # plt.savefig("plots/free_energy_phi0={}.png".format(phi_0))
    # np.savetxt("data/free_energy_phi0={}.csv".format(phi_0), simulation.free_energy)

main()
