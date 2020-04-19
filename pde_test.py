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
mode = sys.argv[6]

def plot_vector_field_2D(x_comp, y_comp):
    """
    Code which flattens input arrays and their indices. Required to make
    matplotlib.pyplot.quiver plot work properly, unfortunately.
    """
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

    plt.quiver(i_vals, j_vals, e_vals_x, e_vals_y)

    # Return a flattened array which can be exported to a gnuplot processable format.
    stacked_matrix = np.stack([i_vals, j_vals, e_vals_x, e_vals_y], axis=1)
    return(stacked_matrix)

def main():

    if (l==m) and (m==n):
        slice_index = l // 2
    else: # Testing case.
        slice_index = 25

    if mode == "oilplot":
        simulation = CH_Lattice(a=0.1, M=0.1, K=0.1, phi_0=phi_0,
                                dx=1.0, dt=1.0, size=(n,m)
                                )
        simulation.run(animate=False, max_iter=max_iter)
        # np.savetxt("data/free_energy_phi0={}.csv".format(phi_0), simulation.free_energy)
        # simulation.free_energy = np.loadtxt("data/free_energy_phi0={}.csv".format(phi_0))
        plt.plot(range(len(simulation.free_energy)), simulation.free_energy)
        plt.title("Free Energy for phi_0={}".format(phi_0))
        plt.savefig("plots/free_energy_phi_0={}.png".format(phi_0))

    elif mode == "monopole":
        e_simulation = Poisson_Lattice(epsilon=1.,phi_0=0.,dx=1.,dt=1., size=(l,n,m))
        e_simulation.make_monopole()
        e_simulation.run(animate=False, max_iter=max_iter, tol=0.01)

        phi_slice_xy = e_simulation.phi[:,:,slice_index]
        phi_slice_yz = e_simulation.phi[slice_index,:,:]
        phi_slice_xz = e_simulation.phi[:,slice_index,:]
        plt.imshow(phi_slice_xy, origin="lower", cmap="viridis")
        plt.colorbar()
        plt.show()
        plt.clf()

        efield_x, efield_y, efield_z = e_simulation.electric_field_comp()
        xy_plane_x = efield_x[:,:,slice_index]
        xy_plane_y = efield_y[:,:,slice_index]
        plt.imshow(np.sqrt(xy_plane_x**2+xy_plane_y**2), origin="lower", cmap="viridis")
        plt.colorbar()
        plot_vector_field_2D(xy_plane_x, xy_plane_y)
        plt.title("Electric Field Direction & Magnitude")
        plt.show()
        plt.clf()

    elif mode == "wires":

        m_simulation = Poisson_Lattice(epsilon=1.,phi_0=0.,dx=1.,dt=1., size=(l,n,m))
        m_simulation.make_wire()
        m_simulation.run(animate=False, max_iter=max_iter, tol=0.01)

        mag_pot_x, mag_pot_y, mag_pot_z = m_simulation.magnetic_vector_potential()
        mag_pot_x_xy = mag_pot_x[:,:,slice_index]
        mag_pot_y_xy = mag_pot_y[:,:,slice_index]

        plt.imshow(np.sqrt(mag_pot_x_xy**2+mag_pot_y_xy**2), origin="lower", cmap="viridis")
        plt.colorbar()
        mag_field_x, mag_field_y, mag_field_z = m_simulation.magnetic_field_comp()
        mag_field_x_xy = mag_field_x[:,:,slice_index]
        mag_field_y_xy = mag_field_y[:,:,slice_index]
        plot_vector_field_2D(mag_field_x_xy, mag_field_y_xy)
        plt.title("Magnetic Field Direction & Magnetic Potential Magnitude")
        plt.show()
        plt.clf()

    elif mode == "SOR":

        start, stop, steps = 1.0, 1.95, 96
        iters_to_conv = np.zeros(steps)
        omega_vals = np.linspace(start,stop,steps)

        for omega in omega_vals:
            e_simulation = Poisson_Lattice(epsilon=1.,phi_0=0.,dx=1.,dt=1., size=(l,n,m))
            e_simulation.make_monopole()
            e_simulation.set_omega(omega)
            iters = e_simulation.run(animate=False, max_iter=max_iter, tol=0.01)
            index = np.where(omega_vals==omega)
            iters_to_conv[index] = iters

        # stacked_matrix = np.stack([omega_vals, iters_to_conv], axis=1)
        # np.savetxt("data/iters_to_conv_phi_0={}.csv".format(phi_0), stacked_matrix)

        plt.plot(np.linspace(start,stop,steps), iters_to_conv, "or")
        plt.xlabel("Over-relaxation Parameter (omega)")
        plt.ylabel("Iterations to Convergence")
        min_index = np.where(iters_to_conv==np.min(iters_to_conv))[0]
        plt.title("phi=0.5, Minimum at omega={}".format(omega[min_index][0]))
        plt.savefig("plots/sor_phi0={}.png".format(phi_0))
        plt.show()
        plt.clf()
main()
