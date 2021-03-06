import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import signal

class CH_Lattice(object):
    """
    Class for Cahn-Hilliard phase separation simulation.
    """

    def __init__(self, **kwargs):

        # Parameters for the simulation. What are their physical meanings?
        self.a = kwargs.get("a")
        self.M = kwargs.get("M")
        self.K = kwargs.get("K")
        # Spatial and temporal resolution of the the simulation.
        self.dx = kwargs.get("dx")
        self.dt = kwargs.get("dt")
        # Initial value for compositional order parameter.
        self.phi_0 = kwargs.get("phi_0")
        # Empty list for storing free energy.
        self.free_energy = []
        # Construct the required lattices for simulation.
        self.size = kwargs.get("size")
        self.build()

    def build(self):
        """
        Generates two scalar fields and stores them as class attributes.
        """
        self.phi = self.phi_0 + np.random.uniform(-1.,1., size=self.size)
        self.mu = self.chemical_potential()

    def disc_laplacian(self, lattice, indices):
        """
        DEPRECATED
        Recieves a lattice of scalar values and pair of indices, calculates
        and returns the discretised laplacian at the corresponding point on
        the lattice using a centered difference estimate.
        # TODO: generalise this to n dimensions!
        """
        i, j = indices
        laplacian_x = (lattice[self.bc((i+1,j))] + lattice[self.bc((i-1,j))] - 2 * lattice[i,j]) / self.dx**2
        laplacian_y = (lattice[self.bc((i,j+1))] + lattice[self.bc((i,j-1))] - 2 * lattice[i,j]) / self.dx**2
        return(laplacian_x + laplacian_y)

    def disc_gradient(self, lattice, indices):
        """
        DEPRECATED
        Recieves a lattice of scalar values and pair of indices, calculates
        and returns the discretised gradient at the corresponding point on
        the lattice using a centered difference estimate.
        # TODO: generalise this to n dimensions!
        """
        i, j = indices
        gradient_x = (lattice[self.bc((i+1,j))] - lattice[self.bc((i-1,j))]) / (2.0 * self.dx)
        gradient_y = (lattice[self.bc((i,j+1))] - lattice[self.bc((i,j-1))]) / (2.0 * self.dx)
        return(gradient_x + gradient_y)

    def conv_laplacian(self, field):
        """
        Calculate the laplacian at every point on the lattice simultaneously
        by convolving with a kernel. Return the entire field of laplacians.
        """
        kernel = np.array([[0.0,1.0,0.0],[1.0,-4.0,1.0],[0.0,1.0,0.0]])
        return(signal.convolve2d(field, kernel, boundary='wrap', mode='same') / self.dx**2)

    def conv_gradient(self, field):
        """
        DEPRECATED
        Calculate the gradient at every point on the lattice simultaneously
        by convolving with a kernel. Return the entire field of gradient values.
        """
        kernel = np.array([[0.0,-1.0,0.0],[-1.0,0.0,1.0],[0.0,1.0,0.0]])
        return(signal.convolve2d(field, kernel, boundary='wrap', mode='same') / (2.0 * self.dx))

    def chemical_potential(self):
        """
        Recieves a pair of indices, calculates and returns a scalar value of
        the chemical potential of phi at the corresponding point on the lattice.
        """
        chem_potential = (- self.a * self.phi
                          + self.a * self.phi**3
                          - self.K * self.conv_laplacian(self.phi)
                          )
        return(chem_potential)

    def free_energy_density(self):
        """
        Recieves a pair of indices, calculates and returns the free energy
        density of the system at the corresponding point on the lattice.
        """
        free_energy_density = (-(self.a/2.0) * self.phi**2
                               +(self.a/4.0) * self.phi**4
                               +(self.K/2.0) * self.conv_gradient(self.phi)**2
                               )
        return(free_energy_density)

    def euler_step(self):
        """
        Recieves a pair of indices, calculates and returns a scalar value of
        phi for the next iteration of the simulation at the corresponding
        point on the lattice, using the (explicit) Euler algorithm.
        """

        phi_next = self.phi + ((self.M * self.dt / self.dx**2) * self.conv_laplacian(self.mu))
        return(phi_next)

    def step_forward(self, *args):
        """
        Steps the simulation forward one iteration using a specified
        integration algorithm.
        """
        self.phi = self.euler_step()
        self.mu = self.chemical_potential()
        self.free_energy.append(np.sum(self.free_energy_density()))

        # Return an image object if the animation argument is enabled.
        if self.animate == True:
            self.steps += 1
            print("Simulation step {}...".format(self.steps), end="\r"),
            self.image.set_array(self.phi)
            return(self.image,)

    def run(self, **kwargs):
        """
        Sets up a figure, image, and FuncAnimation instance, then runs the
        simulation to the specified maximum number of iterations.
        """

        # Boolean, whether or not step_forward() should return an image.
        self.animate = kwargs.get("animate")
        # Number of simulation steps using step_forward() method.
        max_iter = kwargs.get("max_iter")

        # Run the simulation using FuncAnimation if animate argument is enabled.
        if self.animate == True:
            self.figure = plt.figure()
            self.image = plt.imshow(self.phi, cmap='seismic', interpolation='bicubic', animated=True)
            plt.colorbar()
            self.steps = 0
            self.animation = animation.FuncAnimation(self.figure,
                                                     self.step_forward,
                                                     frames=max_iter,
                                                     repeat=False,
                                                     interval=75, blit=True
                                                     )
            plt.show()
            plt.clf()

        # Otherwise, manually update the simulation with step_forward() method.
        elif self.animate == False:
            for step in range(max_iter):
                print("Simulation step {} of {}...".format(step, max_iter), end="\r"),
                self.step_forward()
            print()

    def export_animation(self, filename, dotsPerInch):
        """
        Exports the animation to a .gif file without compression. (Linux
        distributions with package "imagemagick" only. Files can be large!)
        # TODO: Add support for other image writing packages.
        """
        self.animation.save(filename, dpi=dotsPerInch, writer="imagemagick")
