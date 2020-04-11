import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
        # Boolean, whether or not step_forward() should return an image to be drawn.
        self.animate = kwargs.get("animate")
        # Construct the required lattices for simulation.
        self.size = kwargs.get("size")
        self.build()

    def build(self):
        """
            Generates two scalar fields and stores them as class attributes.
        """

        self.phi = np.random.choice(a=[-1.0,1.0], size=self.size) * np.random.random(self.size)
        self.mu = self.gen_lattice(self.size, self.chemical_potential)

    def bc(self, indices):
        """
            Determines if a pair of indices falls outside the boundary of the
            lattice and if so, applies a periodic (toroidal) boundary condition
            to return new indices.
        """
        return((indices[0]%self.size[0], indices[1]%self.size[1]))

    def disc_laplacian(self, lattice, indices):
        """
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
            Recieves a lattice of scalar values and pair of indices, calculates
            and returns the discretised gradient at the corresponding point on
            the lattice using a centered difference estimate.
            # TODO: generalise this to n dimensions!
        """
        i, j = indices
        gradient_x = (lattice[self.bc((i+1,j))] - lattice[self.bc((i-1,j))]) / (2.0 * self.dx)
        gradient_y = (lattice[self.bc((i,j+1))] - lattice[self.bc((i,j-1))]) / (2.0 * self.dx)
        return(gradient_x + gradient_y)


    def chemical_potential(self, indices):
        """
            Recieves a pair of indices, calculates and returns a scalar value of
            the chemical potential of phi at the corresponding point on the lattice.
        """
        chem_potential = (- self.a * self.phi[indices]
                          + self.a * self.phi[indices]**3
                          - self.K * self.disc_laplacian(self.phi, indices)
                          )
        return(chem_potential)

    def free_energy_density(self, indices):
        """
            Recieves a pair of indices, calculates and returns the free energy
            density of the system at the corresponding point on the lattice.
        """
        free_energy_density = (-(self.a/2.0) * self.phi[indices]**2
                               +(self.a/4.0) * self.phi[indices]**4
                               +(self.K/2.0) * self.disc_gradient(self.phi, indices)**2
                               )
        return(free_energy_density)

    def euler_step(self, indices):
        """
            Recieves a pair of indices, calculates and returns a scalar value of
            phi for the next iteration of the simulation at the corresponding
            point on the lattice, using the (explicit) Euler algorithm.
        """

        i, j = indices
        nn_sum = (self.mu[self.bc((i-1,j))] + self.mu[self.bc((i+1,j))]
                  + self.mu[self.bc((i,j-1))] + self.mu[self.bc((i,j+1))]
                  - 4 * self.mu[i,j]
                  )
        next_value = self.phi[i,j] + (self.M * self.dt / self.dx**2) * nn_sum
        return(next_value)

    def jacobi_step(self):
        pass

    def gauss_seidel_step(self):
        pass

    def gen_lattice(self, shape, func):
        """
            Generate a new lattice of specified shape using a specified function
            handle to a function or method which recieves a pair of indices and
            returns a scalar value.
        """
        new_lattice = np.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                new_lattice[i,j] = func((i,j))
        return(new_lattice)

    def step_forward(self, *args):
        """
            Steps the simulation forward one iteration using a specified
            integration algorithm.
        """
        # algorithm = kwargs.get("algorithm")
        self.phi = self.gen_lattice(self.size, self.euler_step)
        self.mu = self.gen_lattice(self.size, self.chemical_potential)

        # self.free_energy.append(np.sum(self.gen_lattice(self.size, self.free_energy_density)))

        if self.animate == True:
            self.steps += 1
            print("Simulation step {}".format(self.steps), end="\r"),
            self.image.set_array(self.phi)
            return(self.image,)

    def run(self, **kwargs):
        """
            Sets up a figure, image, and FuncAnimation instance, then runs the
            simulation to the specified maximum number of iterations.
        """

        max_iter = kwargs.get("max_iter")

        self.free_energy = []

        self.figure = plt.figure()
        self.image = plt.imshow(self.phi, cmap='viridis', animated=True)

        self.steps = 0
        self.animation = animation.FuncAnimation(self.figure, self.step_forward,
                                                 frames=max_iter, repeat=False,
                                                 interval=75, blit=True
                                                 )
        plt.show()
        plt.clf()

        plt.plot(range(len(self.free_energy)), self.free_energy)
        plt.savefig("plots/free_energy.png")
        plt.clf()

    def export_animation(self, filename, dotsPerInch):
        """
            Exports the animation to a .gif file without compression. (Linux
            distributions with package "imagemagick" only. Files can be large!)
            # TODO: Add support for other image writing packages.
        """
        self.animation.save(filename, dpi=dotsPerInch, writer="imagemagick")
