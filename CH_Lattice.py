import numpy as np
import matplotlib.pyplot as plt

class CH_Lattice(object):
    """
        Class for Cahn-Hilliard phase separation simulation.
    """

    def __init__(self, **kwargs):

        # Parameters for the simulation. What are their physical meanings?
        self.a = kwargs.get("a")
        self.M = kwargs.get("M")
        self.K = kwargs.get("K")
        # Boolean, whether or not step_forward() should return an image to be drawn.
        self.animate = kwargs.get("animate")
        # Construct the required lattices for simulation.
        self.size = kwargs.get("size")
        self.build()

    def build(self):
        """
            Generates two scalar fields and stores them as class attributes.
        """

        self.phi = np.random.choice(a=[-1.0,0.0,1.0], size=self.size)
        self.mu = self.gen_lattice(self.size, self.chemical_potential)

    def disc_laplacian(self, lattice, indices):
        """
            Recieves a lattice of scalar values and pair of indices, calculates
            and returns the discretised laplacian at the corresponding point on
            the lattice using a centered difference estimate.
            # TODO: generalise this to n dimensions!
        """
        i, j = indices
        laplacian_x = lattice[i+1,j] + lattice[i-1,j] - 2 * lattice[i,j]
        laplacian_y = lattice[i,j+1] + lattice[i,j-1] - 2 * lattice[i,j]
        laplacian = (laplacian_x + laplacian_y) / dx**2
        return(laplacian)

    def disc_gradient(self, lattice, indices):
        """
            Recieves a lattice of scalar values and pair of indices, calculates
            and returns the discretised gradient at the corresponding point on
            the lattice using a centered difference estimate.
            # TODO: generalise this to n dimensions!
        """
        i, j = indices
        gradient_x = lattice[i+1,j] - lattice[i-1,j]
        gradient_y = lattice[i,j+1] - lattice[i,j-1]
        gradient = (gradient_x + gradient_y) / (2.0 * dx)
        return(gradient)

    def chemical_potential(self, indices):
        """
            Recieves a pair of indices, calculates and returns a scalar value of
            the chemical potential of phi at the corresponding point on the lattice.
        """
        chem_potential = (-self.a * self.phi[indices]
                          + self.a * self.phi[indices]**3
                          - self.kappa * self.disc_laplacian(self.phi, indices)
                          )
        return(chem_potential)

    def free_energy_density(self, indices):
        """
            Recieves a pair of indices, calculates and returns the free energy
            density of the system at the corresponding point on the lattice.
        """
        free_energy_density = (-(self.a/2.0) * self.phi[indices]**2
                               +(self.a/4.0) * self.phi[indices]**4
                               +(self.K/2.0) * disc_gradient(self.phi, indices)**2
                               )


    def euler_step(self, indices):
        """
            Recieves a pair of indices, calculates and returns a scalar value of
            phi for the next iteration of the simulation at the corresponding
            point on the lattice, using the (explicit) Euler algorithm.
        """

        i, j = indices
        nn_sum = (self.mu[i-1,j] + self.mu[i+1,j]
                  + self.mu[i,j-1] + self.mu[i,j+1]
                  - 4 * self.mu[i,j]
                  )
        phi_next = self.phi[i,j] + self.M * (dt / dx**2) * nn_sum
        return(phi_next)

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
                new_lattice[i,j] = func(indices)
        return(new_lattice)

    def step_forward(self, **kwargs):
        """
            Steps the simulation forward one iteration using a specified
            integration algorithm.
        """
        algorithm = kwargs.get("algorithm")
        self.phi = gen_lattice(self.size, algorithm)
        self.mu = gen_lattice(self.size, chemical_potential)

        if self.animate == True:
            self.image.set_array(self.lattice)
            return(self.image,)

    def run(self, **kwargs):
        """
            Sets up a figure, image, and FuncAnimation instance, then runs the
            simulation to the specified maximum number of iterations.
        """

        max_iter = kwargs.get("max_iter")

        self.figure = plt.figure()
        self.image = plt.imshow(self.lattice, animated=True)
        self.animation = animation.FuncAnimation(self.figure, self.step_forward,
                                                 frames=max_iter, repeat=False,
                                                 interval=100, blit=True
                                                 )
        plt.show()

    def export_animation(self):
        """
            Exports the animation to a .gif file without compression. (Linux
            distributions with package "imagemagick" only. Files can be large!)
            # TODO: Add support for other image writing packages.
        """
        self.animation.save(filename, dpi=dotsPerInch, writer="imagemagick")
