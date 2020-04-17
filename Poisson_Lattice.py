import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import signal

class Poisson_Lattice(object):
    """
    Class for Poisson equation solution simulation.
    """

    def __init__(self, **kwargs):

        # Parameters for the simulation.
        self.epsilon = kwargs.get("epsilon")
        # Spatial and temporal resolution of the the simulation.
        self.dx = kwargs.get("dx")
        self.dt = kwargs.get("dt")
        # Initial value for the electric potential.
        self.phi_0 = kwargs.get("phi_0")
        # Construct the required lattices for simulation.
        self.size = kwargs.get("size")
        self.build()

    def build(self):
        """
        Generates two scalar fields and stores them as class attributes.
        """
        self.phi = self.phi_0 + np.random.uniform(-0.01,0.011, size=self.size)
        self.rho = np.zeros(self.size, dtype=float)

    def make_monopole(self):
        """
        Create monopole at center of the charge density lattice.
        """
        self.rho[self.size[0]//2, self.size[1]//2, self.size[2]//2] = 1.0

    def make_wire(self):
        """
        Create wire along the z-axis of the charge density lattice.
        """
        self.rho[self.size[0]//2, self.size[1]//2, :] = 1.0 / self.size[2]

    def set_dirchlect_boundary(self):
        self.phi[0,:,:], self.phi[-1,:,:] = 0.0, 0.0
        self.phi[:,0,:], self.phi[:,-1,:] = 0.0, 0.0
        self.phi[:,:,0], self.phi[:,:,-1] = 0.0, 0.0

    def conv_laplacian_3D(self, field):
        """
        Calculate the laplacian at every point on the lattice simultaneously
        by convolving with a kernel. Return the entire field of laplacians.
        """
        kernel = np.array([[0.0,1.0,0.0],[1.0,-4.0,1.0],[0.0,1.0,0.0]])
        return(signal.convolve2d(field, kernel, boundary='wrap', mode='same')/self.dx**2)

    def conv_gradient_3D(self, field):
        """
        Calculate the gradient at every point on the lattice simultaneously
        by convolving with a kernel. Return the entire field of gradient values.
        """
        kernel = np.array([[[0.0,0.0,0.0],
                            [0.0,0.0,0.0],
                            [0.0,0.0,0.0]],

                           [[0.0,0.0,0.0],
                            [-1.0,0.0,1.0],
                            [0.0,0.0,0.0]],

                           [[0.0,0.0,0.0],
                            [0.0,0.0,0.0],
                            [0.0,0.0,0.0]]
                          ]
                         )
        x_grad = signal.fftconvolve(field, kernel, mode='same')/(2.0*self.dx)
        kernel = np.array([[[0.0,0.0,0.0],
                            [0.0,0.0,0.0],
                            [0.0,0.0,0.0]],

                           [[0.0,-1.0,0.0],
                            [0.0,0.0,0.0],
                            [0.0,1.0,0.0]],

                           [[0.0,0.0,0.0],
                            [0.0,0.0,0.0],
                            [0.0,0.0,0.0]]
                          ]
                         )
        y_grad = signal.fftconvolve(field, kernel, mode='same')/(2.0*self.dx)
        kernel = np.array([[[0.0,0.0,0.0],
                            [0.0,-1.0,0.0],
                            [0.0,0.0,0.0]],

                           [[0.0,0.0,0.0],
                            [0.0,0.0,0.0],
                            [0.0,0.0,0.0]],

                           [[0.0,0.0,0.0],
                            [0.0,1.0,0.0],
                            [0.0,0.0,0.0]]
                          ]
                         )
        z_grad = signal.fftconvolve(field, kernel, mode='same')/(2.0*self.dx)
        return(x_grad, y_grad, z_grad)

    def jacobi_step(self):
        kernel = np.array([[[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.0]],
                           [[0.0,1.0,0.0],[1.0,0.0,1.0],[0.0,1.0,0.0]],
                           [[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.0]]])
        neighbors = signal.fftconvolve(self.phi, kernel, mode='same')
        phi_next = (neighbors + self.rho) / 6.0
        return(phi_next)

    def gauss_seidel_step(self, **kwargs):
        kernel = np.array([[[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.0]],
                           [[0.0,1.0,0.0],[1.0,0.0,1.0],[0.0,1.0,0.0]],
                           [[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.0]]])
        self.phi = (signal.fftconvolve(self.phi, kernel, mode='same') + self.rho) / 6.0

        # Over relax, if over-relaxation parameter provided.
        if "omega" in kwargs:
            self.phi += kwargs.get("omega")

    def electric_field_comp(self):
        """
        Calculate the x, y, and z components of the electric field in 3D space
        and return them as three, 3D arrays of x, y, and z components.
        """
        E_field = self.conv_gradient_3D(self.phi)
        # E_field = np.gradient(self.phi)
        return(-E_field[0], -E_field[1], -E_field[2])

    def step_forward(self, *args):
        """
        Steps the simulation forward one iteration using a specified
        integration algorithm.
        """
        self.phi = self.jacobi_step()
        # self.gauss_seidel_step()
        self.set_dirchlect_boundary()

        # Return an image object if the animation argument is enabled.
        if self.animate == True:
            self.steps += 1
            print("Simulation step {}...".format(self.steps), end="\r")
            self.image.set_array(self.phi)
            return(self.image,)

    def converged(self, ref_state, tolerance):
        """
        Check whether the solution has converged within some tolerance with
        respect to some reference state.
        """
        if np.sum(abs(self.phi-ref_state), axis=None) <= tolerance:
            return(True)
        else:
            return(False)

    def run(self, **kwargs):
        """
        Sets up a figure, image, and FuncAnimation instance, then runs the
        simulation to the specified maximum number of iterations.
        """
        # Boolean, whether or not step_forward() should return an image.
        self.animate = kwargs.get("animate")
        # Number of simulation steps using step_forward() method.
        max_iter = kwargs.get("max_iter")
        # Tolerance for convergence of the solution.
        tolerance = kwargs.get("tol")

        # Run the simulation using FuncAnimation if animate argument is enabled.
        if self.animate == True:
            self.figure = plt.figure()
            self.image = plt.imshow(self.phi, cmap='viridis', animated=True)
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
                print("Simulation step {} of {}...".format(step, max_iter), end="\r")
                ref_state = self.phi
                self.step_forward()
                if self.converged(ref_state, tolerance) == True:
                    print("\nConvergence after {} steps!".format(step))
                    break
                elif step == max_iter - 1:
                    print("\nReached max iterations w/o convergence!")
                else:
                    pass

        return(step)

    def export_animation(self, filename, dotsPerInch):
        """
        Exports the animation to a .gif file without compression. (Linux
        distributions with package "imagemagick" only. Files can be large!)
        # TODO: Add support for other image writing packages.
        """
        self.animation.save(filename, dpi=dotsPerInch, writer="imagemagick")
