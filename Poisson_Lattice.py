import numpy as np
import matplotlib.pyplot as plt
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
        Create wire along the z-axis of the lattice. For use only when the
        scalar field attribute phi is actually used to contain a current density
        along the z axis.
        """
        self.rho[self.size[0]//2, self.size[1]//2, :] = 1.0 / self.size[2]

    def set_omega(self, omega):
        """
        Set the over-relaxation parameter to something other than one, for use
        in the Gauss-Seidel update algorithm.
        """
        self.omega = omega

    def set_dirchlect_boundary(self):
        """
        Enforce the Dirchlecht boundary condition by setting the faces of the
        data cube to zero along each axis.
        """
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
        DEPRECATED
        Calculate the gradient at every point on the lattice simultaneously
        by convolving with a kernel. Return the entire field of gradient values.
        """
        kernel = np.array([[[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]],
                           [[0.0,0.0,0.0],[-1.0,0.0,1.0],[0.0,0.0,0.0]],
                           [[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]]
                          ])
        x_grad = signal.fftconvolve(field, kernel, mode='same')/(2.0*self.dx)
        kernel = np.array([[[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]],
                           [[0.0,-1.0,0.0],[0.0,0.0,0.0],[0.0,1.0,0.0]],
                           [[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]]
                          ])
        y_grad = signal.fftconvolve(field, kernel, mode='same')/(2.0*self.dx)
        kernel = np.array([[[0.0,0.0,0.0],[0.0,-1.0,0.0],[0.0,0.0,0.0]],
                           [[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]],
                           [[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.0]]
                          ])
        z_grad = signal.fftconvolve(field, kernel, mode='same')/(2.0*self.dx)
        return(x_grad, y_grad, z_grad)

    def jacobi_step(self):
        """
        Calculate and return the next iteration of the scalar array using the
        Jacobi algorithm, by convolving with a 3D kernel to get the nearest
        neighbors summation.
        """
        kernel = np.array([[[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.0]],
                           [[0.0,1.0,0.0],[1.0,0.0,1.0],[0.0,1.0,0.0]],
                           [[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.0]]])
        neighbors = signal.fftconvolve(self.phi, kernel, mode='same')
        phi_next = (neighbors + self.rho) / 6.0
        return(phi_next)

    def gauss_seidel_step(self):
        """
        Calculate and return the next iteration of the scalar array using the
        Gauss-Seidel algorithm with the option using an over-relaxation
        parameter if available.

        Scalar array is updated in a three dimensional "checkerboard" pattern
        using the Fast Fourier Transform to update alternating sites in parallel.
        """
        # Use over-relaxation parameter, if provided, and otherwise set to 1.0.
        try:
            omega = self.omega
        except AttributeError:
            omega = 1.0

        # Create alternating indices for easier indexing in checkerboard pattern.
        checker_tile = np.array([[[0,1],[1,0]],[[1,0],[0,1]]])
        checkerboard = np.tile(checker_tile, (self.size[0]//2,self.size[1]//2,self.size[2]//2))
        black_squ = np.where(checkerboard==1)
        white_squ = np.where(checkerboard==0)

        # Initialise the nearest neighbors kernel for convolution.
        kernel = np.array([[[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.0]],
                           [[0.0,1.0,0.0],[1.0,0.0,1.0],[0.0,1.0,0.0]],
                           [[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.0]]])

        # Calculate and update the "black" sites and "white" sites in turn.
        neighbors = signal.fftconvolve(self.phi, kernel, mode='same')
        self.phi[black_squ] = (((neighbors[black_squ] + self.rho[black_squ]) / 6.0) - self.phi[black_squ]) * omega + self.phi[black_squ]
        neighbors = signal.fftconvolve(self.phi, kernel, mode='same')
        self.phi[white_squ] = (((neighbors[white_squ] + self.rho[white_squ]) / 6.0) - self.phi[white_squ]) * omega + self.phi[white_squ]

    def electric_field_comp(self):
        """
        Calculate the x, y, and z components of the electric field in 3D space
        and return them as three, 3D arrays of x, y, and z components.
        """
        # E_field = self.conv_gradient_3D(self.phi)
        E_field = np.gradient(self.phi)
        return(-E_field[0], -E_field[1], -E_field[2])

    def magnetic_vector_potential(self):
        """
        Calculate the x, y, and z components of the magnetic potential in 3D
        space and return them as three, 3D arrays of x, y, and z components.

        For use only when the scalar field attribute phi is actually used to
        contain a current density along the z axis.
        """
        A_field = np.gradient(self.phi)
        return(-A_field[0], -A_field[1], -A_field[2])

    def magnetic_field_comp(self):
        """
        Calculate the x, y, and z components of the magnetic field in 3D space
        from the magentic potential and return them as three, 3D arrays of x, y,
        and z components.

        For use only when the scalar field attribute phi is actually used to
        contain a current density along the z axis.
        """
        a_field_x, a_field_y, a_field_z = self.magnetic_vector_potential()
        # Calculate and return the curl of the vector potential field.
        B_field_x = a_field_y - a_field_z
        B_field_y = a_field_z - a_field_x
        B_field_z = a_field_x - a_field_y
        return(B_field_x, B_field_y, B_field_z)


    def step_forward(self):
        """
        Steps the simulation forward one iteration using a specified
        integration algorithm, and enforces the boundary condition.
        """
        if self.algorithm == "jacobi":
            self.phi = self.jacobi_step()
        elif self.algorithm == "gauss-seidel":
            self.gauss_seidel_step()

        self.set_dirchlect_boundary()

    def is_converged(self, ref_state, tolerance):
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
        # Number of simulation steps using step_forward() method.
        max_iter = kwargs.get("max_iter")
        # Tolerance for convergence of the solution.
        tolerance = kwargs.get("tol")
        # Which update algorithm to use, can be "jacobi" or "gauss-seidel"
        self.algorithm = kwargs.get("alg")

        # Run the simulation, manually updating the simulation with step_forward() method.
        for step in range(max_iter):
            print("Simulation step {} of {}...".format(step, max_iter), end="\r")
            ref_state = np.array(self.phi)
            self.step_forward()
            if self.is_converged(ref_state, tolerance) == True:
                print("\nConvergence after {} steps!".format(step))
                break
            elif step == max_iter - 1:
                print("\nReached max iterations w/o convergence!")
            else:
                pass

        # Return the number of the last iteration before convergence.
        return(step)

    def export_animation(self, filename, dotsPerInch):
        """
        Exports the animation to a .gif file without compression. (Linux
        distributions with package "imagemagick" only. Files can be large!)
        # TODO: Add support for other image writing packages.
        """
        self.animation.save(filename, dpi=dotsPerInch, writer="imagemagick")
