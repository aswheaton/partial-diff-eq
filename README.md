# partial-diff-eq
Numerical solutions to partial differential equations (PDEs) in discretised
space and time for initial value and boundary value problems. Contains two
classes, CH_Lattice and Poisson_Lattice, and one test script.

## Cahn-Hilliard Lattice

Simulates a water/oil mixture on a lattice. Uses the fast Fourier transform
(FFT) to perform Laplacian and gradient calculations in parallel, and step
the simulation forward using the Euler algorithm.

## Poisson Lattice

Simulates a water/oil mixture on a lattice. Uses either the Jacobi or the
Gauss-Seidel algorithm to step the simulation forwards, and then sets a
Dirchlecht boundary condition. The FFT is trivially utilised to step the
Jacobi algorithm forward in time. A three-dimensional ''checkerboard'' 
pattern is used to update the lattice using the Gauss-Seidel algorithm, so
that adjacent sites are updated using the FFT alternatingly.

## Usage
The test script `pde_test.py` illustrates the utility of these classes. It
is invoked with:

`python3 pde_test.py [l] [m] [n] [phi_0] [max_iter] [mode]`

The system arguments:

`l`, `m`, `n`: (int) the dimensions of the lattice. `n`
parameter is not used by Cahn-Hilliard class instances but a ''dummy''
parameter may be supplied. 50x50x50 is sufficiently large to illustrate the
behaviour of both classes.

`phi_0`: (float) intital scalar value with which to initalise the lattice.
Small random noise is added to this value on construction of an instance.

`max_iter`: (int) the maximum number of iterations to which the simulation
will be run before exiting. Simulations with a $-0.5 <= \phi_0 <= 0.5$ will
reach converge within 10,000 iterations. 

`mode`: (str) invokes one of several test cases for the simulation (see
descriptions below). Can be `oildrop`, `monopole`, `wires`, or `SOR`.

## Modes

`oildrop`: initalises and runs an animation of the Cahn-Hilliard oil/water
mixture. Values of $\phi_0 = \pm 0.5$ will demonstrate oil drop behaviour.
Records and plots the free energy of the simulation as a function of iterations.

`monopole`: initialises a Poisson_Lattice instance and solves the Poisson
equation using the Gauss-Seidel algorithm with over-relaxation for a single
charge at the center of the lattice. Once converged to the solution, plots the
a slice through the electric potential in the xy-plane and the resulting
electric field.

`wires`: reappropriates the Poisson_Lattice to solve one of Maxwell's
equation, with some small changes, using the Gauss-Seidel algorithm with over-
relaxation for a current density along the z-axis. Plots the magnetic vector
potential $A$ and the resulting magnetic field $B$.

`SOR`: tests a range of values for the over-relaxation parameter $omega$ and
finds the optimal value for convergence, for an electric monopole on the
Poisson_Lattice. Plots the iterations to convergence for each value of omega.
