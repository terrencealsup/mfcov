import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
from scipy.special import factorial
#from tqdm import tqdm


def forwardModel(z, n, n_obs):
    """
    Evaluate the forward model at the parameters z using n grid points.

    args:
    z - numpy.ndarray(N, d), N is the number of parameters, d is the dimension
    n - int, the number of grid points to use in the finite difference approximation
    n_obs - int, the dimension of the observation vector

    returns:
    fwd - numpy.ndarray(N, n_obs), the observation vector for each parameter z[i]
    """
    
    if z.ndim == 1:
        z = z.reshape((1, len(z)))
    
    N, d = z.shape
    # Shape of the output
    fwd = np.zeros((N, n_obs))
    x_obs = np.linspace(0, 1, n_obs+2)[1:n_obs+1]

    # Set the boundary conditions and forcing function
    u0 = 0
    un = 1
    f = lambda x: np.ones(x.shape)

    # Set up the finite difference grid and staggered grid
    x = np.linspace(0, 1, n+1)
    h = 1/n           # Mesh width
    xs = x[:n] + h/2  # Staggered grid

    # Loop over all parameters
    for i in range(N):
        zi = z[i]
        # Evaluate the diffusion coefficient on the staggered grid
        # kappa[0] = \kappa_{1/2},...,kappa[n-1] = \kappa_{n-1/2}
        # kappa[i] = \kappa_{i + 1/2} for i = 0,...,n-1
        #log_kappa = np.polyval(zi/factorial(np.arange(d-1,-1,-1)), xs)
        #log_kappa = np.polyval(zi/factorial(np.arange(d)), xs)
        log_kappa = np.matmul(zi, np.sin(2*np.pi*np.outer(np.arange(1, d+1), xs)))
        kappa = np.exp(log_kappa)
        # Construct the LHS of the discretized equation
        B = np.zeros((3, n-1))
        B[0] = np.insert(-kappa[1:n-1], n-2, 0)
        B[1] = kappa[:n-1] + kappa[1:]
        B[2] = np.insert(-kappa[1:n-1], 0, 0)
        A = n**2 * spdiags(B, [-1, 0, 1], n-1, n-1, format='csr')
        # RHS
        b = f(x[1:n])
        b[0] += kappa[0]*u0*n**2
        b[n-2] += kappa[n-2]*un*n**2
        # Solve the system
        u = spsolve(A, b)
        # Append the boundary conditions
        u = np.insert(u, [0, n-1], [u0, un])
        # Interpolate to the observation points
        fwd[i] = np.interp(x_obs, x, u)
    return fwd

