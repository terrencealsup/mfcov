from pyqg import sqg_model
import numpy as np
from scipy.interpolate import interp2d
from scipy.stats import multivariate_normal as mvnrnd


def forwardModel(z, npts, n_obs_coord):
    """
    z is the parameter vector (d, )

    z[0] is aspect ratio
    z[1] is beta
    z[2] is bouyancy frequency, Nb
    z[3] is background zonal flow, U
    z[4] is the width
    """

    c    = z[0]    # log aspect ratio, default=0, std=0.3
    beta = z[1]    # gradient of coriolis parameter, default=0, std=0.003
    Nb   = z[2]    # log bouyancy frequency, default=0
    U    = z[3]    # background zonal flow, default=0, std=0.2
    wdt  = z[4]    # width parameter, default=6, std=1


    # create the model object
    year = 1.
    m = sqg_model.SQGModel(L=2.*np.pi, nx=npts, tmax=24.0, beta=beta, Nb=np.exp(Nb), U=U, H=1.,
                           rek=0., dt=0.005, twrite=1e10, log_level=0)

    x = np.linspace(m.dx/2, 2*np.pi, m.nx) - np.pi
    y = np.linspace(m.dy/2, 2*np.pi, m.ny) - np.pi
    xx, yy = np.meshgrid(x, y)

    # Choose ICs from Held et al. (1995)
    # case i) Elliptical vortex
    qi = -np.exp(-(xx**2 + (np.exp(c)*yy)**2)/(m.L/np.abs(wdt))**2)


    # initialize the model with that initial condition
    m.set_q(qi[np.newaxis,:,:])

    # Run the model and get the solution at the grid points
    m.run()
    u = m.q.squeeze() + m.beta * m.y

    # Get the interpolant on the grid
    obs_map = interp2d(x, y, u)

    # Get the interpolated values
    new_x = np.linspace(m.dx/2, 2*np.pi, n_obs_coord+2)[1:-1] - np.pi
    new_y = np.linspace(m.dy/2, 2*np.pi, n_obs_coord+2)[1:-1] - np.pi

    obs = obs_map(new_x, new_y)
    
    new_xx, new_yy = np.meshgrid(new_x, new_y)
    obs_pts = np.vstack([new_xx.flatten(), new_yy.flatten()]).T
    
    # Uncomment to return the grid, solution, observation, and observation points
    # return (xx, yy, u, obs.flatten(), obs_pts)
    return obs.flatten() 






