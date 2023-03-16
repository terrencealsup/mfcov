import numpy as np
from time import time
from scipy.stats import multivariate_normal as mvnrnd
import pandas as pd

from spd import Log, Exp
from utils import load_params

import sys


# Select the example problem
example = sys.argv[1]

# Import the forward model code
if example == 'heat':
    from heat.forwardModel import forwardModel
elif example == 'sqg':
    from sqg.forwardModel import forwardModel
else:
    print('Example must be one of {heat, sqg}, using heat by default')


# Load parameters for the problem
config = load_params(example + '/params.csv')

d_in = config['dim_in']       # Input dimension
d_out = config['dim_out']     # Output dimension (observable)

high_fid = config['ngrid_high'] # Grid points for high-fidelity model
low_fid = config['ngrid_low']   # Grid points for low-fidelity model



###############    Timings    ########################

# Time high-fidelity model
ntrials = 100
x = mvnrnd.rvs(size=ntrials, mean=np.zeros(d_in), cov=np.eye(d_in))
start = time()
y = forwardModel(x, high_fid, d_out)
elapsed = time() - start
hf_time = elapsed/ntrials

# Time low-fidelity model
ntrials = 1000
x = mvnrnd.rvs(size=ntrials, mean=np.zeros(d_in), cov=np.eye(d_in))
start = time()
y = forwardModel(x, low_fid, d_out)
elapsed = time() - start
lf_time = elapsed/ntrials

# Time matrix Exp
ntrials = 1000
A = np.zeros((ntrials, d_out, d_out))
# Create random matrices
for i in range(ntrials):
    A[i] = np.random.randn(d_out**2).reshape((d_out, d_out))

# Time Exp
start = time()
for i in range(ntrials):
    A[i] = Exp(A[i], np.eye(d_out))
    #A[i] = expm(A[i])  # scipy expm
elapsed = time() - start
exp_time = elapsed/ntrials

# Time Log
start = time()
for i in range(ntrials):
    A[i] = Log(A[i], np.eye(d_out))
    #A[i], _ = logm(A[i], disp=False)  # scipy logm
elapsed = time() - start
log_time = elapsed/ntrials




# Print and save data
print('\nTimings\n')
print('High-fidelity:\t{:0.3e}(s)'.format(hf_time))
print('Low-fidelity:\t{:0.3e}(s)\n'.format(lf_time))

print('Matrix Exp:\t{:0.3e}(s)'.format(exp_time))
print('Matrix Log:\t{:0.3e}(s)\n'.format(log_time))


df = pd.DataFrame(
    index=['High-fidelity', 'Low-fidelity', 'Matrix Exp', 'Matrix Log'],
    data = {'time(s)':[hf_time, lf_time, exp_time, log_time]}
)

df.to_csv(example + '/results/timings.csv')
print('Results saved to: ' + example + '/results/timings.csv')