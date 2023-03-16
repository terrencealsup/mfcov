import numpy as np
from scipy.linalg import cholesky


def learnMetricSamples(x, y, t=0.1, method='fast'):

    if method != 'fast':
        S = np.zeros((x.shape[1], x.shape[1]))
        D = np.zeros((x.shape[1], x.shape[1]))
        for i in range(len(x)):
            for j in range(i+1, len(x)):
                S += np.outer(x[i]-x[j], x[i]-x[j])
        for i in range(len(y)):
            for j in range(i+1, len(y)):
                S += np.outer(y[i]-y[j], y[i]-y[j])
        for i in range(len(x)):
            for j in range(len(y)):
                D += np.outer(x[i]-y[j], x[i]-y[j])
    else:
        covx = np.cov(x.T)
        covy = np.cov(y.T)
        mx = np.mean(x, axis=0)
        my = np.mean(y, axis=0)
        S = covx + covy 
        D = S + np.outer(mx-my, mx-my)
    
    A = np.linalg.inv(S)
    B = D
    
    RA = cholesky(A)
    RB = cholesky(B)
    
    Z = np.linalg.solve(RA.T, RB.T).T
    
    eigvals, eigvecs = np.linalg.eig(Z.T @ Z);
    T = np.diag(eigvals**(t/2)) @ eigvecs.T @ RA;
    G = T.T @ T;
    
    return G
    


def learnMetric(covA, covB, mA, mB, t=0.1):
    """
    Compute the SPD point along the geodesic between A and B using the Cholesky-Schur method.
    """

    # Similarity inverse
    S = covA + covB

    # Dissimilarity
    D = (covA + covB + np.outer(mA-mB, mA-mB))

    A = np.linalg.inv(S)
    B = D
    
    RA = cholesky(A)
    RB = cholesky(B)
    
    Z = np.linalg.solve(RA.T, RB.T).T
    
    eigvals, eigvecs = np.linalg.eig(Z.T @ Z)
    T = np.diag(np.abs(eigvals)**(t/2)) @ eigvecs.T @ RA
    G = T.T @ T
    
    return G


