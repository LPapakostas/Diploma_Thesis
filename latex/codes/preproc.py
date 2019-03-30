import numpy as np
from scipy import linalg as la

def mean_remove(X,rows,cols): 
    '''
    PARAMETERS:
    X(array rows,cols):array that needs its mean value removed from all observations
    OUTPUTS:
    X_tilde (array rows,cols):array with zero mean 
    '''
    p,n = rows,cols
    # We compute the mean value of the columns of X array and 
    # reshape in a px1 array (p = number of different signals)
    mx = X.mean(axis=-1).reshape((p,1))
    # remove the mean matrix(px1) from the signals 
    X_tilde = X - mx.dot(np.ones((1,n)))
    
    # Check if mean of X_tilde is close to zero
    mxt = X_tilde.mean(axis=-1).reshape((p,1))
    if np.all(np.abs(mxt) > 10**(-7)):
        raise ValueError("Mean remove has failed")
    
    return X_tilde

def pca_eig(X,comp):
    p,n = X.shape
    if(comp > min(X.shape)):
        print("Principal components must be less than signals")
        comp = min(X.shape)
    # Compute the covariance matrix of X 
    Cx = np.cov(X) 
    # e is the eigvector matrix where i-th column
    # responds to i-th eigvalue
    e,V = la.eig(Cx) 
    # returns index sort in ascending order and reverse it
    eig_sort = np.argsort(e)[::-1]
    # re-arrange the vectors in the correct order
    e,V = e[eig_sort] , V[:,eig_sort]
    # keep the N first eigvalues and eigvectors as given by components
    e,V = e[:comp] , V[:,:comp]
    return e,V

def data_whitening(X,e,V):
    '''
    PARAMETERS:
    X(array p,n): array that will be whiten
    e(array p,1): eigvalues of X array that are sorted in descending order
    V(array p,p): matrix that consists the right eigvectors that corresponds 
                  on each eigvalue
    OUTPOUTS:
    Z(array p,n): array that has whitened
    A(array p,p): whiten trasformation matrix
    
    NOTES:
    1) The white trasformation is reversible
    2) Must perform after PCA algorithm
    '''
    # Compute D^(-1/2) matrix (D= diag(d1,..,dp))
    D = np.diag(1./np.sqrt(e))
    A = (np.real(V@D)).T
    # A = D^(-1/2)*V.T
    #print((V@D).T - D@V.T)
    #A = np.real(D@V.T)
    Z = np.real(np.dot(A,X))
    return Z,A