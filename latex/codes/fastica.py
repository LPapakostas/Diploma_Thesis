import numpy as np
from numpy import linalg as la
import preproc as prc
from deflational_method import ica_def
from symmetric_method  import ica_sym

def fastica(X,n_comp=None,algorithm = 'deflation',whiten=True,fc = 'exp',alpha = 1.0,
            max_it = 10000,tol = 10**-6,w_init = None):
    '''
    PARAMETERS:
    X ( array p,n) : contains the observations measured on p variables
    OPTIONAL:
    n_comp (int) : number of components to extract
    algorithm(string 'symmetric' or 'deflational') : FastICA solving method
    whiten (bool) : do data whitening .If False , we assume that the data is already whiten 
    fc ( string 'logcosh'/'exp'/'cube') : form of G function for comuting the negentropy
    alpha (float) : only use for fc = 'logcosh' . Default is 1.0
    max_it(int) : maximum iteration number
    tol(float) : tolerance which the um-mixing matrix is considered that has converged
    w_init ( array n_comp,n_comp) : initial values for un-mixing array

    OUTPOUTS:
    K(array n_comp,p): pre-whitening matrix
    W(array n_comp,n_comp): estimated un-mixing matrix
    S(array p,n): estimated source signal matrix

    '''
    if (alpha < 1.0) or (alpha > 2.0) :
        raise ValueError("alpha must be in [1,2]")
    
    if type(fc) is str:
        if fc == 'logcosh':
            def g(x,alpha):
                return np.tanh(x)
            def gdot(x,alpha):
                return alpha*(1 - (np.tanh(x))**2)
        elif fc == 'exp':
            def g(x,alpha):
                return x*np.exp(-0.5*(x**2))
            def gdot(x,alpha):
                return np.exp(-0.5*(x**2))*(1-x**2)
        elif fc == 'cube':
            def g(x,alpha):
                return x**3
            def gdot(x,alpha):
                return 3*x**2
        else:
            raise ValueError("Function does not included.")
        
    sensors,samples = X.shape

    if n_comp is None :
        n_comp = int(min(sensors,samples))
    elif (n_comp > min(sensors,samples)):
        n_comp = min(sensors,samples)
        print("Components to large.It was set",n_comp)

    if whiten :
        # Centering the columns / variables
        X = prc.mean_remove(X,sensors,samples)
        # Do the PCA algorithm and then the data whitening
        eig,e = prc.pca_eig(X,n_comp)
        Z,K = prc.data_whitening(X,eig,e)
    else:
        Z = X.copy()
        K = None
    
    if w_init is None:
        # Choose random initial values 
        # ||w|| must be 1
        w_init = np.random.normal(size = (n_comp,n_comp))
    else:
        w_init = np.asarray(w_init)
        if w_init.shape != (n_comp,n_comp):
            raise ValueError("Wrong Shape. The correct dimensions are",(n_comp,n_comp))
  
    if algorithm == 'symmetric':
        W = ica_sym(Z,n_comp,max_it,tol,g,gdot,w_init,alpha)
    elif algorithm == 'deflation':
        W = ica_def(Z,n_comp,max_it,tol,g,gdot,w_init,alpha)
    else:
        raise ValueError("Wrong Argument")
    S_e = W@(K@X)
    return K,W,S_e,Z