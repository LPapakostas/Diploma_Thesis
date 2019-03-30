import numpy as np
from math import sqrt
from numpy import linalg as la

def ica_sym(X,comp,max_it,tol,g,gdot,w_init,alpha):
    '''
    PARAMETERS:
    X(array p,n): "white" array of signals that we need to estimate
    comp(int): independent components that we need to estimate
    max_it(int): maximum iterations of algorithm running
    tol(float): tolerance error that the algorithm converge
    g(function): contrast function that we define on fast_ica function
    gdot(function): contrast function that we define on fast_ica function
    w_init(array p,p): initial weights array
    alpha(float): parameter for g and gdot
    OUTPUTS:
    W(array p,p): estimated un-mixing matrix

    NOTES:
    We use symmetrical orthogonalization for the whole un-mixing matrix 
    '''
    # We choose the initial matrix and we normalize that
    W = w_init.copy()
    W /= sqrt(sum (sum(W**2)))
    # Orthogonalize through sqrt of W W.T  matrix
    # Compute eigvalues of W@W.T
    ew,Vw = la.eig(W@W.T)
    # Make the diagonal matrix with the eigvalues
    D_sqrt = np.diag(1./np.sqrt(ew))
    # Compute the square root of W W.T
    sqrt_WWT = (Vw @ D_sqrt)@Vw.T
    # W = sqrt(WW.T) * W
    W = sqrt_WWT@W
    
    # prellocate the correct un-mixing matrix
    W1 = np.zeros_like(W,dtype = float)
    
    iterations = 0 ; lim = 1.01
    # we set lim = 1.01 to be sure that the algorithm will run at least one time
    while((lim>tol) and (iterations < max_it -1)):
        for i in range(comp):
            w = W[i,:].copy()
            w = w/la.norm(w)
            # Calculate w*X product in order to calculate g(w*X) and g'(w*X)
            wtx = np.dot(w,X)[np.newaxis,:] #shape 1xn
            gdotwtx = gdot(wtx,alpha) #shape 1xn
            gwtx = g(wtx,alpha) #shape 1xn
            # the first term of w1 computes the mean value for all the rows of X*g(wX).T (vector)
            # and the second term computes the mean value of g'(w*X) matrix (number)
            w1 = np.mean(X.dot(gwtx.T),axis=1) - (np.mean(gdotwtx))*w #shape 1xp
            W1[i,:] = w1
        '''
        After we compute all wp vectors, we do the symmetric orthogonalization 
        with the square root General Eigvalue Decompotion of WW.T matrix
        '''
        # Same procedure as the beginning
        ew1,Vw1 = la.eig(W1@W1.T)
        D1_sqrt = np.diag(1./np.sqrt(ew1))
        sqrt_W1W1T = (Vw1 @ D1_sqrt)@Vw1.T
        W1 = sqrt_W1W1T@W1
        # the algorithm converge when the inner product of W(k+1)
        # and W(k) is close to identity matrix. So we define lim as the maximum element
        # of W_new @ W_old.T - I_n 
        lim = np.max(np.abs(np.diag(np.dot(W1,W.T))-np.identity(comp)))
        W = W1 ; iterations+=1
    return W

