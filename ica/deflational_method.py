import numpy as np
from numpy import linalg as la

def ica_def(X,comp,max_it,tol,g,gdot,w_init,alpha): 
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
    We use deflational orthogonalization based on Gram-Schidt method 
    for un-mixing matrix

    '''
    # prelocate un-mixing matrix 
    W = np.zeros((comp,comp) , dtype = np.float64)
    p,n = X.shape
    # We choose comp indepented components to estimate
    for i in range(comp):
        # Choose an initial value and normalize it 
        w = w_init[i,:].copy()
        w /= la.norm(w) 
        
        iterations = 0 ; lim = 1.01
        # we set lim = 1.01 to be sure that the algorithm will run at least one time
        while( (iterations < max_it -1 ) and (lim > tol) ):
            # Calculate w*X product in order to calculate g(w*X) and g'(w*X)
            wtx = np.dot(w,X)[np.newaxis,:] # shape 1xn
            gwtx = g(wtx,alpha) # shape 1xn
            gdotwtx = gdot(wtx,alpha) # shape 1xn
            # the first term of w1 computes the mean value for all the rows of X*g(wX).T (vector)
            # and the second term computes the mean value of g'(w*X) matrix (number)
            w1 = (X@gwtx.T).mean(axis=1) - (np.mean(gdotwtx))*w # shape 1xp
            '''
            Gram - Schmidt method:
            After with estimate wp vectors , we do the one unit algorithm for wp+1
            and after every iteration step we substract from wp+1 the projections 
            (wp+1*wj)wj j=1, ..,p of the previously estimated vectors
            and then we normalize the wp+1
            '''
            temp = np.zeros_like(w1)
            for j in range(i):
                wt = W[j,:].copy()
                # Calculate sum{(w1*w)w}
                temp+= (w1@wt)*wt
            w1-=temp
            # Normalization
            w1/=la.norm(w1)
            # the algorithm converge when the inner product of W(k+1)
            # and W(k) is close to 1. So we define lim as the absolute sum of the 
            # inner product w1 and w and we subtract it from 1
            lim = np.abs(np.abs((w1*w).sum())-1)
            w = w1 ; iterations+=1
        W[i,:] = w
        '''
        The algorithm ends if one of below coditions is fullfilled
        i) The inner product of W(k+1) and W(K) is close to 1
        ii) If it surpasses the maximum number of iterations 
        in the second , case the algorithm might not coverged 
        '''
    return W