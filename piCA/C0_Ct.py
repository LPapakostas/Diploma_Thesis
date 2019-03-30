import numpy as np

def compute(X,k,norm):
    '''
    PARAMETERS:
    X(array p,n): Array that contains the mixed signals
    k(list): values from [k_min,k_max] (minlag,maxlag)
    norm(boolean): If True , we normalize the output by multipling with 1/(samples-k) 
    flag(character): '0' to compute C0 array / 't' to compute Ct
    OUTPUTS:
    C(array k,p,n): C0/Ct array 
    '''
    sensors,samples = X.shape
    
    # We compute for each k the matrix below. Note that we must match the C[k]
    # with k
    L = len(k) 
    
    # Prelocate the C0 or Ct matrix.
    C0 = np.zeros((L,sensors,sensors),dtype = np.float64)
    Ct = np.zeros((L,sensors,sensors),dtype = np.float64)
    
    # We will compute C0 and Ct matrix using (2) 
    # C0[k-1] = C0[k] + X[m-k+1]*(X[m-k+1])T
    # Ct[k-1] = Ct[k] + X[k]*(X[k])T
    # for the non-normalized matrix C0 and Ct
    # In other words , we start from the max(k) value and we compute
    # until the min(k) value
    #'''
    # First we compute the C0 and Ct matrix for k = maxlag
    x0 = X[:,:(samples-1)-k[-1]]  ; C0[-1,:,:] = x0@x0.T
    xt = X[:,k[-1]+1:(samples-1)] ; Ct[-1,:,:] = xt@xt.T 
     
    # We do the recursive algorithm about C0/Ct for k = maxlag-1,...,minlag
    for i in range(L-1,0,-1):
        x0 = X[:,(samples-1)-k[i]+1] ; x0 = x0.reshape((sensors,1)) 
        C0[i-1,:,:] = C0[i,:,:] + x0@x0.T  
        xt =  X[:,k[i]] ; xt = xt.reshape((sensors,1)) 
        Ct[i-1,:,:] = Ct[i,:,:] + xt@xt.T
   
    if norm:
        for i in range(L):
            C0[i,:,:] = C0[i,:,:] /((samples-1)-k[i])
            Ct[i,:,:] = Ct[i,:,:] /((samples-1)-k[i])
    
    # Make the matrix as symmetric as possible from (1)
    for i in range(L):
        C0[i,:,:] = 0.5*(C0[i,:,:]+C0[i,:,:].T)
        Ct[i,:,:] = 0.5*(Ct[i,:,:]+Ct[i,:,:].T)
        
    return Ct,C0
