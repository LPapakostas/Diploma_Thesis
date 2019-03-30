import numpy as np
from numpy import linalg as la

def find(A,C,preproc,amuse,circ):
    '''
    PARAMETERS:
    A(array k,n,n): array that was computed on piCA
    C(array n,n): array that was computed on piCA
    preproc(boolean): True if data was preprocessed(mean removed - PCA - data whitening)
    amuse(boolean): True if AMUSE method is used
    OUTPUTS:
    E(array k,1): eigvalues that minimize Rayleigh fraction
    W(array k,n): eigvectors of above eigvalues
    '''
    # We will find the smallest eigvalue with the matching 
    # eigvector for each k
    sensors,sensors = C.shape ; L = A.shape[0]
    
    # Prellocate the W and E matrix that will have the 
    # smallest eigvalue and the matching eigvector for each k
    W = np.zeros((L,sensors,sensors),dtype = np.float64) # rows are eigvectors transpose
    E = np.zeros((L),dtype = np.float64)
    
    for i in range(L):
        # if data is preprocessed , the General Eigvalue Problem
        # becomes a simple Eigvalue Problem from (8)
        if (preproc):
            e,w = la.eig(A[i,:,:])#,right = False,left = True)
        else:
            e,w = la.eig(A[i,:,:],C,right = True)
        # Find the ideal position of eigvalues in descending order
        # and the smallest value must be in 0-place
        e_index = np.argsort(e)
        if (amuse or circ):
            e_index= e_index[::-1]
        # Sort eigvalues with the matching eigvectors
        e = e[e_index] ; w = w[:,e_index]
        # Keep the smallest eigvalue and matching eigvector
        # in row form
        E[i] = np.real(e[0]) ; W[i,:,:] = np.real(w.T)
    # return the smallest eightvalues and eigvectors for all k
    return E,W