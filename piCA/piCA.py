import numpy as np
from misc import is_def
import preproc as pre
import C0_Ct ,C_e
import rayleigh_minimum as rm
import min_max 

def piCA(X,f,minlag,maxlag,preproc = True,norm = True,amuse = False,circ = False) :
    '''
    PARAMETERS:
    X (array nxm): mixed signals 
    f(float) : sampling frequency
    minlag(float): minimum limit of time 
    maxlag(float): maximum limit of time
    preproc(bool): must be true to preprocess the data(mean remove - data whitening)
    norm(bool): if True, we do the normalization
    amuse(bool): if True , we use AMUSE method
    OUTPUTS:
    Er(array k,1): Minimun periodicity error
    K(array n,n): Whitening matrix
    W(array k,n): Eigvectors of local minimum eigvalues
    periods(list): Position(k) of local minima
    Z(array n,m): Whiten data 
    '''
    if (len(X.shape) != 1):
        if X.shape[0] > X.shape[1] :
            print("Signals must be represented be as rows of X array")
            X = X.T   
        # nxm dimension of X array
        sensors,samples = X.shape
    else:
        sensors,samples = 1,len(X)

    # Check errors about time lag
    if (minlag < 0) or (maxlag < 0):
        raise ValueError("Lag values must be non-negative")
    elif (minlag > samples) or (maxlag > samples):
        raise ValueError("Lag values must be less than the observations")
    elif (minlag > maxlag):
        raise ValueError("Minimum Lag value must be less than Maximum Lag value")
        
    # Check error about Sampling Frequency
    if (f <= 0):
        raise ValueError("Sampling Frequencyby be positive")
    
    if ( not is_def(np.cov(X)) and preproc == True ):
        print("Covariance matrix of X array must be positive in order to do data whitening")
        print("preproc value is set to False")
        preproc = False
    
    # Check conditions for AMUSE algorithm 
    if (amuse == True and preproc == False):
        raise ValueError("AMUSE need whiten data")

    if (circ == True and preproc == False ):
        raise ValueError("Circular piCA needs whiten data")

    if (amuse == True and circ == True):
        raise ValueError("Choose one of methods and then proceed")

    if preproc:
        X_0 = pre.mean_remove(X,sensors,samples)
        # we will compute the same number of PCA components as the sensors
        eig,v = pre.pca_eig(X_0,sensors)
        Z,K = pre.data_whitening(X_0,eig,v)
        # Covariance matrix of Z is I
    else:
        Z = X.copy()
        K = None
    
    # Choose time values for algorithm evaluation
    k_max = int(maxlag*f); k_min = int(minlag*f)
    print(k_max,k_min)
    length_k = k_max - k_min + 1
    if (length_k == 1):
        k = [k_min] # k is list that has only one term
    else:
        # k is a list that has values in [minlag,maxlag]
        k = [x for x in range(k_min,k_max+1)]
    print("k length",len(k))

    # Compute the C matrix. C has (sensors)x(sensors) dimension
    C = Z@Z.T/(samples-1)  # C matrix is semi-definitive
    # We make the C matrix as symmetric as possible from (1)
    print(C)
    C = 0.5*(C+C.T) 
    print("C dimension",C.shape)
    
    if not (amuse or circ):
        C0,Ct = C0_Ct.compute(Z,k,norm)
        
        if (not is_def(C0)) :
            raise ValueError("C0 matrix is not semi-definitive")
        if (not is_def(Ct)) :
            raise ValueError("Ct matrix is not semi-definitive")
        
    # Compute Ce matrix = (Ct0 + Ct0.T)/2 
    Ce = C_e.compute(Z,k,amuse,circ,norm)       
        
    # Compute A matrix
    A = np.zeros((len(k),sensors,sensors),dtype = np.float64)

    for j in range(len(k)):
        if (amuse or circ):
            A[j,:,:] = Ce[j,:,:]    
        else:
            A[j,:,:] = C0[j,:,:]+Ct[j,:,:]-2*Ce[j,:,:]
            
    # Make A matrix as symmetric as possible from (1)
    if not (amuse or circ):
        for i in range(A.shape[0]):
            A[i,:,:] = (0.5*(A[i,:,:]+A[i,:,:].T))
    
    if (not is_def(A) and not (amuse or circ)) :
        raise ValueError("A matrix is not semi-definitive")
    
    # Find the eigvalues with their matching eigvector that minimize
    # the Rayleigh fraction (wT*A[k]*w)/(wT*C*w) for each k.
    Er,Wr = rm.find(A,C,preproc,amuse,circ)

    # Find the local minima/maxima through minimum eigvlaues and plot them
    if (length_k != 1):
        k_ideal_full = min_max.find(Er,k,amuse,circ)
        
        if (k_ideal_full == []):
            raise ValueError("Not found any periodic components")
        
        # With below algorithm, we will only keep the basic periods by removing
        # the multiples of them
        k_ideal_full = [x+int(minlag*f) for x in k_ideal_full ]
        k_ideal = []
        for i in range(sensors):
            k_ideal.append(k_ideal_full[0])
            k_ideal_full = [ x if x%k_ideal_full[0] != 0 else 0.0 for x in k_ideal_full]
            k_ideal_full = [ x for x in k_ideal_full if x!= 0.0]
            if (k_ideal_full == []):
                break
        k_ideal = [ x-int(minlag*f) for x in k_ideal]
        # use only for plotting
        periods = [round((x)/f,3) for x in k_ideal] 
        # Prellocate un-mixing matrix
        W = np.zeros((len(k_ideal),sensors,sensors) , dtype = np.float64)
        # Save the matching eigvectors of minimum/maximum positions
        for kx in range(len(k_ideal)):
            W[kx,:,:] = Wr[k_ideal[kx],:,:] 

    else:
        W = Wr.copy() ; k_ideal = None
        periods = [round(k[0]/f,3)]
    
    return Er,K,W,periods