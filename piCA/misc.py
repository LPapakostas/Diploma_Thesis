from scipy import linalg as la

def is_def(X,tol=-5*(10**-3)):
    '''
    PARAMETERS:
    X (array p,n): array that will be checked if is positive semi-definitive
    tol(float): tolerance of eigvalue that will be considered non-negative
    
    OUTPUTS:
    True/False(boolean):True if X array is positive definitive. 
                        Else , False
    
    An array is positive semi-definitive if all its eightvalues are non-negative.
    We implement Sylvester's criterion in order to deduce if a matrix is definitive
     '''
    count = 0 
    # For 3 dimension arrays 
    if ( len(X.shape) == 3):   
        Max = X.shape[0] ; n_of_pos = X.shape[1]
        for i in range(Max):
            p = 0;
            for j in range(n_of_pos):
                size = X[i,j,j].shape
                if size == ():
                    size = (1,1)
                # T is the ixi , i = 1,...,n matrix that we will compute its
                # minor determinant
                T = (X[i,j,j]).reshape((size[0],size[0]))
                if( la.det(T) > tol):
                    p+=1
            if (p == n_of_pos):
                count+=1
        # if all minor determinants are positive, then X matrix is positive definite
        if (count >= Max-1):
            return True 
        return False 
    else :
        n_of_pos = X.shape[0]
        for j in range(n_of_pos):
            size = X[j,j].shape
            if size == ():
                size = (1,1)
                # T is the ixi , i = 1,...,n matrix that we will compute its
                # minor determinant
            T = (X[j,j]).reshape((size[0],size[0]))
            if(la.det(T) > tol):
                count+=1
        # if all minor determinants are positive, then X matrix is positive definite
        if (count == n_of_pos):
                return True
        return False