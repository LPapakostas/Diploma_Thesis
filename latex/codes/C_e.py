import numpy as np
from numpy import fft

def compute(X,k,amuse,circ,norm):
    '''
    PARAMETERS:
    X(array p,n): Array that contains the mixed signals
    k(list): values from [kmin,k_max] (minlag,maxlag)
    amuse(boolean): If True , we do AMUSE πCA
    circ(boolean): If True , we do circular πCA
    norm(boolean): If True , we normalize the output by multipling with 1/(samples-k)
    
    OUTPUTS:
    C(array k,p,n): Ce array 
	
    Ce[k] = 0.5*(Ct0[k]+Ct0[k].T)
    We will use Fast Fourier Tranform to compute Ce array
    As we know , the elements of Ct0[k] are the cross-corelations between signals of
    X array and we observed that it is the circular convolution of Xi with the 
    mirrored X for k = 0 , if we add zeros on both of signals so that we have
    2*(samples)-1 samples  
    '''
	
    sensors,samples = X.shape ; L = len(k)
   
    def nextpow2(i):
        n = 1
        while (n < i): 
            n *= 2
        return n
    
    if circ:
        fft_samples = samples
    else:
         fft_samples = 2^nextpow2(2*samples-1)
   
    # Calculate the FFT of the rows of X array
    Xf = fft.fft(X,fft_samples)
    Xfc = Xf.conj()
    
    # Calculate the DFT of the cross-correlation
    Cs = np.empty( (int(0.5*sensors*(sensors+1)),fft_samples),dtype = np.float64); l = 0 
    for i in range(sensors):
        for j in range(i,sensors):
            # Compute F(cij) = Re[F{Xf(i)}*F{Xfc(j)}]
            Cs[i*(sensors-1)+j-l,:] = np.real(Xf[i,:]*Xfc[j,:])
        l+=i

    # Compute the inverse FFT of cross correlation and keep only those that we need
    Cs = fft.ifft(Cs) ; Cs = Cs[:,:samples]
    Cs = Cs[:,k[0]:k[-1]+1]

    Ce = np.zeros((L,sensors,sensors) , dtype=np.float64) ; m = 0
    for i in range(sensors):
        for j in range(i,sensors):
            # Compute F(cij) = Re[F{Xf(i)}*F{Xfc(j)}]
            Ce[:,j,i] = np.real(Cs[i*(sensors-1)+j-m,:])
            if not(i==j):
                Ce[:,i,j] = Ce [:,j,i]
        m+=i
        
    if norm:
        for i in range(L):
            Ce[i,:,:] = Ce[i,:,:] /(samples-1-k[i])
  
    return Ce