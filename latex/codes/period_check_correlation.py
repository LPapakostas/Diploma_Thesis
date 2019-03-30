import numpy as np
import matplotlib.pyplot as plt
from scipy import signal 

def check(S,P,Fs,t):
    '''
    PARAMETERS:
    S(array nxm): signals that produced from πCA algorithm
    P(list): periods of signals that produced from πCA algorithm
    Fs(float): sampling frequency of continious signals
    
    in this function , we find the peaks of autocorrelation for each signal produced on 
    πCA algorithm. Based on theory , the time difference of peaks give us the period of each signal.
    If the difference is the same as the period of the signal , we deduce that our method is correct
    '''
    P_correct = []
    for i in range(S.shape[0]):
        # Remove the mean value of each signal
        s_m = S[i,:] - np.mean(S[i,:])*np.ones((S.shape[1])) 
        # Compute the autocorrelation of signal 
        s = np.correlate(s_m,s_m,"full")
        s = s[int(s.shape[0]/2):]
        # Normalize the autocorrelation
        s/=max(s)
        # find the maximum peaks
        peaks,_ = signal.find_peaks(s,height = 0.1,distance = 20,width=1)
        # The time between autocorrelation spikes must be equal to the period 
        # of the signal
        if (len(peaks) != 0):
            print("Period of signal " + str(i+1)+ " is " + str(round((peaks[0])/Fs,3))+" seconds")
            P_correct.append(peaks[0]/Fs)
    
    return P_correct