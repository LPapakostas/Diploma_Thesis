import numpy as np
import matplotlib.pyplot as plt
import time 
from piCA import piCA
from scipy import signal

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

start = time.time(); noise_flag = False
Fs = 1000 ; duration = 12 ; n_samples = int(duration*Fs)
t = [(1./Fs)*t for t in range(0,n_samples)]
t = np.asarray(t)

S = (np.loadtxt('C:\\Users\\Labis Papakwstas\\Desktop\\Diplwmatikh\\Data\\r01.txt'))[:,1:6]
S = S.T ; X = S[:,:n_samples]
#%%
S_ff = np.fft.fft(S[0,:])
S_fft =S_ff[:int(n_samples/2)+1]
S_fft = np.real( S_fft*S_fft.conj())
F = [ (Fs/n_samples)*x for x in range(0,n_samples)] ; F = F[:int(n_samples/2)+1]
misc.plot_Signals(S_fft,F,"ff")
#%%
#top=0.946,bottom=0.047,left=0.071,right=0.981,hspace=0.787,wspace=0.12
top=0.941 ; bottom=0.122 ;left=0.071 ;right=0.981 ; hspace=0.512 ;wspace=0.15
fig = plt.figure()
plt.subplot(511)
plt.plot(t,X[0,:])
plt.title("Direct ECG - Reference ")
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.subplot(512)
plt.plot(t,X[1,:])
plt.title("Abominal ECGs")
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.subplot(513)
plt.plot(t,X[2,:])
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.subplot(514)
plt.plot(t,X[3,:])
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.subplot(515)
plt.plot(t,X[4,:])
plt.xlabel("Time(s)")
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.waitforbuttonpress() ; plt.close()

'''
# plot fft to make butterworth filter
T = 1./Fs

fft1 = np.fft.fft(S[3,:])
Pfft1 = np.real(fft1 * fft1.conj())
Pfft1 = Pfft1[:int(n_samples/2)+1]
f = np.linspace(0.0, 1.0/(2.0*T), int(n_samples/2)+1)
plt.plot(f,Pfft1)

b,a = butter_lowpass(30,Fs)
X = signal.filtfilt(b,a,S)

# plotting results
misc.plot_Signals(X,t,"Filt Signals")
'''
#%%
X_pica = X[1:6,:]
minlag,maxlag = 0.0,1.6
E_pica,K_pica,W_pica,P_pica= piCA(X_pica,Fs,minlag,maxlag,preproc=True,norm=True,amuse=False)
E_a,K_a,W_a,P_a = piCA(X_pica,Fs,minlag,maxlag,preproc=True,norm=True,amuse=True)
#0.7 10 10 piCA      0.28 10 10 AMUSE

#%%
# Help find RR intervals
direct_ECG = X[0,:]
peaks,_ = signal.find_peaks(direct_ECG,distance = 250)
plt.plot(t,direct_ECG)
plt.plot(peaks/Fs,direct_ECG[peaks],"X")
plt.title("RR-Peaks Annotations")
#%%
S_pica = np.empty((W_pica.shape[0],X_pica.shape[0],X_pica.shape[1]))
for i in range(W_pica.shape[0]):
    if K_pica is None:
        S_pica[i,:] = W_pica[i,:,:]@X_pica
    else:
        S_pica[i,:,:] = (W_pica[i,:,:]@K_pica)@X_pica
    #misc.plot_Signals(S_pica,t,"Period = "+str(P_pica[i]))
#%%
S_a = np.empty((W_a.shape[0],X_pica.shape[0],X_pica.shape[1]))
for i in range(W_a.shape[0]):
    if K_a is None:
        S_a[i,:] = W_a[i,:,:]@X_pica
    else:
        S_a[i,:,:] = (W_a[i,:,:]@K_a)@X_pica
#%%
#top=0.881,bottom=0.122,left=0.071,right=0.971,hspace=0.54,wspace=0.15
# Fetal
S_pica0 = S_pica[0,:,:]
fig = plt.figure()
fig.suptitle("Periodic Component Analysis")
plt.subplot(411)
plt.plot(t,S_pica0[0,:])
plt.plot(peaks/Fs,S_pica0[0,:][peaks],"X")
plt.title("T1= "+str(round(P_pica[0],3))+" s")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.subplot(412)
plt.plot(t,S_pica0[1,:])
plt.plot(peaks/Fs,S_pica0[1,:][peaks],"X")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.subplot(413)
plt.plot(t,S_pica0[2,:])
plt.plot(peaks/Fs,S_pica0[2,:][peaks],"X")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.subplot(414)
plt.plot(t,S_pica0[3,:])
plt.plot(peaks/Fs,S_pica0[3,:][peaks],"X")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.xlabel("Time(s)")
plt.waitforbuttonpress() ; plt.close()
#%%
S_a0 = S_a[0,:,:]
fig = plt.figure()
fig.suptitle("AMUSE")
plt.subplot(411)
plt.plot(t,S_a0[0,:])
plt.plot(peaks/Fs,S_a0[0,:][peaks],"X")
plt.title("T1= "+str(round(P_a[0],3))+" s")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.subplot(412)
plt.plot(t,S_a0[1,:])
plt.plot(peaks/Fs,S_a0[1,:][peaks],"X")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.subplot(413)
plt.plot(t,S_a0[2,:])
plt.plot(peaks/Fs,S_a0[2,:][peaks],"X")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.subplot(414)
plt.plot(t,S_a0[3,:])
plt.plot(peaks/Fs,S_a0[3,:][peaks],"X")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.xlabel("Time(s)")
plt.waitforbuttonpress() ; plt.close()
#%%
S_pica1 = S_pica[1,:,:]
maternal_ECG = S_pica1[0,:]
peaks_m,_ = signal.find_peaks(-maternal_ECG,distance = 500)
plt.plot(t,maternal_ECG)
plt.plot(peaks_m/Fs,maternal_ECG[peaks_m],'P')
#%%
S_a1 = S_a[1,:,:]
maternal_ECG_AMUSE = S_a1[0,:]
peaks_ma,_ = signal.find_peaks(maternal_ECG_AMUSE,distance = 500)
plt.plot(t,maternal_ECG_AMUSE)
plt.plot(peaks_ma/Fs,maternal_ECG_AMUSE[peaks_ma],'P')
#%%
# top=0.891, bottom=0.122, left=0.071, right=0.971, hspace=0.535,wspace=0.15
S_pica1 = S_pica[1,:,:]
fig = plt.figure()
fig.suptitle("Periodic Component Analysis")
plt.subplot(411)
plt.plot(t,S_pica1[0,:])
plt.plot(peaks_m/Fs,S_pica1[0,:][peaks_m],"P")
plt.title("T2= "+str(round(P_pica[1],3))+" s")
plt.subplots_adjust(top=0.891, bottom=0.122, left=0.071, right=0.971, hspace=0.535,wspace=0.15)
plt.subplot(412)
plt.plot(t,S_pica1[1,:])
plt.plot(peaks_m/Fs,S_pica1[1,:][peaks_m],"P")
plt.subplots_adjust(top=0.891, bottom=0.122, left=0.071, right=0.971, hspace=0.535,wspace=0.15)
plt.subplot(413)
plt.plot(t,S_pica1[2,:])
plt.plot(peaks_m/Fs,S_pica1[2,:][peaks_m],"P")
plt.subplots_adjust(top=0.891, bottom=0.122, left=0.071, right=0.971, hspace=0.535,wspace=0.15)
plt.subplot(414)
plt.plot(t,S_pica1[3,:])
plt.plot(peaks_m/Fs,S_pica1[3,:][peaks_m],"P")
plt.subplots_adjust(top=0.891, bottom=0.122, left=0.071, right=0.971, hspace=0.535,wspace=0.15)
plt.xlabel("Time(s)")
plt.waitforbuttonpress() ; plt.close()
#%%
S_a1 = S_a[1,:,:]
fig = plt.figure()
fig.suptitle("AMUSE")
plt.subplot(411)
plt.plot(t,S_a1[0,:])
plt.plot(peaks_ma/Fs,S_a1[0,:][peaks_ma],"P")
plt.title("T2= "+str(round(P_a[1],3))+" s")
plt.subplots_adjust(top=0.891, bottom=0.122, left=0.071, right=0.971, hspace=0.535,wspace=0.15)
plt.subplot(412)
plt.plot(t,S_a1[1,:])
plt.plot(peaks_ma/Fs,S_a1[1,:][peaks_ma],"P")
plt.subplots_adjust(top=0.891, bottom=0.122, left=0.071, right=0.971, hspace=0.535,wspace=0.15)
plt.subplot(413)
plt.plot(t,S_a1[2,:])
plt.plot(peaks_ma/Fs,S_a1[2,:][peaks_ma],"P")
plt.subplots_adjust(top=0.891, bottom=0.122, left=0.071, right=0.971, hspace=0.535,wspace=0.15)
plt.subplot(414)
plt.plot(t,S_a1[3,:])
plt.plot(peaks_ma/Fs,S_a1[3,:][peaks_ma],"P")
plt.subplots_adjust(top=0.891, bottom=0.122, left=0.071, right=0.971, hspace=0.535,wspace=0.15)
plt.xlabel("Time(s)")
plt.waitforbuttonpress() ; plt.close()
#%%
min_t = int(minlag*Fs) ; max_t = int(maxlag*Fs)
t_e = t[min_t:max_t+1]
plt.plot(t_e,E_pica); plt.title("Minimum Periodicity Error for piCA")
plt.plot(t_e,np.zeros_like(t_e),'-.')
plt.xlabel("Time(s)")
#%%
min_t = int(minlag*Fs) ; max_t = int(maxlag*Fs)
t_e = t[min_t:max_t+1]
plt.plot(t_e,E_a); plt.title("Minimum Periodicity Error for AMUSE")
plt.plot(t_e,np.ones_like(t_e),'-.')
plt.xlabel("Time(s)")
