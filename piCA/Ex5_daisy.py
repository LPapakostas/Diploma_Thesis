import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time 
from piCA import piCA
import misc
import period_check_correlation as pcc

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

start = time.time(); noise_flag = False
Fs = 250 ; duration = 10 ; n_samples = int(duration*Fs)
t = [(1./Fs)*t for t in range(0,n_samples)]
t = np.asarray(t)
S = (np.loadtxt('C:\\Users\\Labis Papakwstas\\Desktop\\Diplwmatikh\\Data\\Daisy.dat'))[:,1:9]
S = (S[:n_samples]).T 
#b,a = butter_bandpass(3,17,Fs)
#X = signal.lfilter(b, a,S)
abdominals = S[:5,:]; thoracic = S[5:,:]
#%%
#top=0.946,bottom=0.047,left=0.071,right=0.981,hspace=0.787,wspace=0.12
top=0.941 ; bottom=0.122 ;left=0.071 ;right=0.981 ; hspace=0.512 ;wspace=0.15
fig = plt.figure()
plt.subplot(511)
plt.plot(t,abdominals[0,:])
plt.title("Abdominal ECGs")
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.subplot(512)
plt.plot(t,abdominals[1,:])
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.subplot(513)
plt.plot(t,abdominals[2,:])
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.subplot(514)
plt.plot(t,abdominals[3,:])
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.subplot(515)
plt.plot(t,abdominals[4,:])
plt.xlabel("Time(s)")
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.waitforbuttonpress() ; plt.close()
#%%
fig = plt.figure()
fig.suptitle("Thoracic ECGs")
plt.subplot(311)
plt.plot(t,thoracic[0,:])
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.subplot(312)
plt.plot(t,thoracic[1,:])
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.subplot(313)
plt.plot(t,thoracic[2,:])
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.xlabel("Time(s)")
plt.waitforbuttonpress() ; plt.close()
#%%
maternal_ECG = S[6,:]
peaks,_ = signal.find_peaks(maternal_ECG,distance = 100)
plt.plot(t,maternal_ECG)
plt.plot(peaks/Fs,maternal_ECG[peaks],"X")
plt.title("R Peaks Annotations")
'''
S_ff = np.fft.fft(S[7,:])
S_fft =S_ff[:int(n_samples/2)+1]
S_fft = np.real( S_fft*S_fft.conj())
F = [ (Fs/n_samples)*x for x in range(0,n_samples)] ; F = F[:int(n_samples/2)+1]
misc.plot_Signals(S_fft,F,"ff")
'''
#%%
minlag,maxlag = 0.0,1.2
E_pica,K_pica,W_pica,P_pica = piCA(S,Fs,minlag,maxlag,preproc=True,norm=True,amuse=False)
E_a,K_a,W_a,P_a = piCA(S,Fs,minlag,maxlag,preproc=True,norm=True,amuse=True)
# 0.55 30 1 pica     0.35 30 1 AMUSE
#%%
S_pica = np.empty((W_pica.shape[0],S.shape[0],S.shape[1]))
for i in range(W_pica.shape[0]):
    if K_pica is None:
        S_pica[i,:] = W_pica[i,:,:]@S
    else:
        S_pica[i,:,:] = (W_pica[i,:,:]@K_pica)@S
   # misc.plot_Signals(S_pica[i,:,:],t,"Period = "+str(P_pica[i]))
#%%
S_a = np.empty((W_a.shape[0],S.shape[0],S.shape[1]))
for i in range(W_a.shape[0]):
    if K_a is None:
        S_a[i,:] = W_a[i,:,:]@S
    else:
        S_a[i,:,:] = (W_a[i,:,:]@K_a)@S
    #misc.plot_Signals(S_a[i,:,:],t,"Period = "+str(P_a[i]))
#%%
fetal_ECG = S_pica[0,0,:]
peaks_f,_ = signal.find_peaks(fetal_ECG,distance = 100)
plt.plot(t,fetal_ECG)
plt.plot(peaks_f/Fs,fetal_ECG[peaks_f],'P')
#%%
S_pica0 = S_pica[0,:,:]
fig = plt.figure()
fig.suptitle("Periodic Component Analysis")
plt.subplot(811)
plt.plot(t,S_pica0[0,:])
plt.plot(peaks_f/Fs,S_pica0[0,:][peaks_f],"X")
plt.title("T1= "+str(round(P_pica[0],3))+" s")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.subplot(812)
plt.plot(t,S_pica0[1,:])
plt.plot(peaks_f/Fs,S_pica0[1,:][peaks_f],"X")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.subplot(813)
plt.plot(t,S_pica0[2,:])
plt.plot(peaks_f/Fs,S_pica0[2,:][peaks_f],"X")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.subplot(814)
plt.plot(t,S_pica0[3,:])
plt.plot(peaks_f/Fs,S_pica0[3,:][peaks_f],"X")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.subplot(815)
plt.plot(t,S_pica0[4,:])
plt.plot(peaks_f/Fs,S_pica0[4,:][peaks_f],"X")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.subplot(816)
plt.plot(t,S_pica0[5,:])
plt.plot(peaks_f/Fs,S_pica0[5,:][peaks_f],"X")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.subplot(817)
plt.plot(t,S_pica0[6,:])
plt.plot(peaks_f/Fs,S_pica0[6,:][peaks_f],"X")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.subplot(818)
plt.plot(t,S_pica0[7,:])
plt.plot(peaks_f/Fs,S_pica0[7,:][peaks_f],"X")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.xlabel("Time(s)")
#%%
S_pica1 = S_pica[1,:,:]
fig = plt.figure()
fig.suptitle("Periodic Component Analysis")
plt.subplot(811)
plt.plot(t,S_pica1[0,:])
plt.plot(peaks/Fs,S_pica1[0,:][peaks],"X")
plt.title("T2= "+str(round(P_pica[1],3))+" s")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.subplot(812)
plt.plot(t,S_pica1[1,:])
plt.plot(peaks/Fs,S_pica1[1,:][peaks],"X")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.subplot(813)
plt.plot(t,S_pica1[2,:])
plt.plot(peaks/Fs,S_pica1[2,:][peaks],"X")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.subplot(814)
plt.plot(t,S_pica1[3,:])
plt.plot(peaks/Fs,S_pica1[3,:][peaks],"X")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.subplot(815)
plt.plot(t,S_pica1[4,:])
plt.plot(peaks/Fs,S_pica1[4,:][peaks],"X")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.subplot(816)
plt.plot(t,S_pica1[5,:])
plt.plot(peaks/Fs,S_pica1[5,:][peaks],"X")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.subplot(817)
plt.plot(t,S_pica1[6,:])
plt.plot(peaks/Fs,S_pica1[6,:][peaks],"X")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.subplot(818)
plt.plot(t,S_pica1[7,:])
plt.plot(peaks/Fs,S_pica1[7,:][peaks],"X")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.xlabel("Time(s)")
#%%
S_a0 = S_a[0,:,:]
fig = plt.figure()
fig.suptitle("AMUSE")
plt.subplot(811)
plt.plot(t,S_a0[0,:])
plt.plot(peaks_f/Fs,S_a0[0,:][peaks_f],"X")
plt.title("T1= "+str(round(P_a[0],3))+" s")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.subplot(812)
plt.plot(t,S_a0[1,:])
plt.plot(peaks_f/Fs,S_a0[1,:][peaks_f],"X")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.subplot(813)
plt.plot(t,S_a0[2,:])
plt.plot(peaks_f/Fs,S_a0[2,:][peaks_f],"X")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.subplot(814)
plt.plot(t,S_a0[3,:])
plt.plot(peaks_f/Fs,S_a0[3,:][peaks_f],"X")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.subplot(815)
plt.plot(t,S_a0[4,:])
plt.plot(peaks_f/Fs,S_a0[4,:][peaks_f],"X")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.subplot(816)
plt.plot(t,S_a0[5,:])
plt.plot(peaks_f/Fs,S_a0[5,:][peaks_f],"X")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.subplot(817)
plt.plot(t,S_a0[6,:])
plt.plot(peaks_f/Fs,S_a0[6,:][peaks_f],"X")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.subplot(818)
plt.plot(t,S_a0[7,:])
plt.plot(peaks_f/Fs,S_a0[7,:][peaks_f],"X")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.xlabel("Time(s)")
plt.waitforbuttonpress() ; plt.close()
#%%
S_a1 = S_a[2,:,:]
fig = plt.figure()
fig.suptitle("AMUSE")
plt.subplot(811)
plt.plot(t,S_a1[0,:])
plt.plot(peaks/Fs,S_a1[0,:][peaks],"X")
plt.title("T2= "+str(round(P_a[2],3))+" s")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.subplot(812)
plt.plot(t,S_a1[1,:])
plt.plot(peaks/Fs,S_a1[1,:][peaks],"X")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.subplot(813)
plt.plot(t,S_a1[2,:])
plt.plot(peaks/Fs,S_a1[2,:][peaks],"X")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.subplot(814)
plt.plot(t,S_a1[3,:])
plt.plot(peaks/Fs,S_a1[3,:][peaks],"X")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.subplot(815)
plt.plot(t,S_a1[4,:])
plt.plot(peaks/Fs,S_a1[4,:][peaks],"X")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.subplot(816)
plt.plot(t,S_a1[5,:])
plt.plot(peaks/Fs,S_a1[5,:][peaks],"X")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.subplot(817)
plt.plot(t,S_a1[6,:])
plt.plot(peaks/Fs,S_a1[6,:][peaks],"X")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.subplot(818)
plt.plot(t,S_a1[7,:])
plt.plot(peaks/Fs,S_a1[7,:][peaks],"X")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.xlabel("Time(s)")
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
#%%
S_ff = np.fft.fft(S[7,:])
S_fft =S_ff[:int(n_samples/2)+1]
S_fft = np.real( S_fft*S_fft.conj())
F = [ (Fs/n_samples)*x for x in range(0,n_samples)] ; F = F[:int(n_samples/2)+1]
plt.plot(F,S_fft)
plt.title("FFT of observations")
plt.xlabel("Frequency(Hz)")



