import numpy as np
import matplotlib.pyplot as plt
from fastica import fastica
from scipy import signal
import time

start = time.time(); noise_flag = False
Fs = 250 ; duration = 10 ; n_samples = int(duration*Fs)
t = [(1./Fs)*t for t in range(0,n_samples)]
t = np.asarray(t)
S = (np.loadtxt('C:\\Users\\Labis Papakwstas\\Desktop\\Diplwmatikh\\Data\\Daisy.dat'))[:,1:9]
S = (S[:n_samples]).T 
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
#%%
X_ica = S; ica_comp = min(X_ica.shape)
# The lines of X present the signals and the columns present the 
# observations of each signal at kTs time 
init = np.random.normal(size = (ica_comp,ica_comp))
#algorithm = 'symmetric' , 'deflation'
Kd,Wd,Yd,Zd = fastica(X_ica,algorithm = 'deflation',whiten=True,fc = 'cube',w_init = init)
Ks,Ws,Ys,Zs = fastica(X_ica,algorithm = 'symmetric',whiten=True,fc = 'cube',w_init = init)
print("Time Evaluated:",time.time()-start,"seconds")
#%%
fig = plt.figure()
fig.suptitle("Deflation Method")
plt.subplot(811)
plt.plot(t,Yd[0,:])
plt.plot(peaks/Fs,Yd[0,:][peaks],"X")
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.subplot(812)
plt.plot(t,Yd[1,:])
plt.plot(peaks/Fs,Yd[1,:][peaks],"X")
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.subplot(813)
plt.plot(t,Yd[2,:])
plt.plot(peaks/Fs,Yd[2,:][peaks],"X")
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.subplot(814)
plt.plot(t,Yd[3,:])
plt.plot(peaks/Fs,Yd[3,:][peaks],"X")
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.subplot(815)
plt.plot(t,Yd[4,:])
plt.plot(peaks/Fs,Yd[4,:][peaks],"X")
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.subplot(816)
plt.plot(t,Yd[5,:])
plt.plot(peaks/Fs,Yd[5,:][peaks],"X")
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.subplot(817)
plt.plot(t,Yd[6,:])
plt.plot(peaks/Fs,Yd[6,:][peaks],"X")
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.subplot(818)
plt.plot(t,Yd[7,:])
plt.plot(peaks/Fs,Yd[7,:][peaks],"X")
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.xlabel("Time(s)")
#%%
fig = plt.figure()
fig.suptitle("Symmetric Method")
plt.subplot(811)
plt.plot(t,Ys[0,:])
plt.plot(peaks/Fs,Ys[0,:][peaks],"X")
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.subplot(812)
plt.plot(t,Ys[1,:])
plt.plot(peaks/Fs,Ys[1,:][peaks],"X")
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.subplot(813)
plt.plot(t,Ys[2,:])
plt.plot(peaks/Fs,Ys[2,:][peaks],"X")
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.subplot(814)
plt.plot(t,Ys[3,:])
plt.plot(peaks/Fs,Ys[3,:][peaks],"X")
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.subplot(815)
plt.plot(t,Ys[4,:])
plt.plot(peaks/Fs,Ys[4,:][peaks],"X")
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.subplot(816)
plt.plot(t,Ys[5,:])
plt.plot(peaks/Fs,Ys[5,:][peaks],"X")
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.subplot(817)
plt.plot(t,Ys[6,:])
plt.plot(peaks/Fs,Ys[6,:][peaks],"X")
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.subplot(818)
plt.plot(t,Ys[7,:])
plt.plot(peaks/Fs,Ys[7,:][peaks],"X")
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.xlabel("Time(s)")