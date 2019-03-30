import numpy as np
import matplotlib.pyplot as plt
from fastica import fastica
from scipy import signal
import time

start = time.time(); noise_flag = False
Fs = 1000 ; duration = 12; n_samples = int(duration*Fs)
t = [(1./Fs)*t for t in range(0,n_samples)]
t = np.asarray(t)

S = (np.loadtxt('C:\\Users\\Labis Papakwstas\\Desktop\\Diplwmatikh\\Data\\r01.txt'))[:,1:6]
S = S.T ; X = S[:,:n_samples]
#%%
top=0.941 ; bottom=0.122 ;left=0.071 ;right=0.981 ; hspace=0.512 ;wspace=0.15
fig = plt.figure()
fig.suptitle("Source Signals")
plt.subplot(511)
plt.plot(t,X[0,:])
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.subplot(512)
plt.plot(t,X[1,:])
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
#%%
direct_ECG = X[0,:]
peaks,_ = signal.find_peaks(direct_ECG,distance = 250)
#%%
# mallon filtrarisma
X_ica = X[1:5,:]
ica_comp = min(X_ica.shape)
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
plt.subplot(411)
plt.plot(t,Yd[0,:])
plt.plot(peaks/Fs,Yd[0,:][peaks],"X")
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.subplot(412)
plt.plot(t,Yd[1,:])
plt.plot(peaks/Fs,Yd[1,:][peaks],"X")
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.subplot(413)
plt.plot(t,Yd[2,:])
plt.plot(peaks/Fs,Yd[2,:][peaks],"X")
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.subplot(414)
plt.plot(t,Yd[3,:])
plt.plot(peaks/Fs,Yd[3,:][peaks],"X")
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.xlabel("Time(s)")
plt.waitforbuttonpress() ; plt.close()
#%%
fig = plt.figure()
fig.suptitle("Symmetric Method")
plt.subplot(411)
plt.plot(t,Ys[0,:])
plt.plot(peaks/Fs,Ys[0,:][peaks],"X")
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.subplot(412)
plt.plot(t,Ys[1,:])
plt.plot(peaks/Fs,Ys[1,:][peaks],"X")
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.subplot(413)
plt.plot(t,Ys[2,:])
plt.plot(peaks/Fs,Ys[2,:][peaks],"X")
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.subplot(414)
plt.plot(t,Ys[3,:])
plt.plot(peaks/Fs,Ys[3,:][peaks],"X")
plt.xlabel("Time(s)")
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.waitforbuttonpress() ; plt.close()