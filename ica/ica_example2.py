import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la
from fastica import fastica
from scipy import signal 
import time
import random

start = time.time(); noise_flag = False

# Sampling frequency = 500 Hz , duration = 2.1 sec
Fs = 1000 ; duration = 5 ; n_samples = int(duration*Fs)
t = np.asarray([(1./Fs)*t for t in range(0,n_samples)])

s1 = signal.square((2*np.pi/(0.47))*t,0.5) 
# Pseudo-periodic signal
tt2 = -2 ; tt3 = -7
print(tt2,tt3) 

s2 = np.ones(300+10*tt2) # period = 300/1000 = 0.3s
s2 = np.concatenate((s2,signal.gaussian(200,std=20))) #period_ofgaussian = 200/1000 = 0.2s
s2 = np.concatenate((s2,s2,s2,s2,s2,s2,s2,s2,s2,s2,s2,s2))[:n_samples] #period of signal = 0.5s

s3a = np.abs(signal.sawtooth((2*np.pi/0.6)*t))[:200] #period = 100/500 = 0.2s
s3b = np.abs(np.sin((2*np.pi/0.3)*t))[:(514+10*tt3)]# period = 227/500 = 0.514s
s3 = np.concatenate((s3a,s3b)) #period of signal = 0.654s
del s3a,s3b
s3 = np.concatenate((s3,s3,s3,s3,s3,s3,s3,s3,s3,s3,s3))[:n_samples] 

s4 = np.random.normal(0,1,n_samples)
    
S = (np.c_[s2,s3,s4]).T 
del s1,s2,s3,s4
mode = 'hard'
if mode == 'easy':
    B = np.array([(0.9,0.05,0.05),(0.05,0.9,0.05),(0.05,0.05,0.9)])
elif mode == 'normal':
    B = np.array([(0.6,0.3,0.1),(0.1,0.6,0.3),(0.3,0.1,0.6)])
elif mode == 'hard':
    #B = np.random.randn(S.shape[0],S.shape[0])
    B = np.array([( 0.00740553,  1.05634461, -0.52022631),
       (-0.00531442,  0.95287344, -0.26333002),
       (-2.39713001, -0.52477855, -0.67513928)])
else :
    raise ValueError("mode variable has undentified value")
X = np.dot(B,S)

ica_comp = min(X.shape)
# The lines of X present the signals and the columns present the 
# observations of each signal at kTs time 
init = np.random.normal(size = (ica_comp,ica_comp))
#algorithm = 'symmetric' , 'deflation'
Kd,Wd,Yd,Zd = fastica(X,algorithm = 'deflation',whiten=True,fc = 'cube',w_init = init)
Ks,Ws,Ys,Zs = fastica(X,algorithm = 'symmetric',whiten=True,fc = 'cube',w_init = init)

print("Time Evaluated:",time.time()-start,"seconds")
#%%
#Performance Index
Ps = Ws@Ks@B ; Pd = Wd@Kd@B 
# For ICA, we must bring P matrix into diagonal form becauses FastICA method 
# cannot find the proper order of estimated signals 
# Find the position of maximum element for each row. This wil represent the correct order
# of Indipendent Components
ks = [np.argmax(abs(Ps[i,:])) for i in range(Ps.shape[0])]
kd = [np.argmax(abs(Pd[i,:])) for i in range(Pd.shape[0])]
# Find the position of each row in order P matrix to be as diagonal as it can
ks = np.argsort(np.asarray(ks)) ; kd = np.argsort(np.asarray(kd))
Ps = Ps[ks,:] ; Pd = Pd[kd,:]
PIs = sum( sum( (Ps -np.diag(np.diag(Ps)))**2   ) ) / sum(sum( Ps**2 ))
print("Performance of Symmetric method is : ",PIs)
PId = sum( sum( (Pd -np.diag(np.diag(Pd)))**2   ) ) / sum(sum( Pd**2 ))
print("Performance of Deflational method is : ",PId)
#%%
#Plot parameters
'''
top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15
'''
top=0.941 ; bottom=0.122 ;left=0.071 ;right=0.981 ; hspace=0.512 ;wspace=0.15
fig = plt.figure()
fig.suptitle("Source Signals")
plt.subplot(311)
plt.plot(t,S[0,:])
plt.xlabel("Time(s)")
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.subplot(312)
plt.plot(t,S[1,:])
plt.xlabel("Time(s)")
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.subplot(313)
plt.plot(t,S[2,:])
plt.xlabel("Time(s)")
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.waitforbuttonpress() ; plt.close()

fig = plt.figure()
fig.suptitle("Mixed Signals")
plt.subplot(311)
plt.plot(t,X[0,:])
plt.xlabel("Time(s)")
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.subplot(312)
plt.plot(t,X[1,:])
plt.xlabel("Time(s)")
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.subplot(313)
plt.plot(t,X[2,:])
plt.xlabel("Time(s)")
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.waitforbuttonpress() ; plt.close()

#%%
fig = plt.figure()
fig.suptitle("Deflation Method")
plt.subplot(311)
plt.plot(t,Yd[0,:])
plt.xlabel("Time(s)")
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.subplot(312)
plt.plot(t,Yd[1,:])
plt.xlabel("Time(s)")
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.subplot(313)
plt.plot(t,Yd[2,:])
plt.xlabel("Time(s)")
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.waitforbuttonpress() ; plt.close()

fig = plt.figure()
fig.suptitle("Symmetric Method")
plt.subplot(311)
plt.plot(t,Ys[0,:])
plt.xlabel("Time(s)")
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.subplot(312)
plt.plot(t,Ys[1,:])
plt.xlabel("Time(s)")
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.subplot(313)
plt.plot(t,Ys[2,:])
plt.xlabel("Time(s)")
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.waitforbuttonpress() ; plt.close()


