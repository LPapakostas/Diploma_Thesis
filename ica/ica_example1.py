import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from fastica import fastica
from scipy import signal 
import time

start = time.time() ; noise_flag = True
Fs = 500 ; duration = 5 ; n_samples = int(duration*Fs)
t = np.asarray([(1./Fs)*t for t in range(0,n_samples)])
# Given known periods
T1,T2,T3 = 0.11,0.25,0.47
# Technical signals
s1 = signal.square((2*np.pi/T1)*t,0.5) + int(noise_flag)*np.random.normal(0,0.3,n_samples)
s2 = np.sin((2*np.pi/T2)*t) + int(noise_flag)*np.random.normal(0,0.1,n_samples)
s3 = signal.sawtooth((2*np.pi/T3)*t) + int(noise_flag)*np.random.normal(0,0.4,n_samples)

S = (np.c_[s1,s2,s3]).T 
del s1,s2,s3
mode = 'hard'
if mode == 'easy':
    B = np.array([(0.9,0.05,0.05),(0.05,0.9,0.05),(0.05,0.05,0.9)])
elif mode == 'normal':
    B = np.array([(0.6,0.3,0.1),(0.1,0.6,0.3),(0.3,0.1,0.6)])
elif mode == 'hard':
    B = np.array([ (-0.5814322 , -2.96086181 , 1.23499346) , \
                  (-0.87836063 , 0.2580805 , 1.52501841) , \
                  (1.20337885 ,  0.8913265 , 0.91867676)])
    print(B, la.det(B))
else :
    raise ValueError("mode variable has undentified value")
X = np.dot(B,S)

ica_comp = min(X.shape)
# The lines of X present the signals and the columns present the 
# observations of each signal at kTs time 
init = np.random.normal(size = (ica_comp,ica_comp))
#algorithm = 'symmetric' , 'deflation'
Kd,Wd,Yd,Zd = fastica(X,algorithm = 'deflation',whiten = True,fc = 'cube',w_init = init)
Ks,Ws,Ys,Zs = fastica(X,algorithm = 'symmetric',whiten = True,fc = 'cube',w_init = init)
print("Time Evaluated:",time.time()-start,"seconds")
del start
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

