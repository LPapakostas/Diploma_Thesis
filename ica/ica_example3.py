import numpy as np
import matplotlib.pyplot as plt
from fastica import fastica
import time
            
start = time.time(); noise_flag = False

# contains measurements with Fs = 720Hz and duration = 10 sec
Fs = 720 ; duration = 10 ; n_samples = int(duration*Fs)
t = [(1./Fs)*t for t in range(0,n_samples)]
t = np.asarray(t)
# We dont know the period of those signals
s1 = (np.loadtxt('C:\\Users\\Labis Papakwstas\\Desktop\\Diplwmatikh\\Data\\aami3a.txt'))[:,1]
s2 = (np.loadtxt('C:\\Users\\Labis Papakwstas\\Desktop\\Diplwmatikh\\Data\\aami3b.txt'))[:,1]
s3 = (np.loadtxt('C:\\Users\\Labis Papakwstas\\Desktop\\Diplwmatikh\\Data\\aami4a.txt'))[:,1]
    
S = (np.c_[s1,s2,s3]).T 
del s1,s2,s3
mode = 'easy'
if mode == 'easy':
    B = np.array([(0.9,0.05,0.05),(0.05,0.9,0.05),(0.05,0.05,0.9)])
elif mode == 'normal':
    B = np.array([(0.6,0.3,0.1),(0.1,0.6,0.3),(0.3,0.1,0.6)])
elif mode == 'hard':
    B = np.random.randn(S.shape[0],S.shape[0])
else :
    raise ValueError("mode variable has undentified value")
X = np.dot(B,S)

ica_comp = min(S.shape)
# The lines of X present the signals and the columns present the 
# observations of each signal at kTs time 
init = np.random.normal(size = (ica_comp,ica_comp))
#algorithm = 'symmetric' , 'deflation'
Kd,Wd,Yd,Zd = fastica(S,algorithm = 'deflation',whiten=True,fc = 'cube',w_init = init)
Ks,Ws,Ys,Zs = fastica(S,algorithm = 'symmetric',whiten=True,fc = 'cube',w_init = init)
print("Time Evaluated:",time.time()-start,"seconds")
#%%
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
#%%
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