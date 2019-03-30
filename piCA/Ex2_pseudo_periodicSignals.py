# import functions from another files
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal 
import time 
from piCA import piCA
import misc
import period_check_correlation as pcc

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
s3 = np.concatenate((s3a,s3b)) #period of signal = 0.754s
del s3a,s3b
s3 = np.concatenate((s3,s3,s3,s3,s3,s3,s3,s3,s3,s3,s3))[:n_samples] 

s4 = np.random.normal(0,1,n_samples)
    
S = (np.c_[s2,s3,s4]).T 
del s1,s2,s3,s4

minlag,maxlag = 0.0,4.1

mode = 'hard'
if mode == 'easy':
    B = np.array([(0.9,0.05,0.05),(0.05,0.9,0.05),(0.05,0.05,0.9)])
elif mode == 'normal':
    B = np.array([(0.6,0.3,0.1),(0.1,0.6,0.3),(0.3,0.1,0.6)])
elif mode == 'hard':
    B = np.array([( 0.00740553,  1.05634461, -0.52022631),
       (-0.00531442,  0.95287344, -0.26333002),
       (-2.39713001, -0.52477855, -0.67513928)])
else :
    raise ValueError("mode variable has undentified value")
X = np.dot(B,S)

E_pica,K_pica,W_pica,P_pica= piCA(X,Fs,minlag,maxlag,preproc=True,norm=True)
E_a,K_a,W_a,P_a= piCA(X,Fs,minlag,maxlag,preproc=True,norm=True,amuse=True)
print("Time Evaluated:",time.time()-start,"seconds")
del start
#%%
Wpica = np.asarray([ W_pica[i,0,:] for i in range(W_pica.shape[0])])
Wpica = np.vstack((Wpica,W_pica[0,2,:]))
Wa = np.asarray([ W_a[i,0,:] for i in range(W_a.shape[0])])
Wa = np.vstack((Wa,W_a[0,2,:]))
S_pica = (Wpica @ K_pica )@X ; S_a = (Wa @ K_a )@X
#%%
#Performance Index
Ppi = Wpica@K_pica@B ; Pa = Wa@K_a@B 
PI_pi = sum( sum( (Ppi -np.diag(np.diag(Ppi)))**2   ) ) / sum(sum( Ppi**2 ))
print("Performance of piCa is : ",PI_pi)
PI_a = sum( sum( (Pa -np.diag(np.diag(Pa)))**2   ) ) / sum(sum( Pa**2 ))
print("Performance of AMUSE is : ",PI_pi)
#%%
P_pica = pcc.check(S_pica,P_pica,Fs,t)
P_a = pcc.check(S_a,P_a,Fs,t)
#%%
#top=0.941 ; bottom=0.122 ;left=0.071 ;right=0.981 ; hspace=0.512 ;wspace=0.15
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
fig.suptitle("Periodic Component Analysis")
plt.subplot(311)
plt.plot(t,S_pica[0,:])
plt.xlabel("Time(s)")
plt.title("T1= "+str(P_pica[0])+" s")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.subplot(312)
plt.plot(t,S_pica[1,:])
plt.title("T2= "+str(P_pica[1])+" s")
plt.xlabel("Time(s)")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.subplot(313)
plt.plot(t,S_pica[2,:])
plt.title("Not Periodic")
plt.xlabel("Time(s)")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.waitforbuttonpress() ; plt.close()
#%%
fig = plt.figure()
fig.suptitle("AMUSE")
plt.subplot(311)
plt.plot(t,S_a[0,:])
plt.xlabel("Time(s)")
plt.title("T1= "+str(P_a[0])+" s")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.subplot(312)
plt.plot(t,S_a[1,:])
plt.title("T2= "+str(P_a[1])+" s")
plt.xlabel("Time(s)")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.subplot(313)
plt.plot(t,S_a[2,:])
plt.title("Not Periodic")
plt.xlabel("Time(s)")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.waitforbuttonpress() ; plt.close()
#%%
min_t = int(minlag*Fs) ; max_t = int(maxlag*Fs)
t_e = t[min_t:max_t+1]
plt.plot(t_e,E_pica); plt.title("Minimum Periodicity Error for piCA")
plt.plot(t_e,np.zeros_like(t_e),'--')
plt.waitforbuttonpress() ; plt.close()
plt.plot(t_e,E_a); plt.title("Minimum Periodicity Error for AMUSE")
plt.plot(t_e,np.ones_like(t_e),'--')
plt.waitforbuttonpress() ; plt.close()

