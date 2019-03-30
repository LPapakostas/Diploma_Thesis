# import functions from another files
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal 
import time 
from piCA import piCA
import period_check_correlation as pcc

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
minlag,maxlag = 0.0,1.2
mode = 'hard'
if mode == 'easy':
    B = np.array([(0.9,0.05,0.05),(0.05,0.9,0.05),(0.05,0.05,0.9)])
elif mode == 'normal':
    B = np.array([(0.6,0.3,0.1),(0.1,0.6,0.3),(0.3,0.1,0.6)])
elif mode == 'hard':
    B = np.array([ (-0.5814322 , -2.96086181 , 1.23499346) , \
                  (-0.87836063 , 0.2580805 , 1.52501841) , \
                  (1.20337885 ,  0.8913265 , 0.91867676)])
else :
    raise ValueError("mode variable has undentified value")
X = np.dot(B,S)

E_pica,K_pica,W_pica,P_pica= piCA(X,Fs,minlag,maxlag,preproc=True,norm=True)
E_a,K_a,W_a,P_a= piCA(X,Fs,minlag,maxlag,preproc=True,norm=True,amuse=True)
#E_c,K_c,W_c,P_c= piCA(X,Fs,minlag,maxlag,preproc=True,norm=True,circ=True)
# As we know from piCA theory, the first row of W[i,:,:] matrix repsesents 
#the signal with minimum periodicity error in this k
Wpica = np.asarray([ W_pica[i,0,:] for i in range(W_pica.shape[0])])
Wa = np.asarray([ W_a[i,0,:] for i in range(W_a.shape[0])])
#Wc = np.asarray([ W_c[i,0,:] for i in range(W_c.shape[0])])
# Correct matrix
S_pica = (Wpica @ K_pica )@X ; S_a = (Wa @ K_a )@X ; #S_c = (Wc @ K_c )@X
print("Time Evaluated:",time.time()-start,"seconds")
del start

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
plt.title("T3= "+str(P_pica[2])+" s")
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
plt.title("T3= "+str(P_a[2])+" s")
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


    
