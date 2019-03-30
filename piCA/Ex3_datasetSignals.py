# import functions from another files
import numpy as np
import matplotlib.pyplot as plt
import time 
from piCA import piCA
import period_check_correlation as pcc

start = time.time(); noise_flag = False

  # contains measurements with Fs = 720Hz and duration = 10 sec
Fs = 720 ; duration = 10 ; n_samples = int(duration*Fs)
t = [(1./Fs)*t for t in range(0,n_samples)]
t = np.asarray(t)
# We dont know the period of those signals
s1 = (np.loadtxt('C:\\Users\\Labis Papakwstas\\Desktop\\Diplwmatikh\\Data\\aami3a.txt'))[:,1]
s2 = (np.loadtxt('C:\\Users\\Labis Papakwstas\\Desktop\\Diplwmatikh\\Data\\aami3b.txt'))[:,1]
s3 = (np.loadtxt('C:\\Users\\Labis Papakwstas\\Desktop\\Diplwmatikh\\Data\\aami4a.txt'))[:,1]
    
minlag,maxlag = 0.0,5.2

S = (np.c_[s1,s2,s3]).T 
del s1,s2,s3
mode = 'hard'
if mode == 'easy':
    B = np.array([(0.9,0.05,0.05),(0.05,0.9,0.05),(0.05,0.05,0.9)])
elif mode == 'normal':
    B = np.array([(0.6,0.3,0.1),(0.1,0.6,0.3),(0.3,0.1,0.6)])
elif mode == 'hard':
    B = np.array([(-1.33641602,  2.47580729, -2.072029 ),
       ( 3.04875862, -0.26041481,  0.52304732),
       (-0.46639687, -0.03951659, -0.82693773)])
else :
    raise ValueError("mode variable has undentified value")
X = np.dot(B,S)

E_pica,K_pica,W_pica,P_pica = piCA(S,Fs,minlag,maxlag,preproc=True,norm=True,amuse=False)
# 2.1 1 1 parameters
E_a,K_a,W_a,P_a = piCA(S,Fs,minlag,maxlag,preproc=True,norm=True,amuse=True)
# 0.99 5 1 parameters
#%%
Wpica = np.asarray([ W_pica[i,0,:] for i in range(W_pica.shape[0])])
Wa = np.asarray([ W_a[i,0,:] for i in range(W_a.shape[0])])
S_pica = (Wpica @ K_pica )@S ; S_a = (Wa @ K_a )@S
#%%
P_pica = pcc.check(S_pica,P_pica,Fs,t)
P_a = pcc.check(S_a,P_a,Fs,t)
#%%

# plotting results
#top=0.941 ; bottom=0.122 ;left=0.071 ;right=0.981 ; hspace=0.512 ;wspace=0.15
top=0.941 ; bottom=0.122 ;left=0.071 ;right=0.981 ; hspace=0.512 ;wspace=0.15
fig = plt.figure()
fig.suptitle("AAMI Dataset")
plt.subplot(311)
plt.plot(t,S[0,:])
plt.xlabel("Time(s)")
plt.ylabel("Amplitude(mV)")
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.subplot(312)
plt.plot(t,S[1,:])
plt.xlabel("Time(s)")
plt.ylabel("Amplitude(mV)")
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.subplot(313)
plt.plot(t,S[2,:])
plt.xlabel("Time(s)")
plt.ylabel("Amplitude(mV)")
plt.subplots_adjust(top=0.941, bottom=0.122, left=0.071, right=0.981, hspace=0.512, wspace=0.15)
plt.waitforbuttonpress() ; plt.close()
#%%
fig = plt.figure()
fig.suptitle("Periodic Component Analysis")
plt.subplot(311)
plt.plot(t,S_pica[0,:])
plt.xlabel("Time(s)")
plt.title("T1= "+str(round(P_pica[0],3))+" s")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.subplot(312)
plt.plot(t,S_pica[1,:])
plt.title("T2= "+str(round(P_pica[1],3))+" s")
plt.xlabel("Time(s)")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.subplot(313)
plt.plot(t,S_pica[2,:])
plt.title("T3= "+str(round(P_pica[2],3))+" s")
plt.xlabel("Time(s)")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.waitforbuttonpress() ; plt.close()

#%%
fig = plt.figure()
fig.suptitle("AMUSE")
plt.subplot(311)
plt.plot(t,S_a[0,:])
plt.xlabel("Time(s)")
plt.title("T1= "+str(round(P_a[0],3))+" s")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.subplot(312)
plt.plot(t,S_a[1,:])
plt.title("T2= "+str(round(P_a[1],3))+" s")
plt.xlabel("Time(s)")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.subplot(313)
plt.plot(t,S_a[2,:])
plt.title("T3= "+str(round(P_a[2],3))+" s")
plt.xlabel("Time(s)")
plt.subplots_adjust(top=0.891,bottom=0.122,left=0.071,right=0.971,hspace=1.0,wspace=0.15)
plt.waitforbuttonpress() ; plt.close()
#%%
min_t = int(minlag*Fs) ; max_t = int(maxlag*Fs)
t_e = t[min_t:max_t+1]
plt.plot(t_e,E_pica); plt.title("Minimum Periodicity Error for piCA")
plt.plot(t_e,np.zeros_like(t_e),'--')
plt.xlabel("Time(s)")
plt.waitforbuttonpress() ; plt.close()
plt.plot(t_e,E_a); plt.title("Minimum Periodicity Error for AMUSE")
plt.plot(t_e,np.ones_like(t_e),'--')
plt.xlabel("Time(s)")
plt.waitforbuttonpress() ; plt.close()
