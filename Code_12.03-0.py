#!/usr/bin/env python
# coding: utf-8

# In[60]:


import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import time

import seaborn as sns


# In[81]:


############################## LIF ##############################

# global constants
SHORT_TEST = True   # True => very small test run; False => full run (6000 trials per stim)
SEED = 5447
np.random.seed(SEED)


# simulation details
#itvl_num = 6000
iter_num = 2#10
trial_num = 6#300
dt=.1#.01
total_T = iter_num * trial_num * 4 * 1000
time=np.arange(0,total_T,dt)
Nt=len(time)
lr = 0.001

# stimulus values (ms^-1)
s1 = 180.0/1000
s2 = 185.0/1000
s0 = 182.5/1000
ss = [s1,s2]

# functions
# ΣJ Σh(t-t')
# g(s)
g = lambda s: (s)
# Σ_kMζ
def noise(N_x):
    zeta_i = np.random.normal(0.0, var_ind, N_x)
    zeta_0 = np.random.normal(0.0, var_s)
    return sigma_ind * zeta_i + sigma_s * zeta_0

# linear estimator, locally optimal
sest_LIF = lambda w,re,ri: ((1-w)*re+w*ri)
# Fisher information (~ -loss)
#fisherI = lambda w:


# LIF network constants
N_arr = [75,125,250,500]
Iw_res = np.zeros(4)

for kk in range(len(N_arr)):
    N = N_arr[kk]                       # number of neurons in network (tunable)
    exc_frac = 0.8                      # fraction excitatory
    Ne = int(np.round(exc_frac * N))    # number of excitatory neurons
    Ni = N - Ne                         # number of inhibitory neurons
    T = 2*1000                          # observation time window (s->ms)
    taum = 20                           # membrane time constant (ms)
    taue = 2                            # excitatory synaptic time constant (ms)
    taui = 3                            # inhibitory synaptic time constant (ms)
    theta_i = 1                         # threshold 
    V_r = 0                             # reset
    var_ind = 76.5/1000                 # variance of the independent noise (s^-1->ms^-1)
    var_s = 3.5/1000                    # variance of the shared noise (s^-1->ms^-1)
    jee = (6/(Ne-1))                    # E -> E connection strength
    jei = (-9.5/Ni)                     # I -> E connection strength
    jie = (7.6/Ne)                      # E -> I connection strength
    jii = (-11.2/(Ni-1))                # I -> I connection strength
    sigma_ind = np.sqrt(var_ind)
    sigma_s = np.sqrt(var_s)

    # connection matrices
    Jee = jee*np.ones((Ne,Ne))
    Jei = jei*np.ones((Ne,Ni))
    Jie = jie*np.ones((Ni,Ne))
    Jii = jii*np.ones((Ni,Ni))
    np.fill_diagonal(Jee,0)
    np.fill_diagonal(Jei,0)
    np.fill_diagonal(Jie,0)
    np.fill_diagonal(Jii,0)

    # initialization of membrane potentials randomly
    Ve=np.random.rand(len(ss),Ne)
    Vi=np.random.rand(len(ss),Ni)
    # initialize all else as zeros
    Se=np.zeros((len(ss),Ne,len(time)))
    Si=np.zeros((len(ss),Ni,len(time)))
    # spike counts
    ne = np.zeros((len(ss),Ne))
    ni = np.zeros((len(ss),Ni))
    # firing rates
    re = np.zeros(len(ss))
    ri = np.zeros(len(ss))
    # estimator result
    slif_train = np.empty((0,len(ss)))
    slif_test = np.empty((0,len(ss)))
    # fisher info
    Iw_train = np.zeros((trial_num*iter_num))
    Iw_test = np.zeros((trial_num*iter_num))
    # optimal weights
    w_opt = -1
    

    # simulation
    for ii in range(iter_num):      # 10 iterations
        # weight(s) init
        w = np.zeros(trial_num+1)
        if ii == 0: w[0] = np.random.rand()
        else: w[0] = w_opt

        print(f"iteration {ii}, training: ")
        ### Training Set ###
        for jj in trange(trial_num): # 300 trials
            for s_i in range(len(ss)):       # for both stimuli (2 separate array)
                for i in range(int(T/dt)):
                    # get the current time, t
                    t = (2*ii*trial_num+jj)*int(T/dt) + i

                    # Euler step for membrane potentials
                    Ve[s_i] = Ve[s_i] + dt * (-Ve[s_i]/taum+Jee@Se[s_i,:,t]+Jei@Si[s_i,:,t]+g(ss[s_i])+noise(Ne)) 
                    Vi[s_i] = Vi[s_i] + dt * (-Vi[s_i]/taum+Jie@Se[s_i,:,t]+Jii@Si[s_i,:,t]+g(ss[s_i])+noise(Ni))
                    #if i < 10: print(Ve[s_i][0])
                    #if i == 10: print()

                    # Find which excitatory neurons spiked.
                    Inds=np.nonzero(Ve[s_i]>=theta_i)[0]
                    # Reset membrane potentials
                    Ve[s_i][Inds]=V_r    
                    # Store spikes as delta functions
                    Se[s_i,Inds,t+1]=1/dt

                    # Now do the same for inhibitory neurons
                    Inds=np.nonzero(Vi[s_i]>=theta_i)[0] 
                    # Reset membrane potentials
                    Vi[s_i][Inds]=V_r
                    # Store spikes as delta functions
                    Si[s_i,Inds,t+1]=1/dt

                # get spike count, firing rate
                #low=(2*ii*trial_num+jj)*int(T/dt)
                #high=(2*ii*trial_num+jj)*int(T/dt)+int(T/dt)
                Sn_len = (2*ii+1)*trial_num*T
                ne[s_i] = np.sum(Se[s_i],axis=1)
                ni[s_i] = np.sum(Si[s_i],axis=1)
                re[s_i] = np.mean(ne[s_i]/Sn_len)                # unit: ms^-1
                ri[s_i] = np.mean(ni[s_i]/Sn_len)                # size: (2,)
                print(re,ri)

            # do prediction with estimator
            slif_train = np.vstack((slif_train, sest_LIF(w[jj],re,ri)))

            # calculate fisher info (~ -loss)
            if (np.var(slif_train[:,0])+np.var(slif_train[:,1]) != 0): # require nonzero
                Iw_train[ii*trial_num+jj] = (np.mean(slif_train[:,0])-np.mean(slif_train[:,1]))**2/(s1-s2)**2 * 2/(np.var(slif_train[:,0])+np.var(slif_train[:,1]))

            # update weights ; considered only if jj >= 2, jj=0: I_w not defined, jj=1: only 1 I_w
            if (ii>0 or jj>=2):
                w[jj+1] = w[jj] + (Iw_train[ii*trial_num+jj] - Iw_train[ii*trial_num+jj-1])/(w[jj]-w[jj-1]) * lr
                print((Iw_train[ii*trial_num+jj] - Iw_train[ii*trial_num+jj-1])/(w[jj]-w[jj-1])*lr)
            else:
                w[jj+1] = np.random.rand()
                #print("PRE")

        # choose best weights
        w_opt = w[np.argmax(Iw_train[ii*trial_num:ii*trial_num+trial_num])]

        # print results
        print(f"estimator prediction: {slif_train}")
        print(f"weight history: {w}")
        print(f"optimal weight: {w_opt}")
        print()

        # clear estimator result
        #slif = np.empty((0,len(ss)))


        print(f"iteration {ii}, test: ")
        ### Test Set ###
        for jj in trange(trial_num): # 300 trials
            for s_i in range(len(ss)):       # for both stimuli (2 separate array)
                for i in range(int(T/dt)):
                    # get the current time, t
                    t = ((2*ii+1)*trial_num+jj)*int(T/dt) + i

                    # Euler step for membrane potentials
                    Ve[s_i] = Ve[s_i] + dt * (-Ve[s_i]/taum+Jee@Se[s_i,:,t]+Jei@Si[s_i,:,t]+g(ss[s_i])+noise(Ne)) 
                    Vi[s_i] = Vi[s_i] + dt * (-Vi[s_i]/taum+Jie@Se[s_i,:,t]+Jii@Si[s_i,:,t]+g(ss[s_i])+noise(Ni)) 

                    # Find which excitatory neurons spiked.
                    Inds=np.nonzero(Ve[s_i]>=theta_i)[0]
                    # Reset membrane potentials
                    Ve[s_i][Inds]=V_r    
                    # Store spikes as delta functions
                    if (t+1 < iter_num*2*trial_num*int(T/dt)):
                        Se[s_i,Inds,t+1]=1/dt

                    # Now do the same for inhibitory neurons
                    Inds=np.nonzero(Vi[s_i]>=theta_i)[0] 
                    # Reset membrane potentials
                    Vi[s_i][Inds]=V_r
                    # Store spikes as delta functions
                    if (t+1 < iter_num*2*trial_num*int(T/dt)):
                        Si[s_i,Inds,t+1]=1/dt

                # get spike count, firing rate
                #low=(2*ii*trial_num+jj)*int(T/dt)
                #high=(2*ii*trial_num+jj)*int(T/dt)+int(T/dt)
                Sn_len = 2*(ii+1)*trial_num*T
                ne[s_i] = np.sum(Se[s_i],axis=1)
                ni[s_i] = np.sum(Si[s_i],axis=1)
                re[s_i] = np.mean(ne[s_i]/Sn_len)                 # unit: ms^-1
                ri[s_i] = np.mean(ni[s_i]/Sn_len)                 # size: (2,)

            # do prediction with estimator
            slif_test = np.vstack((slif_test, sest_LIF(w_opt,re,ri)))

            # calculate fisher info (~ -loss)
            if (np.var(slif_test[:,0])+np.var(slif_test[:,1]) != 0): # require nonzero
                Iw_test[ii*trial_num+jj] = (np.mean(slif_test[:,0])-np.mean(slif_test[:,1]))**2/(s1-s2)**2 * 2/(np.var(slif_test[:,0])+np.var(slif_test[:,1]))

        # print results
        print(f"estimator prediction: {slif_test}")
        print(f"optimal weight (from training): {w_opt}")
        print()

    print(f"I_train(w): {Iw_train}")
    print(f"I_test(w): {Iw_test}")

    Iw_res[kk] = np.max(Iw_test)

    
# plots
plt.subplots(1,2,figsize=(16, 8))

plt.subplot(1,2,1)
N_axis = np.arange(0,550,0.1)
I_func = lambda N: (1/(var_s+var_ind/N))
plt.plot(N_axis,I_func(N_axis),'g.',markersize=2)
plt.xlabel('N')
#plt.xlim([0,])
#plt.ylim([0,Nx])
plt.ylabel('fisher information')
plt.title('theoretical',loc='left')
sns.despine()

plt.subplot(1,2,2)
plt.plot(N_arr,Iw_res,'g.',markersize=10)
plt.xlabel('N')
#plt.xlim([0,])
#plt.ylim([0,Nx])
plt.ylabel('fisher information')
plt.title('experimental',loc='left')
sns.despine()


# In[80]:


############################## Non-LIF ##############################

# global constants
SHORT_TEST = True   # True => very small test run; False => full run (6000 trials per stim)
SEED = 5447
np.random.seed(SEED)


# simulation details
#itvl_num = 6000
iter_num = 2#10
trial_num = 6#300
dt=.1#.01
total_T = iter_num * trial_num * 4 * 1000
time=np.arange(0,total_T,dt)
Nt=len(time)
lr = 0.1

# stimulus values (ms^-1)
s1 = 180.0/1000
s2 = 185.0/1000
s0 = 182.5/1000
ss = [s0]

# functions
# ΣJ Σh(t-t')
# g(s)
g = lambda s: (s)
# Σ_kMζ
def noise(N_x):
    zeta_i = np.random.normal(0.0, var_ind, N_x)
    zeta_0 = np.random.normal(0.0, var_s)
    return sigma_ind * zeta_i + sigma_s * zeta_0

# Fisher information (~ -loss)
#fisherI = lambda w:


# LIF network constants
N_arr = [75,125,250,500]
Iw_res = np.zeros(4)

for kk in range(len(N_arr)):
    N = N_arr[kk]                       # number of neurons in network (tunable)
    print(f"N = {N_arr[kk]}")
    exc_frac = 0.8                      # fraction excitatory
    Ne = int(np.round(exc_frac * N))    # number of excitatory neurons
    Ni = N - Ne                         # number of inhibitory neurons
    T = 2*1000                          # observation time window (s->ms)
    taum = 20                           # membrane time constant (ms)
    taue = 2                            # excitatory synaptic time constant (ms)
    taui = 3                            # inhibitory synaptic time constant (ms)
    theta_i = 1                         # threshold 
    V_r = 0                             # reset
    var_ind = 76.5/1000                 # variance of the independent noise (s^-1->ms^-1)
    var_s = 3.5/1000                    # variance of the shared noise (s^-1->ms^-1)
    jee = (6/(Ne-1))                    # E -> E connection strength
    jei = (-9.5/Ni)                     # I -> E connection strength
    jie = (7.6/Ne)                      # E -> I connection strength
    jii = (-11.2/(Ni-1))                # I -> I connection strength
    sigma_ind = np.sqrt(var_ind)
    sigma_s = np.sqrt(var_s)

    # connection matrices
    Jee = jee*np.ones((Ne,Ne))
    Jei = jei*np.ones((Ne,Ni))
    Jie = jie*np.ones((Ni,Ne))
    Jii = jii*np.ones((Ni,Ni))
    np.fill_diagonal(Jee,0)
    np.fill_diagonal(Jei,0)
    np.fill_diagonal(Jie,0)
    np.fill_diagonal(Jii,0)
    
    # linear estimator, locally optimal
    s_LIF = lambda re,ri: ( Ne/N*(1-(jee+jie))*re+Ni/N*(1-(jei+jii))*ri )

    # initialization of membrane potentials randomly
    Ve=np.random.rand(len(ss),Ne)
    Vi=np.random.rand(len(ss),Ni)
    # initialize all else as zeros
    Se=np.zeros((len(ss),Ne,len(time)))
    Si=np.zeros((len(ss),Ni,len(time)))
    # spike counts
    ne = np.zeros((len(ss),Ne))
    ni = np.zeros((len(ss),Ni))
    # firing rates
    re = np.zeros(len(ss))
    ri = np.zeros(len(ss))
    # estimator result
    slif_train = np.empty((0,len(ss)))
    slif_test = np.empty((0,len(ss)))
    # fisher info
    Iw_train = np.zeros((trial_num*iter_num))
    Iw_test = np.zeros((trial_num*iter_num))
    # optimal weights
    w_opt = -1
    

    # simulation
    for ii in range(iter_num):      # 10 iterations
        # weight(s) init
        #w = np.zeros(trial_num+1)
        #if ii == 0: w[0] = np.random.rand()
        #else: w[0] = w_opt

        print(f"iteration {ii}, training: ")
        ### Training Set ###
        for jj in trange(trial_num): # 300 trials
            for s_i in range(len(ss)):       # for both stimuli (2 separate array)
                for i in range(int(T/dt)):
                    # get the current time, t
                    t = (2*ii*trial_num+jj)*int(T/dt) + i

                    # Euler step for membrane potentials
                    Ve[s_i] = Ve[s_i] + dt * (Jee@Se[s_i,:,t]+Jei@Si[s_i,:,t]+g(ss[s_i])+noise(Ne)) 
                    Vi[s_i] = Vi[s_i] + dt * (Jie@Se[s_i,:,t]+Jii@Si[s_i,:,t]+g(ss[s_i])+noise(Ni))
                    #if i < 10: print(Ve[s_i][0])
                    #if i == 10: print()

                    # Find which excitatory neurons spiked.
                    Inds=np.nonzero(Ve[s_i]>=theta_i)[0]
                    # Reset membrane potentials
                    Ve[s_i][Inds]=V_r    
                    # Store spikes as delta functions
                    Se[s_i,Inds,t+1]=1/dt

                    # Now do the same for inhibitory neurons
                    Inds=np.nonzero(Vi[s_i]>=theta_i)[0] 
                    # Reset membrane potentials
                    Vi[s_i][Inds]=V_r
                    # Store spikes as delta functions
                    Si[s_i,Inds,t+1]=1/dt

                # get spike count, firing rate
                #low=(2*ii*trial_num+jj)*int(T/dt)
                #high=(2*ii*trial_num+jj)*int(T/dt)+int(T/dt)
                Sn_len = (2*ii+1)*trial_num*T
                ne[s_i] = np.sum(Se[s_i],axis=1)
                ni[s_i] = np.sum(Si[s_i],axis=1)
                re[s_i] = np.mean(ne[s_i]/Sn_len)                 # unit: ms^-1
                ri[s_i] = np.mean(ni[s_i]/Sn_len)                 # size: (2,)
                #print(f"re: {re} \t ri: {re}")

            # do prediction with estimator
            slif_train = np.vstack((slif_train, s_LIF(re,ri)))

            # calculate fisher info (~ -loss)
            if (np.var(slif_train[:,0]) != 0): # require nonzero
                Iw_train[ii*trial_num+jj] = 1/np.var(slif_train[:,0])

            # update weights ; considered only if jj >= 2, jj=0: I_w not defined, jj=1: only 1 I_w
            #if (ii>0 or jj>=2):
            #    w[jj+1] = w[jj] + (Iw_train[ii*trial_num+jj] - Iw_train[ii*trial_num+jj-1])/(w[jj]-w[jj-1]) * lr
            #    print((Iw_train[ii*trial_num+jj] - Iw_train[ii*trial_num+jj-1])/(w[jj]-w[jj-1])*lr)
            #else:
            #    w[jj+1] = np.random.rand()
                #print("PRE")

        # choose best weights
        #w_opt = w[np.argmax(Iw_train[ii*trial_num:ii*trial_num+trial_num])]

        # print results
        print(f"estimator prediction: {slif_train}")
        #print(f"weight history: {w}")
        #print(f"optimal weight: {w_opt}")
        print()

        # clear estimator result
        #slif = np.empty((0,len(ss)))


        print(f"iteration {ii}, test: ")
        ### Test Set ###
        for jj in trange(trial_num): # 300 trials
            for s_i in range(len(ss)):       # for both stimuli (2 separate array)
                for i in range(int(T/dt)):
                    # get the current time, t
                    t = ((2*ii+1)*trial_num+jj)*int(T/dt) + i

                    # Euler step for membrane potentials
                    Ve[s_i] = Ve[s_i] + dt * (Jee@Se[s_i,:,t]+Jei@Si[s_i,:,t]+g(ss[s_i])+noise(Ne)) 
                    Vi[s_i] = Vi[s_i] + dt * (Jie@Se[s_i,:,t]+Jii@Si[s_i,:,t]+g(ss[s_i])+noise(Ni))

                    # Find which excitatory neurons spiked.
                    Inds=np.nonzero(Ve[s_i]>=theta_i)[0]
                    # Reset membrane potentials
                    Ve[s_i][Inds]=V_r    
                    # Store spikes as delta functions
                    if (t+1 < iter_num*2*trial_num*int(T/dt)):
                        Se[s_i,Inds,t+1]=1/dt

                    # Now do the same for inhibitory neurons
                    Inds=np.nonzero(Vi[s_i]>=theta_i)[0] 
                    # Reset membrane potentials
                    Vi[s_i][Inds]=V_r
                    # Store spikes as delta functions
                    if (t+1 < iter_num*2*trial_num*int(T/dt)):
                        Si[s_i,Inds,t+1]=1/dt

                # get spike count, firing rate
                #low=(2*ii*trial_num+jj)*int(T/dt)
                #high=(2*ii*trial_num+jj)*int(T/dt)+int(T/dt)
                Sn_len = 2*(ii+1)*trial_num*T
                ne[s_i] = np.sum(Se[s_i],axis=1)
                ni[s_i] = np.sum(Si[s_i],axis=1)
                re[s_i] = np.mean(ne[s_i]/Sn_len)                 # unit: ms^-1
                ri[s_i] = np.mean(ni[s_i]/Sn_len)                 # size: (2,)

            # do prediction with estimator
            slif_test = np.vstack((slif_test, s_LIF(re,ri)))

            # calculate fisher info (~ -loss)
            if (np.var(slif_test[:,0]) != 0): # require nonzero
                Iw_test[ii*trial_num+jj] = 1/np.var(slif_test[:,0])

        # print results
        print(f"estimator prediction: {slif_test}")
        #print(f"optimal weight (from training): {w_opt}")
        print()

    print(f"I_train(w): {Iw_train}")
    print(f"I_test(w): {Iw_test}")

    Iw_res[kk] = np.max(Iw_test)
    print(f"I[N={N_arr[kk]}] = {Iw_res[kk]}]")

    
# plots
plt.subplots(1,1,figsize=(10, 10))

plt.subplot(1,1,1)
N_axis = np.arange(0,550,0.1)
I_func = lambda N: (1/(var_s+var_ind/N))
plt.plot(N_axis,I_func(N_axis),'g.',markersize=2)
plt.plot(N_arr,Iw_res*dt,'g.',markersize=10)
plt.xlabel('N')
#plt.xlim([0,])
#plt.ylim([0,Nx])
plt.ylabel('fisher information')
plt.title('theoretical: line; experimental dots',loc='left')
sns.despine()


# In[83]:


############################## Non-LIF (connectivity) ##############################

# global constants
SHORT_TEST = True   # True => very small test run; False => full run (6000 trials per stim)
SEED = 5447
np.random.seed(SEED)


# simulation details
#itvl_num = 6000
iter_num = 2#10
trial_num = 6#300
dt=.1#.01
total_T = iter_num * trial_num * 4 * 1000
time=np.arange(0,total_T,dt)
Nt=len(time)
lr = 0.1

# stimulus values (ms^-1)
s1 = 180.0/1000
s2 = 185.0/1000
s0 = 182.5/1000
ss = [s0]

# functions
# ΣJ Σh(t-t')
# g(s)
g = lambda s: (s)
# Σ_kMζ
def noise(N_x):
    zeta_i = np.random.normal(0.0, var_ind, N_x)
    zeta_0 = np.random.normal(0.0, var_s)
    return sigma_ind * zeta_i + sigma_s * zeta_0

# Fisher information (~ -loss)
#fisherI = lambda w:


# LIF network constants
N_arr = [75,125,250,500]
Iw_res = np.zeros(4)

for kk in range(len(N_arr)):
    N = N_arr[kk]                       # number of neurons in network (tunable)
    print(f"N = {N_arr[kk]}")
    exc_frac = 0.8                      # fraction excitatory
    Ne = int(np.round(exc_frac * N))    # number of excitatory neurons
    Ni = N - Ne                         # number of inhibitory neurons
    T = 2*1000                          # observation time window (s->ms)
    taum = 20                           # membrane time constant (ms)
    taue = 2                            # excitatory synaptic time constant (ms)
    taui = 3                            # inhibitory synaptic time constant (ms)
    theta_i = 1                         # threshold 
    V_r = 0                             # reset
    var_ind = 76.5/1000                 # variance of the independent noise (s^-1->ms^-1)
    var_s = 3.5/1000                    # variance of the shared noise (s^-1->ms^-1)
    jee = (6/(Ne-1))                    # E -> E connection strength
    jei = (-9.5/Ni)                     # I -> E connection strength
    jie = (7.6/Ne)                      # E -> I connection strength
    jii = (-11.2/(Ni-1))                # I -> I connection strength
    pee=.2 
    pei=.2 
    pie=.2 
    pii=.2
    sigma_ind = np.sqrt(var_ind)
    sigma_s = np.sqrt(var_s)

    # connection matrices
    Jee = jee*np.random.binomial(1,pee,(Ne,Ne))
    Jei = jei*np.random.binomial(1,pei,(Ne,Ni))
    Jie = jie*np.random.binomial(1,pie,(Ni,Ne))
    Jii = jii*np.random.binomial(1,pii,(Ni,Ni))
    np.fill_diagonal(Jee,0)
    np.fill_diagonal(Jei,0)
    np.fill_diagonal(Jie,0)
    np.fill_diagonal(Jii,0)
    
    # linear estimator, locally optimal
    s_LIF = lambda re,ri: ( Ne/N*(1-(jee+jie))*re+Ni/N*(1-(jei+jii))*ri )

    # initialization of membrane potentials randomly
    Ve=np.random.rand(len(ss),Ne)
    Vi=np.random.rand(len(ss),Ni)
    # initialize all else as zeros
    Se=np.zeros((len(ss),Ne,len(time)))
    Si=np.zeros((len(ss),Ni,len(time)))
    # spike counts
    ne = np.zeros((len(ss),Ne))
    ni = np.zeros((len(ss),Ni))
    # firing rates
    re = np.zeros(len(ss))
    ri = np.zeros(len(ss))
    # estimator result
    slif_train = np.empty((0,len(ss)))
    slif_test = np.empty((0,len(ss)))
    # fisher info
    Iw_train = np.zeros((trial_num*iter_num))
    Iw_test = np.zeros((trial_num*iter_num))
    # optimal weights
    w_opt = -1
    

    # simulation
    for ii in range(iter_num):      # 10 iterations
        # weight(s) init
        #w = np.zeros(trial_num+1)
        #if ii == 0: w[0] = np.random.rand()
        #else: w[0] = w_opt

        print(f"iteration {ii}, training: ")
        ### Training Set ###
        for jj in trange(trial_num): # 300 trials
            for s_i in range(len(ss)):       # for both stimuli (2 separate array)
                for i in range(int(T/dt)):
                    # get the current time, t
                    t = (2*ii*trial_num+jj)*int(T/dt) + i

                    # Euler step for membrane potentials
                    Ve[s_i] = Ve[s_i] + dt * (Jee@Se[s_i,:,t]+Jei@Si[s_i,:,t]+g(ss[s_i])+noise(Ne)) 
                    Vi[s_i] = Vi[s_i] + dt * (Jie@Se[s_i,:,t]+Jii@Si[s_i,:,t]+g(ss[s_i])+noise(Ni))
                    #if i < 10: print(Ve[s_i][0])
                    #if i == 10: print()

                    # Find which excitatory neurons spiked.
                    Inds=np.nonzero(Ve[s_i]>=theta_i)[0]
                    # Reset membrane potentials
                    Ve[s_i][Inds]=V_r    
                    # Store spikes as delta functions
                    Se[s_i,Inds,t+1]=1/dt

                    # Now do the same for inhibitory neurons
                    Inds=np.nonzero(Vi[s_i]>=theta_i)[0] 
                    # Reset membrane potentials
                    Vi[s_i][Inds]=V_r
                    # Store spikes as delta functions
                    Si[s_i,Inds,t+1]=1/dt

                # get spike count, firing rate
                #low=(2*ii*trial_num+jj)*int(T/dt)
                #high=(2*ii*trial_num+jj)*int(T/dt)+int(T/dt)
                Sn_len = (2*ii+1)*trial_num*T
                ne[s_i] = np.sum(Se[s_i],axis=1)
                ni[s_i] = np.sum(Si[s_i],axis=1)
                re[s_i] = np.mean(ne[s_i]/Sn_len)                 # unit: ms^-1
                ri[s_i] = np.mean(ni[s_i]/Sn_len)                 # size: (2,)
                #print(f"re: {re} \t ri: {re}")

            # do prediction with estimator
            slif_train = np.vstack((slif_train, s_LIF(re,ri)))

            # calculate fisher info (~ -loss)
            if (np.var(slif_train[:,0]) != 0): # require nonzero
                Iw_train[ii*trial_num+jj] = 1/np.var(slif_train[:,0])

            # update weights ; considered only if jj >= 2, jj=0: I_w not defined, jj=1: only 1 I_w
            #if (ii>0 or jj>=2):
            #    w[jj+1] = w[jj] + (Iw_train[ii*trial_num+jj] - Iw_train[ii*trial_num+jj-1])/(w[jj]-w[jj-1]) * lr
            #    print((Iw_train[ii*trial_num+jj] - Iw_train[ii*trial_num+jj-1])/(w[jj]-w[jj-1])*lr)
            #else:
            #    w[jj+1] = np.random.rand()
                #print("PRE")

        # choose best weights
        #w_opt = w[np.argmax(Iw_train[ii*trial_num:ii*trial_num+trial_num])]

        # print results
        print(f"estimator prediction: {slif_train}")
        #print(f"weight history: {w}")
        #print(f"optimal weight: {w_opt}")
        print()

        # clear estimator result
        #slif = np.empty((0,len(ss)))


        print(f"iteration {ii}, test: ")
        ### Test Set ###
        for jj in trange(trial_num): # 300 trials
            for s_i in range(len(ss)):       # for both stimuli (2 separate array)
                for i in range(int(T/dt)):
                    # get the current time, t
                    t = ((2*ii+1)*trial_num+jj)*int(T/dt) + i

                    # Euler step for membrane potentials
                    Ve[s_i] = Ve[s_i] + dt * (Jee@Se[s_i,:,t]+Jei@Si[s_i,:,t]+g(ss[s_i])+noise(Ne)) 
                    Vi[s_i] = Vi[s_i] + dt * (Jie@Se[s_i,:,t]+Jii@Si[s_i,:,t]+g(ss[s_i])+noise(Ni))

                    # Find which excitatory neurons spiked.
                    Inds=np.nonzero(Ve[s_i]>=theta_i)[0]
                    # Reset membrane potentials
                    Ve[s_i][Inds]=V_r    
                    # Store spikes as delta functions
                    if (t+1 < iter_num*2*trial_num*int(T/dt)):
                        Se[s_i,Inds,t+1]=1/dt

                    # Now do the same for inhibitory neurons
                    Inds=np.nonzero(Vi[s_i]>=theta_i)[0] 
                    # Reset membrane potentials
                    Vi[s_i][Inds]=V_r
                    # Store spikes as delta functions
                    if (t+1 < iter_num*2*trial_num*int(T/dt)):
                        Si[s_i,Inds,t+1]=1/dt

                # get spike count, firing rate
                #low=(2*ii*trial_num+jj)*int(T/dt)
                #high=(2*ii*trial_num+jj)*int(T/dt)+int(T/dt)
                Sn_len = 2*(ii+1)*trial_num*T
                ne[s_i] = np.sum(Se[s_i],axis=1)
                ni[s_i] = np.sum(Si[s_i],axis=1)
                re[s_i] = np.mean(ne[s_i]/Sn_len)                 # unit: ms^-1
                ri[s_i] = np.mean(ni[s_i]/Sn_len)                 # size: (2,)

            # do prediction with estimator
            slif_test = np.vstack((slif_test, s_LIF(re,ri)))

            # calculate fisher info (~ -loss)
            if (np.var(slif_test[:,0]) != 0): # require nonzero
                Iw_test[ii*trial_num+jj] = 1/np.var(slif_test[:,0])

        # print results
        print(f"estimator prediction: {slif_test}")
        #print(f"optimal weight (from training): {w_opt}")
        print()

    print(f"I_train(w): {Iw_train}")
    print(f"I_test(w): {Iw_test}")

    Iw_res[kk] = np.max(Iw_test)
    print(f"I[N={N_arr[kk]}] = {Iw_res[kk]}]")

    
# plots
plt.subplots(1,1,figsize=(10, 10))

plt.subplot(1,1,1)
N_axis = np.arange(0,550,0.1)
I_func = lambda N: (1/(var_s+var_ind/N))
plt.plot(N_axis,I_func(N_axis),'g.',markersize=2)
plt.plot(N_arr,Iw_res*dt,'g.',markersize=10)
plt.xlabel('N')
#plt.xlim([0,])
#plt.ylim([0,Nx])
plt.ylabel('fisher information')
plt.title('theoretical: line; experimental dots',loc='left')
sns.despine()


# In[ ]:




