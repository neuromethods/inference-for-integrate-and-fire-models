#!/usr/bin/env python2
# -*- coding: utf-8 -*-
''' 
collection of functions for the generation of synthetic data via simulation of 
the generative models as well as for the evaluation of spike train likelihoods 
-- written by Josef Ladenbauer in 2018/2019
'''

import numpy as np
import scipy.integrate
import scipy.optimize
import multiprocessing
from math import exp
from scipy.linalg import solve_banded
import tables
import time
from warnings import warn
import matplotlib.pyplot as plt
from numba import njit

### ===========================================================================   
'''
1st part: functions for data generation, i.e., numerical simulation of stoch. 
differential equations that implement (networks of) exponential/leaky 
integrate-and-fire neurons subject to (external) fluctuating inputs 
'''
  
@njit 
def simulate_EIF_numba(tgrid,V_init,taum,Vth,Vr,VT,DeltaT,Tref,
                       mu_vec,sigma_vec,rand_vec):
    dt = tgrid[1] - tgrid[0]   
    V = V_init*np.ones(len(tgrid))
    Sp_times_dummy = np.zeros(int(len(tgrid)/10)) 
    sp_count = int(0)  

    sqrt_dt = np.sqrt(dt)
    input_dt = dt*mu_vec + sigma_vec*sqrt_dt*rand_vec

    f1 = -dt/taum
    f2 = dt/taum * DeltaT
    if not f2>0:
        DeltaT = 1.0  # to make sure we don't get errors below

    for i_t in range(1,len(tgrid)):
        V[i_t] = V[i_t-1] + f1*V[i_t-1] + f2*np.exp((V[i_t-1]-VT)/DeltaT) + \
                 input_dt[i_t-1]
        # refr. period
        if sp_count>0 and tgrid[i_t]-Sp_times_dummy[sp_count-1]<Tref:
            V[i_t] = Vr

        if V[i_t]>Vth:
            V[i_t] = Vr
            sp_count += 1
            Sp_times_dummy[sp_count-1] = tgrid[i_t]
 

    Sp_times = np.zeros(sp_count)
    if sp_count>0:
        for i in xrange(sp_count):
            Sp_times[i] = Sp_times_dummy[i]

    return V, Sp_times    
    

@njit 
def simulate_EIF_alphapert_numba(tgrid,V_init,taum,Vth,Vr,VT,DeltaT,Tref,mu,sigma,
                                 tperts,J_alpha,tau_alpha,d,rand_vec):
    dt = tgrid[1] - tgrid[0]   
    V = V_init*np.ones(len(tgrid))
    s = np.zeros_like(tgrid)
    x = np.zeros_like(tgrid)
    Sp_times_dummy = np.zeros(int(len(tgrid)/10)) 
    sp_count = int(0)  

    J_alpha *= np.exp(1)/tau_alpha
    tau2 = tau_alpha**2
    sqrt_dt = np.sqrt(dt)
    bg_input_dt = dt*mu + sigma*sqrt_dt*rand_vec
    pert_cnt = 0
    n_perts = len(tperts)
    
    f1 = -dt/taum
    f2 = dt/taum * DeltaT
    if not f2>0:
        DeltaT = 1.0  # to make sure we don't get errors below

    for i_t in range(1,len(tgrid)):
        s[i_t] = s[i_t-1] + dt*x[i_t-1]
        x[i_t] = x[i_t-1] - dt*(2.0/tau_alpha * x[i_t-1] + s[i_t-1]/tau2)
        V[i_t] = V[i_t-1] + f1*V[i_t-1] + f2*np.exp((V[i_t-1]-VT)/DeltaT) + \
                 dt*J_alpha*s[i_t-1] + bg_input_dt[i_t-1]
        # refr. period
        if sp_count>0 and tgrid[i_t]-Sp_times_dummy[sp_count-1]<Tref:
            V[i_t] = Vr

        if V[i_t]>Vth:
            V[i_t] = Vr
            sp_count += 1
            Sp_times_dummy[sp_count-1] = tgrid[i_t]
        
        if pert_cnt<n_perts:
            if tgrid[i_t]>=tperts[pert_cnt]+d and tgrid[i_t]<tperts[pert_cnt]+d+dt:
                x[i_t] += 1.0
                pert_cnt += 1 

    Sp_times = np.zeros(sp_count)
    if sp_count>0:
        for i in xrange(sp_count):
            Sp_times[i] = Sp_times_dummy[i]

    return V, J_alpha*s, Sp_times   


@njit
def simulate_EIF_deltapert_numba(tgrid,V_init,taum,Vth,Vr,VT,DeltaT,Tref,mu,sigma,
                                 tperts,J,d,rand_vec):
    dt = tgrid[1] - tgrid[0]   
    V = V_init*np.ones(len(tgrid))
    Sp_times_dummy = np.zeros(int(len(tgrid)/10)) 
    sp_count = int(0)  

    sqrt_dt = np.sqrt(dt)
    bg_input_dt = dt*mu + sigma*sqrt_dt*rand_vec
    pert_cnt = 0
    n_perts = len(tperts)
    
    f1 = -dt/taum
    f2 = dt/taum * DeltaT
    if not f2>0:
        DeltaT = 1.0  # to make sure we don't get errors below

    for i_t in range(1,len(tgrid)):
        V[i_t] = V[i_t-1] + f1*V[i_t-1] + f2*np.exp((V[i_t-1]-VT)/DeltaT) + \
                 bg_input_dt[i_t-1]
        # refr. period
        if sp_count>0 and tgrid[i_t]-Sp_times_dummy[sp_count-1]<Tref:
            V[i_t] = Vr

        if V[i_t]>Vth:
            V[i_t] = Vr
            sp_count += 1
            Sp_times_dummy[sp_count-1] = tgrid[i_t]
        
        if pert_cnt<n_perts:
            if tgrid[i_t]>=tperts[pert_cnt]+d and tgrid[i_t]<tperts[pert_cnt]+d+dt:
                V[i_t] += J
                pert_cnt += 1 

    Sp_times = np.zeros(sp_count)
    if sp_count>0:
        for i in xrange(sp_count):
            Sp_times[i] = Sp_times_dummy[i]

    return V, Sp_times 


@njit
def simulate_EIF_mudyn_deltapert_numba(tgrid,V_init,taum,Vth,Vr,VT,DeltaT,Tref,
                                       mu_vec,sigma,tperts,J,d,rand_vec):
    dt = tgrid[1] - tgrid[0]   
    V = V_init
    Sp_times_dummy = np.zeros(int(len(tgrid)/10)) 
    sp_count = int(0)  

    sqrt_dt = np.sqrt(dt)
    bg_input_dt = dt*mu_vec + sigma*sqrt_dt*rand_vec
    pert_cnt = int(0)
    n_perts = len(tperts)
    
    f1 = -dt/taum
    f2 = dt/taum * DeltaT
    if not f2>0:
        DeltaT = 1.0  # to make sure we don't get errors below

    for i_t in range(1,len(tgrid)):
        V += f1*V + f2*np.exp((V-VT)/DeltaT) + bg_input_dt[i_t-1]
        # refr. period
        if sp_count>0 and tgrid[i_t]-Sp_times_dummy[sp_count-1]<Tref:
            V = Vr

        if V>Vth:
            V = Vr
            sp_count += 1
            Sp_times_dummy[sp_count-1] = tgrid[i_t]
        
        if pert_cnt<n_perts:
            if tgrid[i_t]>=tperts[pert_cnt]+d and tgrid[i_t]<tperts[pert_cnt]+d+dt:
                V += J
                pert_cnt += 1 

    Sp_times = np.zeros(sp_count)
    if sp_count>0:
        for i in xrange(sp_count):
            Sp_times[i] = Sp_times_dummy[i]

    return Sp_times


@njit 
def simulate_EIF_adapt_numba(tgrid,V_init,taum,Vth,Vr,VT,DeltaT,Tref,mu,sigma,
                             Dw,tau_w,rand_vec):
    dt = tgrid[1] - tgrid[0]   
    V = V_init*np.ones(len(tgrid))
    w = np.zeros_like(tgrid)
    Sp_times_dummy = np.zeros(int(len(tgrid)/10)) 
    sp_count = int(0)
    sqrt_dt = np.sqrt(dt)
    bg_input_dt = dt*mu + sigma*sqrt_dt*rand_vec
    
    f1 = -dt/taum
    f2 = dt/taum * DeltaT
    if not f2>0:
        DeltaT = 1.0  # to make sure we don't get errors below

    for i_t in range(1,len(tgrid)):
        V[i_t] = V[i_t-1] + f1*V[i_t-1] + f2*np.exp((V[i_t-1]-VT)/DeltaT) - \
                 dt*w[i_t-1] + bg_input_dt[i_t-1]
        w[i_t] = w[i_t-1] - dt*w[i_t-1]/tau_w 
        # refr. period
        if sp_count>0 and tgrid[i_t]-Sp_times_dummy[sp_count-1]<Tref:
            V[i_t] = Vr

        if V[i_t]>Vth:
            V[i_t] = Vr
            w[i_t] += Dw  # def.: += Dw
            sp_count += 1
            Sp_times_dummy[sp_count-1] = tgrid[i_t] 

    Sp_times = np.zeros(sp_count)
    if sp_count>0:
        for i in xrange(sp_count):
            Sp_times[i] = Sp_times_dummy[i]

    return V, w, Sp_times  
    

@njit 
def get_w0_values_numba(Sptimes, tau_w):
    w0_vals = np.zeros_like(Sptimes)
    t = Sptimes[0]
    w = 0.0
    dt = 0.1 #ms
    f = dt/tau_w
    n_sp = len(Sptimes)
    sp_cnt = int(0)
    tend = Sptimes[-1]+2*dt
    w_sum = 0;  w_cnt = 0
    while t<tend:
        if sp_cnt<n_sp:
            if t>=Sptimes[sp_cnt] and t<Sptimes[sp_cnt]+dt:
                w += 1.0
                w0_vals[sp_cnt] = w
                sp_cnt += 1
        w -= f*w 
        t += dt
        w_sum += w
        w_cnt += 1
    w_mean = w_sum/w_cnt
    return w0_vals, w_mean


@njit 
def get_w_trace_numba(Sptimes, tau_w, dt):
    tgrid = np.arange(0, Sptimes[-1]+dt, dt)
    w = np.zeros_like(tgrid)
    f = dt/tau_w
    sp_cnt = int(0)
    for i_t in range(1,len(tgrid)):
        if tgrid[i_t]>=Sptimes[sp_cnt] and tgrid[i_t]<Sptimes[sp_cnt]+dt:
            w[i_t] = w[i_t-1] + 1.0
            sp_cnt += 1
        else:
            w[i_t] = w[i_t-1]*(1.0 - f)
    return tgrid, w
        
        

@njit
def simulate_EIF_net_numba(tgrid,V0vals,taum,Vth,Vr,VT,DeltaT,Tref,muvals,
                           sigmavals,Jmat,delay,input_cc,randnvals,randnvals_c):
    dt = tgrid[1] - tgrid[0]
    N = len(Jmat)
    Nrange = range(N)
    V = V0vals.copy()

    #Dmat_ndt = np.round(Dmat/dt)
    delay_dt = np.round(delay/dt)*dt  # constant delay (needs to be a multiple of dt)
    Sp_times_dummy = np.zeros((N,int(len(tgrid)/10))) 
    sp_counts = np.zeros(N)  
    hasspiked_dpassed = np.zeros(N)  # indicates which neuron has spiked delay time ago
    
    sqrt_dt = np.sqrt(dt)
    common_noise_weight = np.sqrt(input_cc)
    indep_noise_weight = np.sqrt(1.0-input_cc)
    
    f1 = -dt/taum
    f2 = dt/taum * DeltaT
    if not f2>0:
        DeltaT = 1.0  # to make sure we don't get errors below

    for i_t in range(1,len(tgrid)):
        ext_input_dt = dt*muvals + sigmavals*sqrt_dt* \
                        ( common_noise_weight*randnvals_c[i_t-1] + \
                          indep_noise_weight*randnvals[:,i_t-1] )

        hasspiked_dpassed *= 0
        for i_N in Nrange:
            if tgrid[i_t]-Sp_times_dummy[i_N,int(sp_counts[i_N])-1] == delay_dt:
                hasspiked_dpassed[i_N] = 1.0

        for i_N in Nrange:
            V[i_N] += f1*V[i_N] + \
                      f2*np.exp((V[i_N]-VT)/DeltaT) + \
                      ext_input_dt[i_N]
            V[i_N] += np.dot(Jmat[i_N,:], hasspiked_dpassed)
            
            # refr. period
            if sp_counts[i_N]>0 and \
                tgrid[i_t]-Sp_times_dummy[i_N,int(sp_counts[i_N])-1]<Tref:
                V[i_N] = Vr
    
            if V[i_N]>Vth:
                V[i_N] = Vr
                sp_counts[i_N] += 1
                Sp_times_dummy[i_N,int(sp_counts[i_N])-1] = tgrid[i_t]

#    Sp_times_dict = {}
#    for i_N in Nrange:
#        if sp_counts[i_N]>0:
#            Sp_times_dict[i_N] = Sp_times_dummy[i_N,Sp_times_dummy[i_N,:]>0]
#        else:
#            Sp_times_dict[i_N] = np.array([])
         
#    Sp_times_list = list(Nrange)
#    for i_N in Nrange:
#        if sp_counts[i_N]>0:
#            Spt_tmp = Sp_times_dummy[i_N,:]
#            Sp_times_list[i_N] = Spt_tmp[Spt_tmp>0]
#        else:
#            Sp_times_list[i_N] = np.array([])
               
#    return Sp_times_dict, V 
    return Sp_times_dummy, sp_counts, V
    
    
### ===========================================================================
    
'''
2nd part: (core) functions for the calculation of (log-)likelihoods
'''

@njit
def EIF_pertISIdensityhat_numba(V_vec, kr, taum, Vr, VT, DeltaT, Tref,
                                mu, sigma, w_vec, t_pert_vec): 
    # calculates the ISI density p_ISI in the frequency domain: the unperturbed
    # one and the corrections due to perturbations, where here we consider as 
    # perturbations delta kicks at various times given by t_pert_vec (separately)
    # see SI section S2.2 of the paper
    epsilon = 1e-4
    dV = V_vec[1]-V_vec[0]
    sig2term = 2.0/sigma**2
    n = len(V_vec)
    m = len(t_pert_vec)
    krange = range(n-1, 0, -1)
    mrange = range(m)
    wrange = range(len(w_vec))
    w_vec_pos = w_vec  # if w_vec starts with 0 this will be accounted for below 
    pISIhat0_vec = 1j*np.ones(len(w_vec)) 
    pISIhat1_mat = 1j*np.ones((m,len(w_vec)))
    phat_w0 = 1j*np.ones(n)
    phat_wk = 1j*np.ones(n)  
    pah_vec = 1j*np.zeros(n)
    sumsum = 1j*np.zeros((m,n))
    
    if DeltaT>0:
        Psi = DeltaT*np.exp((V_vec-VT)/DeltaT)
    else:
        Psi = 0.0*V_vec
    F = sig2term*( ( V_vec-Psi )/taum - mu ) 
    A = np.exp(dV*F)
    F_dummy = F.copy()
    F_dummy[F_dummy==0.0] = 1.0
    B = (A - 1.0)/F_dummy * sig2term
    sig2term_dV = dV * sig2term
    
    # first the unperturbed ISI density
    # start with zero frequency separately
    if w_vec[0]==0:
        qah = 1.0 + 0.0*1j;  pah = 0.0*1j;  qbh = 0.0*1j;  pbh = 0.0*1j;
        phat_w0[-1] = pbh  # adjusted below  
        for k in krange:
            if k>kr:    
                if not F[k]==0.0:    
                    pah = pah * A[k] + B[k]
                else:
                    pah = pah * A[k] + sig2term_dV
            else:
                if not F[k]==0.0:    
                    pah = pah * A[k] + B[k]
                    pbh = pbh * A[k] - B[k]
                else:
                    pah = pah * A[k] + sig2term_dV
                    pbh = pbh * A[k] - sig2term_dV
            phat_w0[k-1] = pbh
            pah_vec[k-1] = pah
        pISIhat0_vec[0] = 1.0
        phat_w0 += pISIhat0_vec[0]*pah_vec
        w_vec_pos = w_vec[1:]
    # continue with positive frequencies
    for iw in range(len(w_vec_pos)):
        qah = 1.0 + 0.0*1j;  pah = 0.0*1j;  qbh = 0.0*1j;  pbh = 0.0*1j;
        phat_wk[-1] = pbh  # adjusted below
        fw = dV*1j*w_vec_pos[iw]
        refterm = np.exp(-1j*w_vec_pos[iw]*Tref)         
        for k in krange:
            if not k==kr+1:
                qbh_new = qbh + fw*pbh
            else:
                qbh_new = qbh + fw*pbh - refterm
            if not F[k]==0.0:    
                pah_new = pah * A[k] + B[k] * qah
                pbh_new = pbh * A[k] + B[k] * qbh
            else:
                pah_new = pah * A[k] + sig2term_dV * qah
                pbh_new = pbh * A[k] + sig2term_dV * qbh
            qah += fw*pah
            qbh = qbh_new;  pbh = pbh_new;  pah = pah_new;
            phat_wk[k-1] = pbh
            pah_vec[k-1] = pah
  
        pISIhat0_vec[iw+1] = -qbh/qah
        phat_wk += pISIhat0_vec[iw+1]*pah_vec
        for it in mrange:
            sumsum[it,:] += np.exp(1j*w_vec_pos[iw]*t_pert_vec[it]) * phat_wk

    # next the corrections due to perturbation
    for it in mrange:
        p0_tpert = (w_vec[1]-w_vec[0]) * (phat_w0 + sumsum[it,:] + \
                   np.conj(sumsum[it,:])) / (2*np.pi)
        # the following 5 lines implement a correction of the membrane voltage 
        # density at the perturbation time, which may be necessary because of
        # improper parameter values for the numerical scheme (allowing for 
        # increased efficiency and robustness of the method)
        dummy = p0_tpert.real
        k = n-1
        while dummy[k]>=0 and k>0:
            k -= 1
        p0_tpert[:k+1] = 0.0  
        prop_not_spiked = dV*np.sum(p0_tpert[k:])  # can be interpreted as 
        # proportion of trials in which the neuron has not yet spiked 
        if prop_not_spiked.real>epsilon:
            for iw in wrange:
                qah = 1.0 + 0.0*1j;  pah = 0.0*1j;  qbh = 0.0*1j;  pbh = 0.0*1j;
                fw = dV*1j*w_vec[iw] 
                inhom = np.exp(-t_pert_vec[it]*1j*w_vec[iw]) * p0_tpert 
                for k in krange:
                    qbh_new = qbh + fw*pbh
                    if not F[k]==0.0:    
                        pah_new = pah * A[k] + B[k] * qah
                        pbh_new = pbh * A[k] + B[k] * (qbh - inhom[k])
                    else:
                        pah_new = pah * A[k] + sig2term_dV * qah
                        pbh_new = pbh * A[k] + sig2term_dV * (qbh - inhom[k])
                    qah += fw*pah
                    qbh = qbh_new;  pbh = pbh_new;  pah = pah_new;   
                pISIhat1_mat[it,iw] = -qbh/qah
        else:
            pISIhat1_mat[it,:] = 0.0

    return pISIhat0_vec, pISIhat1_mat

    
@njit  
def EIF_ISIdensityhat_numba(V_vec, kr, taum, Vr, VT, DeltaT, Tref,
                            mu, sigma, w_vec):
    # calculates the unperturbed ISI density in the frequency domain,
    # see SI section S2.2 of the paper
    dV = V_vec[1]-V_vec[0]
    sig2term = 2.0/sigma**2
    n = len(V_vec)
    m = len(w_vec)
    mrange = range(m)
    krange = range(n-1, 0, -1)
    fpth_vec = 1j*np.ones(m) 
    if DeltaT>0:
        Psi = DeltaT*np.exp((V_vec-VT)/DeltaT)
    else:
        Psi = 0.0*V_vec
    F = sig2term*( ( V_vec-Psi )/taum - mu ) 
    A = np.exp(dV*F)
    F_dummy = F.copy()
    F_dummy[F_dummy==0.0] = 1.0
    B = (A - 1.0)/F_dummy * sig2term
    sig2term_dV = dV * sig2term
    for iw in mrange:
        qah = 1.0 + 0.0*1j;  pah = 0.0*1j;  qbh = 0.0*1j;  pbh = 0.0*1j;
        fw = dV*1j*w_vec[iw]
        refterm = np.exp(-1j*w_vec[iw]*Tref)  
        for k in krange:
            if not k==kr+1:
                qbh_new = qbh + fw*pbh
            else:
                qbh_new = qbh + fw*pbh - refterm
            if not F[k]==0.0:    
                pah_new = pah * A[k] + B[k] * qah
                pbh_new = pbh * A[k] + B[k] * qbh
            else:
                pah_new = pah * A[k] + sig2term_dV * qah
                pbh_new = pbh * A[k] + sig2term_dV * qbh
            qah += fw*pah
            qbh = qbh_new;  pbh = pbh_new;  pah = pah_new;   
        fpth_vec[iw] = -qbh/qah
    return fpth_vec 



def EIF_ISIdensity(V_vec, kr, taum, Vr, VT, DeltaT, Tref, mu, sigma, f_vec):
    kr = np.argmin(np.abs(V_vec-Vr))  #re-calc to make sure kr corresponds to Vr
    w_vec = 2*np.pi*f_vec
    df = f_vec[1] - f_vec[0]
    n = 2*len(f_vec)-1 #assuming f_vec starts with 0
    dt = 1.0/(df*n)  
    pISI_times = np.arange(0,n)*dt     
    pISIhat = EIF_ISIdensityhat_numba(V_vec, kr, taum, Vr, VT, DeltaT, Tref,
                                      mu, sigma, w_vec) 
    pISI_vals = np.fft.irfft(pISIhat,n)/dt  
    return pISI_times, pISI_vals      
   


def find_mu_init(ISImean, args=()):
    args = args + (ISImean,)
    bnds = (-3.75, 1.75)
    sol =  scipy.optimize.minimize_scalar(find_mu_init_errorfun, bounds=bnds, 
                                          args=args, method='bounded', 
                                          options={'xatol': 1e-2})
    mu_init = sol.x
    return mu_init
    

def find_mu_init_errorfun(mu, *args): 
    params, sigma, ISImean = args
    p_ss, r_ss, q_ss = EIF_steady_state_numba(params['V_vals'], params['V_r_idx'],
                                    params['tau_m'], params['V_r'], params['V_T'],
                                    params['Delta_T'], mu, sigma)
    r_ss_ref = r_ss/(1+r_ss*params['T_ref'])
    ISImean_cand = 1.0/r_ss_ref
    
    return np.abs(ISImean_cand - ISImean)
   
    
    
def spiketrain_likel_musig(p, *args):
    args = args + (p[0], p[1])
    if p[1]<=0.3:  #avoid sigma that is too small for Fokker-Planck method
        loglval = 1e10
    else:
        _, loglval, _, _ = calc_spiketrain_likelihood(args)
        if np.isnan(loglval):
            loglval = 1e10  # optimization must be initialized within "non-nan region" 
            print('WARNING: optimization method stepped into nan-region')
    error = np.abs(loglval) 
    return error
    


def spiketrain_likel_musigtaum(p, *args):
    if p[2]<=0 or p[2]>50 or p[1]<=0.3:  #avoid neg. taum, very large one, or 
        loglval = 1e10                   #sigma that is too small
    else:
        args[0]['tau_m'] = p[2]
        args = args + (p[0], p[1])
        _, loglval, _, _ = calc_spiketrain_likelihood(args)
        if np.isnan(loglval):
            loglval = 1e10  # optimization must be initialized within "non-nan region" 
            print('WARNING: optimization method stepped into nan-region')
    error = np.abs(loglval) 
    return error



def spiketrain_likel_alpha(p, *args):
    # run LNexp model (w/o adaptation) for given parameters over specified 
    # duration and use the resulting spike rate time series to extract the
    # loglikelihood for the given spike train
    verbatim = False
    mu, tperts, Sptimes, lastSpidx, d, tgrid, \
    mu_vals, r_ss_array, tau_mu_array, ISImin, ISImax = args
    J, tau = p          
    
    if tau>=0.1: 
        muf_init = mu
        s_init = 0.0
        x_init = 0.0
        # we have one (possibly) very long spike time series; 
        # for computational reasons we divide it into several parts
        dtsim = tgrid[1]-tgrid[0]
        T_break = 10000.0  #ms (value not optimized)
        N_parts = int(np.floor_divide(Sptimes[-1]-Sptimes[0],T_break) + 1)
        loglval = 0.0
        for k in range(N_parts):
            # generate time points
            if k!=N_parts-1:
                tgrid = np.arange(k*T_break,(k+1)*T_break+dtsim/2,dtsim)
            else: # last part
                tgrid = np.arange(k*T_break,Sptimes[-1]+dtsim/2,dtsim)
            tperts_part = tperts[(tperts>=tgrid[0]) & (tperts<=tgrid[-1])]
            rate, signal, muf_last, s_last, x_last = \
                sim_LNexp_sigfix_mupert(tgrid, mu, J, tau, d, tperts_part,
                                        mu_vals, r_ss_array, tau_mu_array,
                                        muf_init, s_init, x_init)
            muf_init, s_init, x_init = muf_last, s_last, x_last
            ratecumsum = np.cumsum(rate)  
            Sptimes_tmp = Sptimes[(Sptimes>=tgrid[0]) & (Sptimes<=tgrid[-1])]    
            lval = loglikelihood_Poisson(tgrid, rate, ratecumsum, Sptimes_tmp, 
                                         lastSpidx, ISImin, ISImax)
            loglval += lval
            
        if np.isnan(loglval):
            print('')
            print('problem at J, tau = ')
            print(J, tau)     
    else:
        loglval = 1e10

    if verbatim:    
        print('J, tau, logl = ')
        print(np.round(p[0],2), np.round(p[1],2), np.round(loglval,2)) 
    return np.abs(loglval)



#def spiketrain_likel_mu_adapt_fast(p, *args):
#    verbatim = True
#    tau_w = p[0]
#    mu, sigma, Dw_vals, mupert_vals, Sptimes, lastSpidx, params, \
#    ISImin, ISImax, lastrun = args
#    # mu, sigma from previous estimation w/o adaptation; keep sigma constant,
#    # but adjust mu0 here such that mean of mu0 - w(t) = mu
#    # mupert_vals and Dw_vals may be coarse for application to real neurons
#        
#    sigma_array = sigma * np.ones_like(params['t_grid'])
#    pISI_vals = np.zeros((len(mupert_vals), len(sigma_array)))
#    sp_times = np.array([0])
#    dummy = np.exp(-params['t_grid']/tau_w)    
#    N = len(Sptimes)    
#    loglike_vals = np.zeros_like(Dw_vals)
#    # get w_0 values (one for each spike time) based on Sptimes and tau_w:
#    w0_vals, w0_mean = get_w0_values_numba(Sptimes, tau_w)
#    for iw, w in enumerate(Dw_vals):   
#        lval = 0.0   
#        i_ts = 0
#        w0_lookup = w*w0_vals
#        mu0 = mu + w*w0_mean
#        print('Dw={} mu={}'.format(w, mu0))  #TEMP!
#        for im, mu1 in enumerate(mupert_vals):
#            # effective mean input
#            mu_array = mu0 - mu1*dummy
#            fvm_dict = pISI_fvm_sg(mu_array, sigma_array, params, fpt=True, rt=sp_times)
#            pISI_vals[im,:] = fvm_dict['pISI_vals']
#        valid = True
#        while i_ts < N-1 and valid:
#            i_ts += 1
#            if not i_ts-1 in lastSpidx:
#                ISI = Sptimes[i_ts] - Sptimes[i_ts-1]
#                if ISI>=ISImin and ISI<=ISImax:
#                    idx, weight = interp1d_getweight(w0_lookup[i_ts-1], mupert_vals)
#                    pISI_lookup = pISI_vals[idx,:]*(1.0-weight) + \
#                                  pISI_vals[idx+1,:]*weight
#                    pISI_lookup_val = interpol(ISI, params['t_grid'], pISI_lookup)
#                    if pISI_lookup_val<=0:
#                        valid = False
#                        # neg. pISI value encoutered; this may be resolved by 
#                        # decreasing discretizations steps for pISI_fvm_sg
#                        # print 'w =', w, ' ==> log-likelihood = nan' 
#                        lval = np.nan
#                    else:
#                        lval += np.log( pISI_lookup_val )
#        loglike_vals[iw] = lval  
#    
#    iw = np.nanargmax(loglike_vals)
#    # loglikelihood maximized along Delta_w for given tau_w:
#    lval = np.nanmax(loglike_vals)  
#    
#    if verbatim:
#        print('')
#        print('Dw, tauw, logl =')
#        print(Dw_vals[iw], tau_w, lval)
#    
#    if lastrun:
#        mu0 = mu + Dw_vals[iw]*w0_mean
#        return Dw_vals[iw], tau_w, mu0, lval
#    else:
#        return np.abs(lval)
    
    
    
def spiketrain_likel_musig_adapt(p, *args):
    verbatim = True
    tau_w, mu, sigma = p
    mu_no_adapt, sigma_no_adapt, Dw_vals, mupert_vals, Sptimes, lastSpidx, params, \
    ISImin, ISImax, lastrun = args

    # get w_0 values (one for each spike time) based on Sptimes and tau_w:
    w0_vals, w0_mean = get_w0_values_numba(Sptimes, tau_w)   
    if mu<=mu_no_adapt or tau_w<1.5*params['tau_m'] or tau_w>1000 or \
        sigma<=sigma_no_adapt or sigma>2*sigma_no_adapt:
        return 1e10
    else:
        # input st. dev.
        sigma_array = sigma * np.ones_like(params['t_grid'])
        pISI_vals = np.zeros((len(mupert_vals), len(sigma_array)))
        sp_times = np.array([0])
        dummy = np.exp(-params['t_grid']/tau_w)
        for im, mu1 in enumerate(mupert_vals):
            # effective mean input
            mu_array = mu - mu1*dummy
            fvm_dict = pISI_fvm_sg(mu_array, sigma_array, params, fpt=True, 
                                   rt=sp_times)
            pISI_vals[im,:] = fvm_dict['pISI_vals']
        
        N = len(Sptimes)    
        loglike_vals = np.zeros_like(Dw_vals)
        # get w_0 values (one for each spike time) based on Sptimes and tau_w:
        w0_vals, _ = get_w0_values_numba(Sptimes, tau_w)
        
        for iw, w in enumerate(Dw_vals):   
            lval = 0.0   
            i_ts = 0
            w0_lookup = w*w0_vals
            valid = True
            while i_ts < N-1 and valid:
                i_ts += 1
                if not i_ts-1 in lastSpidx:
                    ISI = Sptimes[i_ts] - Sptimes[i_ts-1]
                    if ISI>=ISImin and ISI<=ISImax:
                        idx, weight = interp1d_getweight(w0_lookup[i_ts-1], mupert_vals)
                        pISI_lookup = pISI_vals[idx,:]*(1.0-weight) + \
                                      pISI_vals[idx+1,:]*weight
                        pISI_lookup_val = interpol(ISI, params['t_grid'], pISI_lookup)
                        if pISI_lookup_val<=0:
                            valid = False
                            # neg. pISI value encoutered; this may be resolved by 
                            # decreasing discretizations steps for pISI_fvm_sg
                            # print 'w =', w, ' ==> log-likelihood = nan' 
                            lval = np.nan
                        else:
                            lval += np.log( pISI_lookup_val )
            loglike_vals[iw] = lval  
        
        iw = np.nanargmax(loglike_vals)
        # loglikelihood maximized along Delta_w for given tau_w:
        lval = np.nanmax(loglike_vals)  
        
        if verbatim:
            print('')
            print('Dw, tauw, mu, sigma =')
            print(Dw_vals[iw], tau_w, mu, sigma)
        
        if lastrun:
            return Dw_vals[iw], tau_w, mu, sigma, lval
        else:
            return np.abs(lval)



def spiketrain_likel_adapt(p, *args):
    verbatim = False
    tau_w = p[0]
    mu, sigma, Dw_vals, mupert_vals, Sptimes, lastSpidx, params, \
    ISImin, ISImax, lastrun = args
        
    # input st. dev.
    sigma_array = sigma * np.ones_like(params['t_grid'])
    pISI_vals = np.zeros((len(mupert_vals), len(sigma_array)))
    sp_times = np.array([0])
    for im, mu1 in enumerate(mupert_vals):
        # effective mean input
        mu_array = mu - mu1*np.exp(-params['t_grid']/tau_w)

        fvm_dict = pISI_fvm_sg(mu_array, sigma_array, params, fpt=True, 
                               rt=sp_times)
        pISI_vals[im,:] = fvm_dict['pISI_vals']
    
    N = len(Sptimes)    
    loglike_vals = np.zeros_like(Dw_vals)
    # get w_0 values (one for each spike time) based on Sptimes and tau_w:
    w0_vals, _ = get_w0_values_numba(Sptimes, tau_w)
    
    for iw, w in enumerate(Dw_vals):   
        lval = 0.0   
        i_ts = 0
        w0_lookup = w*w0_vals
        valid = True
        while i_ts < N-1 and valid:
            i_ts += 1
            if not i_ts-1 in lastSpidx:
                ISI = Sptimes[i_ts] - Sptimes[i_ts-1]
                if ISI>=ISImin and ISI<=ISImax:
                    idx, weight = interp1d_getweight(w0_lookup[i_ts-1], mupert_vals)
                    pISI_lookup = pISI_vals[idx,:]*(1.0-weight) + \
                                  pISI_vals[idx+1,:]*weight
                    pISI_lookup_val = interpol(ISI, params['t_grid'], pISI_lookup)
                    if pISI_lookup_val<=0:
                        valid = False
                        # neg. pISI value encoutered; this may be resolved by 
                        # decreasing discretizations steps for pISI_fvm_sg
                        # print 'w =', w, ' ==> log-likelihood = nan' 
                        lval = np.nan
                    else:
                        lval += np.log( pISI_lookup_val )
        loglike_vals[iw] = lval  
    
    iw = np.nanargmax(loglike_vals)
    # loglikelihood maximized along Delta_w for given tau_w:
    lval = np.nanmax(loglike_vals)  
    
    if verbatim:
        print('')
        print('w, tauw, logl =')
        print(Dw_vals[iw], tau_w, lval)
    
    if lastrun:
        return Dw_vals[iw], tau_w, lval
    else:
        return np.abs(lval)
 


def Jdij_estim_wrapper(args):  
    # this is called for each postsynaptic neuron and
    # estimates connections for all presynaptic neurons
    i_N, N, args_fixed = args
    Spt_dict, sigma_init, N_tpert, params = args_fixed
    
    ISImin = params['ISI_min']
    N_pseudo = params['N_pseudo']
    jitter_width = np.diff(params['pseudo_jitter_bnds'])[0]
    J_estim_row = np.zeros(N)
    J_estim_bias_row = np.zeros(N)  # estim. bias 
    J_estim_z_row = np.zeros(N)  # z-score
    d_estim_row = np.zeros(N)
    logl_coupled_row = np.zeros(N) * np.nan
        
    # omit spikes that follow a previous one too shortly (ISImin)
    remain_idx = Spt_dict[i_N]>0  # all true
    for i in range(len(Spt_dict[i_N])-1):
        if Spt_dict[i_N][i+1]-Spt_dict[i_N][i]<ISImin:
            Spt_dict[i_N][i+1] = Spt_dict[i_N][i]
            remain_idx[i+1] = False
    Spt_dict[i_N] = Spt_dict[i_N][remain_idx]
    # 1) estimation of baseline input parameters (mu, sigma) for the (postsyn.)  
    # neuron (without using the spike times of other neurons); tau_m can be 
    # fixed to a reasonable value (e.g., in [5,30] ms), but it may also be 
    # included in the estimation, see baseline_input_inference.py and adjust
    # the lines below accordingly
    start = time.time()
    ISIs = np.diff(Spt_dict[i_N])
    ISImax = np.max(ISIs) + 3.0
    ISImean = np.mean(ISIs)  
    print('mean ISI of neuron {n_no} = {ISImean}'.format(n_no=i_N+1, 
          ISImean=np.round(ISImean,2)))

    mu_init = find_mu_init(ISImean, args=(params,sigma_init))
    init_vals = np.array([mu_init, sigma_init])
    sol = scipy.optimize.minimize(spiketrain_likel_musig, init_vals, 
                                  args=(params, ISIs, ISImean, ISImax), 
                                  method='nelder-mead', 
                                  options={'xatol':0.025, 'fatol':0.01})
    mu_estim, sigma_estim = sol.x  
    print('neuron {n_no}: mu_estim = {mu}, sigma_estim = {sig}'.format(
           n_no=i_N+1, mu=np.round(mu_estim,2), sig=np.round(sigma_estim,2)))
    
    args = (params, ISIs, ISImean, ISImax, mu_estim, sigma_estim)
    lval, logl_uncoupled, pISI_times, pISI0 = calc_spiketrain_likelihood(args)
    
    dt = pISI_times[1]-pISI_times[0]
    ISI0mean = dt*np.sum(pISI0*pISI_times)
           
    # 2) calculate ISI density corrections (pISI1) for generic perturbation 
    # times (t_pert): N_tpert time values in [0, t_pert_max], where 
    # t_pert_max > ISI0mean and pISI0[t_pert_max] == epsilon (e.g. 1e-3)  
    # then (further below) use lookups: t_pert that is closest to the observed 
    # arrival time
    epsilon = 1e-3  #1e-4 to 1e-3
    dummy = pISI0.copy()
    dummy[pISI_times<ISI0mean] = 0
    idx = np.argmin(np.abs(dummy - epsilon))
    t_pert_max = pISI_times[idx]
    
    t_pert_vals = np.linspace(0.0, t_pert_max, num=N_tpert)
    d_tp = t_pert_vals[1]-t_pert_vals[0]
    
    if params['pISI_method'] == 'fvm':
        params['t_grid'] = pISI_times
        pISI0, pISI1 = pISI0pISI1_deltaperts_fvm_sg(mu_estim, t_pert_vals, 
                                                    sigma_estim, params)
    elif params['pISI_method'] == 'fourier':
        pISI1 = np.zeros((len(t_pert_vals), len(pISI_times)))
        n = 2*len(params['freq_vals'])-1  #assuming freq_vals start with 0 
        pISI_times_tmp = np.arange(0,n)*dt
        idcs = pISI_times_tmp<=ISImax
        w_vec = 2*np.pi*params['freq_vals']
        pISIhat0, pISIhat1 = EIF_pertISIdensityhat_numba(params['V_vals'], 
                                 params['V_r_idx'], params['tau_m'], params['V_r'], 
                                 params['V_T'], params['Delta_T'], params['T_ref'],
                                 mu_estim, sigma_estim, w_vec, t_pert_vals[1:])
        # for small perturbation times interpolate between t_pert=0 and smallest 
        # t_pert that yields a nonzero output from EIF_FPTpertdensityhat; for
        # t_pert=0 use derivative of pISIhat0 w.r.t. Vr
        dVr = 0.2  #must be a multiple of d_V
        Vr_tp0 = params['V_r']+dVr
        kr_tp0 = np.argmin(np.abs(params['V_vals']-Vr_tp0))  # reset index value
        pISIhat0_dVr = EIF_ISIdensityhat_numba(params['V_vals'], kr_tp0, params['tau_m'], 
                                        Vr_tp0, params['V_T'], params['Delta_T'], 
                                        params['T_ref'], mu_estim, sigma_estim, w_vec)
        pISI0dVr = np.fft.irfft(pISIhat0_dVr,n)/dt
        pISI1[0,:] = (pISI0dVr[idcs] - pISI0)/dVr
        for i_tp in range(1,len(t_pert_vals)):                           
            if any(np.real(pISIhat1[i_tp-1,:]) != 0):
                pISI1[i_tp,:] = np.fft.irfft(pISIhat1[i_tp-1,:],n)[idcs]/dt
            else: #take previous one (should only occur for very small t_perts)
                pISI1[i_tp,:] = pISI1[i_tp-1,:]

        #pISI_inds_eff = pISI_times<=ISImax  
        #pISI_times = pISI_times[pISI_inds_eff]
        #pISI0 = pISI0[pISI_inds_eff]
        #pISI1 = pISI1[:,pISI_inds_eff]
        # everything stored now  
        del pISIhat1  # free some space
        
    # there is a transient peak in pISI1 following t_pert, which causes problems 
    # for the estimation of inhibitory connections (that are not very weak),
    # the following "truncation" over a short period alleviates this difficulty 
    t_dur = 0.1  # ms
    for itp, tp in enumerate(t_pert_vals):
        idcs = (pISI_times>tp-dt) & (pISI_times<=tp+t_dur)
        idx_tmp = (pISI_times>tp+t_dur)
        dummy = pISI1[itp,idx_tmp]
        pISI1[itp,idcs] = dummy[0]
        
    print('precalculations for neuron {n_no} took {dur}s'.format(n_no=i_N+1, 
          dur=np.round(time.time() - start,2)))

    # 3) next, for each ISI of the (postsyn.) neuron iN, determine the appropriate 
    # perturbation times for each presyn. neuron j_N and then determine the 
    # coupling strength J_ij that optimizes the (partial) likelihood for neuron
    # iN, using (looking-up) the appropriate pISI1 corrections
    ISIrange = range(len(ISIs))
    for j_N in range(N):
        if j_N!=i_N and len(Spt_dict[j_N])>0:  # otherwise J_ij = 0, (no autapses, 
                                               # but this can be easily changed)        
            # for each pre spike determine the corresponding post ISI and save the 
            # t_pert lookup index considering delay d for pISI1 at that ISI (looping  
            # over d_grid values to find optimal delay)
            # estimation on actual data:
            Jij_estim_tmp = np.zeros_like(params['d_grid'])
            loglike_tmp = np.zeros_like(params['d_grid'])    
            for i_d, delay in enumerate(params['d_grid']):
                # generate list of i_tp indices (can be >1 per ISI) 
                # which indicate which (discrete) t_pert row(s) in pISI1 to use per ISI            
                tp_idx_list = [[] for l in ISIrange]
                for ind, tspj in enumerate(Spt_dict[j_N]):
                    tspj += delay  # mapping from presyn. spike time to postsyn. 
                                   # perturbation time
                    if tspj>Spt_dict[i_N][0] and tspj<Spt_dict[i_N][-1]:
                    # find the closest previous spike time of neuron i_N which 
                    # determines the ISI under consideration, evaluate the 
                    # duration and save the corresponding i_tp index
                        idx_prev_tspi = np.argmin( np.abs(
                                        Spt_dict[i_N][Spt_dict[i_N]<tspj]-tspj) )
                        t_pert_local = tspj-Spt_dict[i_N][idx_prev_tspi]
                        if t_pert_local<t_pert_max+d_tp:
                            tp_idx_list[idx_prev_tspi] += \
                                       [np.argmin(np.abs(t_pert_vals-t_pert_local))]
                  
                Jij_estim_tmp2 = np.zeros_like(params['J_init'])
                loglike_tmp2 = np.zeros_like(params['J_init'])
                for i_J, J_init in enumerate(params['J_init']):  
                    args = (ISIs, ISImin, ISImax, tp_idx_list, 
                            pISI_times, pISI0, pISI1, params['J_bnds'])
                    sol = scipy.optimize.minimize(spiketrain_likel_Jij, J_init, args=args, 
                                method='nelder-mead', options={'xatol':0.05, 'fatol':1e-4})
                    Jij_estim_tmp2[i_J] = sol.x[0]
                    loglike_tmp2[i_J] = -sol.fun
                J_idx = np.argmax(loglike_tmp2)
                Jij_estim_tmp[i_d] = Jij_estim_tmp2[J_idx]
                loglike_tmp[i_d] = loglike_tmp2[J_idx]
                 
            d_idx = np.argmax(loglike_tmp)    
            J_estim_row[j_N] = Jij_estim_tmp[d_idx] 
            d_estim_row[j_N] = params['d_grid'][d_idx]
            logl_coupled_row[j_N] = loglike_tmp[d_idx]
            print('connection {}<-{} estimated'.format(i_N+1, j_N+1))
            
            if N_pseudo>0:
                # estimation on pseudo data:    
                Jij_estim_pseudo = np.zeros(N_pseudo)
                seeds = range(11,11+N_pseudo)
                for n in range(N_pseudo): 
                    np.random.seed(seeds[n])
                    jitter_vals = params['pseudo_jitter_bnds'][0] + \
                                  jitter_width*np.random.rand(len(Spt_dict[j_N]))
                    prespktimes = Spt_dict[j_N] + jitter_vals
                    
                    tp_idx_list = [[] for l in ISIrange]
                    for ind, tspj in enumerate(prespktimes):
                        if tspj>Spt_dict[i_N][0] and tspj<Spt_dict[i_N][-1]:
                        # find the closest previous spike time of neuron i_N which 
                        # determines the ISI under consideration, evaluate the 
                        # duration and save the corresponding i_tp index
                            idx_prev_tspi = np.argmin( np.abs(
                                            Spt_dict[i_N][Spt_dict[i_N]<tspj]-tspj) )
                            t_pert_local = tspj-Spt_dict[i_N][idx_prev_tspi]
                            if t_pert_local<t_pert_max+d_tp:
                                tp_idx_list[idx_prev_tspi] += \
                                           [np.argmin(np.abs(t_pert_vals-t_pert_local))]
                    
                    J_init = np.random.rand()-0.5
                    args = (ISIs, ISImin, ISImax, tp_idx_list, 
                            pISI_times, pISI0, pISI1, params['J_bnds'])
                    sol = scipy.optimize.minimize(spiketrain_likel_Jij, J_init, args=args, 
                                method='nelder-mead', options={'xatol':0.05, 'fatol':1e-4})
                    Jij_estim_pseudo[n] = sol.x[0]
                    
                J_estim_bias_row[j_N] = np.nanmean(Jij_estim_pseudo)
                J_estim_z_row[j_N] = (J_estim_row[j_N] - J_estim_bias_row[j_N]) / \
                                      np.nanstd(Jij_estim_pseudo)
                        
    return i_N, mu_estim, sigma_estim, logl_uncoupled, J_estim_row, \
           J_estim_bias_row, J_estim_z_row, d_estim_row, logl_coupled_row



def spiketrain_likel_Jij(J_ij, *args):
    ISIs, ISImin, ISImax, tp_idx_list, pISI_times, \
    pISI0_vals, pISI1_vals, J_bnds = args  
    # note that pISI1_vals is a matrix
    if J_ij>J_bnds[1] or J_ij<J_bnds[0]:
        loglval = 1e20
    else:
        loglval = 0.0
        valid = True
        k = 0
        N = len(ISIs)
        while k<N and valid:
            if ISIs[k] >= ISImin and ISIs[k] <= ISImax:
                # construct appropriate pISI
                pISI_vals = pISI0_vals.copy()
                for l in range(len(tp_idx_list[k])):
                    idx = tp_idx_list[k][l]
                    pISI_vals += J_ij*pISI1_vals[idx,:]
                
                pISI_lookup_val = interpol(ISIs[k], pISI_times, pISI_vals)
                if pISI_lookup_val>0:
                    loglval += np.log( pISI_lookup_val )
                else:
                    valid = False
                    #print 'neg. pISI value encoutered for J_ij =', J_ij
                    #print '==> log-likelihood = nan' 
                    loglval = 1e20 #np.nan                        
                    # pISI_vals should not contain neg. values
                    # note that omitting only affected ISIs impairs the comparability 
                    # between overall loglikel. values
            k += 1
    error = np.abs(loglval)
    return error
 

@njit
def sim_LNexp_sigfix_mupert(tgrid, mu, J, tau, d, tperts, mu_vals, 
                            r_ss_array, tau_mu_array, muf_init, s_init, x_init):
    tau2 = tau**2
    J *= np.exp(1)/tau  # normalization such that peak=J 
    dt = tgrid[1] - tgrid[0]
    L = len(tgrid)
    rates = np.zeros(L)
    tau_mu_f = np.zeros(L)
    signal = np.zeros(L)
    # set initial values
    mu_f = muf_init
    x = x_init
    signal[0] = s_init

    pert_cnt = 0
    n_perts = len(tperts)
    for i_t in range(1,L):
        # interpolate
        w1, w2 = interpolate_x(mu_f, mu_vals)
        rates[i_t-1] = lookup_x(r_ss_array, w1, w2)[0]
        tau_mu_f[i_t-1] = lookup_x(tau_mu_array, w1, w2)[0]
        # euler step
        mu_f += dt * (mu + J*signal[i_t-1] - mu_f)/tau_mu_f[i_t-1]
        signal[i_t] = signal[i_t-1] + dt*x
        x -= dt * (2.0/tau * x + signal[i_t-1]/tau2)
                    
        if pert_cnt<n_perts:         
            if tgrid[i_t-1]>=tperts[pert_cnt]+d and \
               tgrid[i_t-1]<tperts[pert_cnt]+d+dt:       
                x += 1.0
                pert_cnt += 1
    
    w1, w2 = interpolate_x(mu_f, mu_vals)
    rates[i_t] = lookup_x(r_ss_array, w1, w2)[0] 

    return rates, J*signal, mu_f, signal[-1], x  # includes the last vals of the state variables


@njit
def sim_only_mu_perturbation(tgrid, tperts, J, tau):
    dt = tgrid[1]-tgrid[0]
    jump = np.exp(1)/tau  # needed such that peak of s = 1 for a single click
    L = len(tgrid)
    s = np.zeros(L)
    x = 0.0
    pert_cnt = 0
    n_perts = len(tperts)
    tau2 = tau**2
    for i_t in range(1,L):
        s[i_t] = s[i_t-1] + dt*x
        x -= dt * (2.0/tau * x + s[i_t-1]/tau2)
                    
        if pert_cnt<n_perts:         
            if tgrid[i_t]>=tperts[pert_cnt] and tgrid[i_t]<tperts[pert_cnt]+dt:       
                x += jump
                pert_cnt += 1
    return J*s



@njit
def interpolate_x(xi, rangex):
    dimx = len(rangex)
    if xi <= rangex[0]:
        idx = 0
        distx = 0.0
    elif xi >= rangex[-1]:
        idx = -1
        distx = 0.0
    else:
        for i in xrange(dimx-1):
            if rangex[i] <= xi and xi < rangex[i+1]:
                idx = i
                distx = (xi-rangex[i])/(rangex[i+1]-rangex[i])

    return idx, distx


@njit
def lookup_x(table, idx, distx):
    val = table[idx]*(1-distx) + table[idx+1]*distx
    return val

  

def calc_spiketrain_likelihood(args):
    params, ISIs, ISImean, ISImax, mu, sigma = args
    # only calculate pISI if the spike rate > a reasonable minimum (e.g. 0.5 Hz)
    # and if the spike rate is within a reasonable interval around the observed 
    # rate -- this is not essential and may be omitted 
    ISImean_min = 0.25*ISImean
    ISImean_max = 4*ISImean
    # LIF/EIF steady state output:
    p_ss, r_ss, q_ss = EIF_steady_state_numba(params['V_vals'], params['V_r_idx'],
                                              params['tau_m'], params['V_r'],
                                              params['V_T'], params['Delta_T'], 
                                              mu, sigma) 
    r_ss_ref = r_ss/(1+r_ss*params['T_ref'])
    ISImean_cand = 1.0/r_ss_ref    
 
    if r_ss_ref>0.0005 and ISImean_cand>ISImean_min and ISImean_cand<ISImean_max:
        if params['pISI_method'] == 'fourier':
            w_vec = 2*np.pi*params['freq_vals']
            df = params['freq_vals'][1] - params['freq_vals'][0]
            n = 2*len(params['freq_vals'])-1  #assuming freq_vals start with 0
            dt = 1.0/(df*n)  
            pISI_times = np.arange(0,n)*dt 
            inds = pISI_times<=ISImax
            pISIhat = EIF_ISIdensityhat_numba(params['V_vals'], params['V_r_idx'],
                                              params['tau_m'], params['V_r'],
                                              params['V_T'], params['Delta_T'], 
                                              params['T_ref'], mu, sigma, w_vec) 
            pISI_vals = np.fft.irfft(pISIhat,n)/dt
            pISI_vals[pISI_vals<=0] = 1e-20  # to avoid neg. values 
                                             # (which are numerically possible)
            pISI_vals = pISI_vals[inds]
            pISI_times = pISI_times[inds] 
        elif params['pISI_method'] == 'fvm':
            params['t_grid'] = np.arange(0, ISImax, params['fvm_dt'])
            mu = mu*np.ones_like(params['t_grid'])
            sigma = sigma*np.ones_like(params['t_grid'])
            fvm_dict = pISI_fvm_sg(mu, sigma, params, fpt=True, rt=np.array([0]))
            pISI_vals = fvm_dict['pISI_vals'] 
            pISI_times = params['t_grid']                  
        ISIs = ISIs[ISIs<ISImax]
        lval = np.nan #likelihood(ISIs, pISI_times[inds], pISI_vals[inds]) 
                      # omited here to save time
        loglval = loglikelihood(ISIs, pISI_times, pISI_vals)
        if np.isnan(loglval):
            print('')
            print('problem at mu,sigma = ')
            print(mu, sigma)
    else:     
        lval = np.nan
        loglval = np.nan
        pISI_vals = np.nan
        pISI_times = np.nan
        
    return lval, loglval, pISI_times, pISI_vals    

    
#@njit 
#def likelihood(ISIs, tvals, FPT): 
#    lval = 1 
#    for k in range(len(ISIs)):
#        lval *= interpol(ISIs[k], tvals, FPT)
#    return lval
    
    
@njit 
def loglikelihood(ISIs, tvals, FPT): 
    lval = 0.0 
    for k in range(len(ISIs)):
        lval += np.log( interpol(ISIs[k], tvals, FPT) )
    return lval


#@njit
def loglikelihood_Poisson(tgrid, rate, rcumsum, Sptimes, 
                          lastSpidx, ISImin, ISImax):
    dt = tgrid[1] - tgrid[0]
    lval = 0.0 
    for i_ts in range(1,len(Sptimes)):
        if not i_ts-1 in lastSpidx:
            t_s = Sptimes[i_ts]
            t_last_s = Sptimes[i_ts-1]
            ISI = t_s-t_last_s
            if ISI >= ISImin and ISI <= ISImax:
                r_integral = dt*( interpol(t_s, tgrid, rcumsum) - \
                                  interpol(t_last_s, tgrid, rcumsum) )
                lval += np.log( interpol(t_s, tgrid, rate) ) - r_integral
    return lval


#def loglikelihood_Poisson_rconst(rate, Sptimes, lastSpidx, ISImin, ISImax):
#    lval = 0.0 
#    for i_ts in range(1,len(Sptimes)):
#        if not i_ts-1 in lastSpidx:
#            t_s = Sptimes[i_ts]
#            t_last_s = Sptimes[i_ts-1]
#            ISI = t_s-t_last_s
#            if ISI >= ISImin and ISI <= ISImax:
#                r_integral = rate*ISI  # doublecheck!
#                lval += np.log(rate) - r_integral
#    return lval

    
@njit        
def interpol(xi, xvals, yvals):
    if xi <= xvals[0]:
        i = 0
        weight = 0
    elif xi >= xvals[-1]:
        i = -1
        weight = 0
    else:
        i = 0
        while xi < xvals[i] or xi >= xvals[i+1]:
            i += 1
        weight = (xi-xvals[i]) / (xvals[i+1]-xvals[i])
    yi = yvals[i] + weight*(yvals[i+1]-yvals[i])
    return yi      
    

@njit        
def interp1d_getweight(xi, xvals):
    if xi <= xvals[0]:
        i = 0
        weight = 0
    elif xi >= xvals[-1]:
        i = -1
        weight = 0
    else:
        i = 0
        while xi < xvals[i] or xi >= xvals[i+1]:
            i += 1
        weight = (xi-xvals[i]) / (xvals[i+1]-xvals[i])
    return i, weight

### ===========================================================================
    
'''
3rd part: functions to calculate and plot the quantities required by the LNexp 
spike rate model (method 2), including functions to compute the steady state and 
the first order spike rate response of an exponential/leaky I&F neuron subject 
to white noise input, adapted from:
Augustin*, Ladenbauer*, Baumann, Obermayer, PLOS Comput. Biol. 2017
https://github.com/neuromethods/fokker-planck-based-spike-rate-models
'''

# COMPUTING FUNCTIONS ---------------------------------------------------------

# prepares data structures and calls computing functions (possibly in parallel)
def calc_EIF_output_and_cascade_quants(mu_vals, sigma_vals, params, 
                                       EIF_output_dict, output_names,
                                       LN_quantities_dict, quantity_names):
        
    N_mu_vals = len(mu_vals)    
    N_sigma_vals = len(sigma_vals)
    if N_sigma_vals<=params['N_procs']:
        N_procs = 1
    else:
        N_procs = params['N_procs']
        
    # create EIF_output_dict arrays to be filled
    for n in output_names:  
        # real values dependent on mu, sigma
        EIF_output_dict[n] = np.zeros((N_mu_vals, N_sigma_vals))
    
    # create quantities_dict arrays to be filled
    for n in quantity_names:  
        # real values dependent on mu, sigma
        LN_quantities_dict[n] = np.zeros((N_mu_vals, N_sigma_vals))
                
    arg_tuple_list = [(isig, sigma_vals[isig], mu_vals, params, output_names,
                       quantity_names) for isig in range(N_sigma_vals)]    
                      
    comp_total_start = time.time()
                  
    if N_procs <= 1:
       # single process version
        pool = False
        result = (output_and_quantities_given_sigma_wrapper(arg_tuple) 
                  for arg_tuple in arg_tuple_list) 
    else:
        # multiproc version
        pool = multiprocessing.Pool(params['N_procs'])
        result = pool.imap_unordered(output_and_quantities_given_sigma_wrapper, 
                                     arg_tuple_list)
        
    finished = 0 
    for isig, res_given_sigma_dict in result:
        finished += 1
        print(('{count} of {tot} steady-state / rate response and LN quantity ' + \
               'calculations completed').format(count=finished*N_mu_vals, 
                                                tot=N_mu_vals*N_sigma_vals)) 
        for k in res_given_sigma_dict.keys():
            for imu, mu in enumerate(mu_vals):
                if k in output_names and k not in ['r1_mumod', 'r1_sigmamod']:
                    EIF_output_dict[k][imu,isig] = res_given_sigma_dict[k][imu] 
                if k in quantity_names:
                    LN_quantities_dict[k][imu,isig] = res_given_sigma_dict[k][imu]    
    if pool:
        pool.close()
            
    # also include mu_vals, sigma_vals, and freq_vals in output dictionaries
    EIF_output_dict['mu_vals'] = mu_vals
    EIF_output_dict['sigma_vals'] = sigma_vals
    EIF_output_dict['freq_vals'] = params['freq_vals'].copy()

    LN_quantities_dict['mu_vals'] = mu_vals
    LN_quantities_dict['sigma_vals'] = sigma_vals
    LN_quantities_dict['freq_vals'] = params['freq_vals'].copy()
    
    print('computation of: {} done'.format(output_names))
    print('total time for computation (N_mu_vals={Nmu}, N_sigma_vals={Nsig}): {rt}s'.
          format(rt=np.round(time.time()-comp_total_start,2), Nmu=N_mu_vals, 
                 Nsig=N_sigma_vals))
      
    return EIF_output_dict, LN_quantities_dict
    

# wrapper function that calls computing functions for a given sigma value and 
# looping over all given mu values (depending on what needs to be computed)
def output_and_quantities_given_sigma_wrapper(arg_tuple):
    isig, sigma, mu_vals, params, output_names, quantity_names = arg_tuple
    # a few shortcuts
    V_vec = params['V_vals']
    Vr = params['V_r']
    kr = params['V_r_idx']
    VT = params['V_T']
    taum = params['tau_m']
    DeltaT = params['Delta_T']
    
    dV = V_vec[1]-V_vec[0]
    Tref = params['T_ref']
    
    N_mu_vals = len(mu_vals)    
    res_given_sigma_dict = dict()
    
    for n in output_names:  
        # real values dependent on mu, sigma
        res_given_sigma_dict[n] = np.zeros(N_mu_vals)
     
    for n in quantity_names:    
        if n not in ['r_ss', 'V_mean_ss']:  # omit doubling
            res_given_sigma_dict[n] = np.zeros(N_mu_vals)
            
    
    for imu, mu in enumerate(mu_vals):    
        # first, steady state output & derivatives drdmu, drdsigma
        p_ss, r_ss, q_ss = EIF_steady_state_numba(V_vec, kr, taum, Vr, VT, DeltaT, 
                                                  mu, sigma) 
        _, r_ss_dmu, _ = EIF_steady_state_numba(V_vec, kr, taum, Vr, VT, DeltaT, 
                                                mu+params['d_mu'], sigma)
        _, r_ss_dsig, _ = EIF_steady_state_numba(V_vec, kr, taum, Vr, VT, DeltaT, 
                                                 mu, sigma+params['d_sigma'])
           
        if 'V_mean_ss' in output_names:
            # disregarding refr. period (otherwise see below)
            V_mean = dV*np.sum(V_vec*p_ss)  
            res_given_sigma_dict['V_mean_ss'][imu] = V_mean
        
        r_ss_ref = r_ss/(1+r_ss*Tref)    
        p_ss = r_ss_ref * p_ss/r_ss
        q_ss = r_ss_ref * q_ss/r_ss  # prob. flux needed for sigma-mod calculation              
        r_ss_dmu_ref = r_ss_dmu/(1+r_ss_dmu*Tref) - r_ss_ref
        r_ss_dsig_ref = r_ss_dsig/(1+r_ss_dsig*Tref) - r_ss_ref   
        
        if 'V_mean_sps_ss' in output_names:
            # when considering spike shape (during refr. period) use this Vmean;
            # note that the density reflecting nonrefr. proportion integrates to 
            # r_ss_ref/r_ss 
            Vmean_sps = dV*np.sum(V_vec*p_ss) + \
                        (1-r_ss_ref/r_ss)*(params['Vcut']+Vr)/2  
            # note: (1-r_ss_ref/r_ss)==r_ss_ref*Tref 
            res_given_sigma_dict['V_mean_sps_ss'][imu] = Vmean_sps
            
        if 'r_ss' in output_names:
            res_given_sigma_dict['r_ss'][imu] = r_ss_ref
            
        if 'dr_ss_dmu' in output_names:      
            dr_ss_dmu = r_ss_dmu_ref/params['d_mu']
            res_given_sigma_dict['dr_ss_dmu'][imu] = dr_ss_dmu
            
        if 'dr_ss_dsigma' in output_names:        
            dr_ss_dsig = r_ss_dsig_ref/params['d_sigma']
            res_given_sigma_dict['dr_ss_dsigma'][imu] = dr_ss_dsig
              
        # next, rate response for mu-modulation across the given modulation 
        # frequency range
        if 'r1_mumod' in output_names:
            w_vec = 2*np.pi*params['freq_vals']
            inhom = params['d_mu']*p_ss
            r1mu_vec = EIF_lin_rate_response_frange_numba(V_vec, kr, taum, Vr, 
                                      VT, DeltaT, Tref, mu, sigma, inhom, w_vec)

        # next, the quantities obtained by fitting the filters semi-analytically
        # (in the Fourier domain)
        
        # fitting exponential function A*exp(-t/tau) to normalized rate response 
        # in Fourier domain: A*tau / (1 + 1i*2*pi*f*tau)  f in kHz, tau in ms 
        # with A = 1/tau to guarantee equality at f=0,
        # see Augustin et al. 2017 before Eq. 85 for more details
        if 'tau_mu_exp' in quantity_names:            
            # use normalized rate response r1 for fitting, normalize such that 
            # its value at f=0 is one (by dividing by r1_mumod_f0, the real value 
            # equal to the time-integral of the filter from 0 to inf)         
            #r1_mumod_f0 = dr_ss_dmu 
            #r1_mumod_normalized = r1mu_vec/params['d_mu'] /r1_mumod_f0
            r1_mumod_normalized = r1mu_vec/r1mu_vec[0]  # this version is more robust 
            # because r1 for f->0 and dr_ss_dmu deviate for some parametrizations
            init_val = 1.0 #ms
            tau = fit_exponential_freqdom(params['freq_vals'], r1_mumod_normalized, 
                                          init_val)
            res_given_sigma_dict['tau_mu_exp'][imu] = tau
                
        #print 'calculation for mu =', mu, 'done'  
    #print 'calculations for sigma =', sigma, 'done' 
    return isig, res_given_sigma_dict

    
                                 
# CORE FUNCTIONS that calculate steady state and 1st order spike rate response 
# to modulations for an EIF/LIF neuron subject to white noise input via the 
# Fokker-Planck system
    
    
@njit
def EIF_steady_state_numba(V_vec, kr, taum, Vr, VT, DeltaT, mu, sigma):
    dV = V_vec[1]-V_vec[0]
    sig2term = 2.0/sigma**2
    n = len(V_vec)
    p_ss = np.zeros(n);  q_ss = np.ones(n);
    if DeltaT>0:
        Psi = DeltaT*np.exp((V_vec-VT)/DeltaT)
    else:
        Psi = 0.0*V_vec
    F = sig2term*( ( V_vec-Psi )/taum - mu ) 
    A = np.exp(dV*F)
    F_dummy = F.copy()
    F_dummy[F_dummy==0.0] = 1.0
    B = (A - 1.0)/F_dummy * sig2term
    sig2term_dV = dV * sig2term
    for k in xrange(n-1, kr, -1):
        if not F[k]==0.0:
            p_ss[k-1] = p_ss[k] * A[k] + B[k]
        else:
            p_ss[k-1] = p_ss[k] * A[k] + sig2term_dV
        q_ss[k-1] = 1.0    
    for k in xrange(kr, 0, -1):  
        p_ss[k-1] = p_ss[k] * A[k]
        q_ss[k-1] = 0.0
    p_ss_sum = np.sum(p_ss)   
    r_ss = 1.0/(dV*p_ss_sum)
    p_ss *= r_ss;  q_ss *= r_ss;
    return p_ss, r_ss, q_ss

    
@njit 
def EIF_lin_rate_response_frange_numba(V_vec, kr, taum, Vr, VT, DeltaT, Tref,
                                       mu, sigma, inhom, w_vec): 
    dV = V_vec[1]-V_vec[0]
    sig2term = 2.0/sigma**2
    n = len(V_vec)
    r1_vec = 1j*np.ones(len(w_vec))    
    if DeltaT>0:
        Psi = DeltaT*np.exp((V_vec-VT)/DeltaT)
    else:
        Psi = 0.0*V_vec
    F = sig2term*( ( V_vec-Psi )/taum - mu )
    A = np.exp(dV*F)
    F_dummy = F.copy()
    F_dummy[F_dummy==0.0] = 1.0
    B = (A - 1.0)/F_dummy * sig2term
    sig2term_dV = dV * sig2term
    for iw in range(len(w_vec)):
        q1a = 1.0 + 0.0*1j;  p1a = 0.0*1j;  q1b = 0.0*1j;  p1b = 0.0*1j;
        fw = dV*1j*w_vec[iw]
        refterm = np.exp(-1j*w_vec[iw]*Tref)  
        for k in xrange(n-1, 0, -1):
            if not k==kr+1:
                q1a_new = q1a + fw*p1a
            else:
                q1a_new = q1a + fw*q1a - refterm
            if not F[k]==0.0:    
                p1a_new = p1a * A[k] + B[k] * q1a
                p1b_new = p1b * A[k] + B[k] * (q1b - inhom[k])
            else:
                p1a_new = p1a * A[k] + sig2term_dV * q1a
                p1b_new = p1b * A[k] + sig2term_dV * (q1b - inhom[k])
            q1b += fw*p1b
            q1a = q1a_new;  p1a = p1a_new;  p1b = p1b_new;   
        r1_vec[iw] = -q1b/q1a
    return r1_vec    
 
    

@njit
def EIF_lin_rate_response_numba(V_vec, kr, taum, Vr, VT, DeltaT, Tref,
                                mu, sigma, inhom, w): 
    dV = V_vec[1]-V_vec[0]
    sig2term = 2.0/sigma**2
    n = len(V_vec)   
    q1a = 1.0 + 0.0*1j;  p1a = 0.0*1j;  q1b = 0.0*1j;  p1b = 0.0*1j;
    fw = dV*1j*w
    refterm = np.exp(-1j*w*Tref)
    if DeltaT>0:
        Psi = DeltaT*np.exp((V_vec-VT)/DeltaT)
    else:
        Psi = 0.0*V_vec
    F = sig2term*( ( V_vec-Psi )/taum - mu ) 
    A = np.exp(dV*F)
    F_dummy = F.copy()
    F_dummy[F_dummy==0.0] = 1.0
    B = (A - 1.0)/F_dummy * sig2term
    sig2term_dV = dV * sig2term
    for k in xrange(n-1, 0, -1):
        if not k==kr+1:
            q1a_new = q1a + fw*p1a
        else:
            q1a_new = q1a + fw*q1a - refterm
        if not F[k]==0.0:    
            p1a_new = p1a * A[k] + B[k] * q1a
            p1b_new = p1b * A[k] + B[k] * (q1b - inhom[k])
        else:
            p1a_new = p1a * A[k] + sig2term_dV * q1a
            p1b_new = p1b * A[k] + sig2term_dV * (q1b - inhom[k])
        q1b += fw*p1b
        q1a = q1a_new;  p1a = p1a_new;  p1b = p1b_new;   
    r1 = -q1b/q1a
    return r1    

# the functions above work efficiently in practice, but the integration schemes 
# might be improved (e.g., based on the Magnus expansion) to allow for larger 
# membrane voltage discretization steps  
    
           
def fit_exponential_freqdom(f, r1_mod_normalized, init_val):
    tau_lb = 0.001  #ms, lower bound 
    tau_ub = 100.0  #ms, upper bound
    # first global brute-force optimization on a coarse grid to reduce risk of 
    # finding a local optimum
    tau_step = 1.0  #ms
    tau_vals = np.arange(tau_lb, tau_ub, tau_step)
    errors = np.zeros_like(tau_vals)
    for i, tau in enumerate(tau_vals):
        errors[i] = exp_mean_sq_dist(tau, f, r1_mod_normalized)
    idx = np.argmin(errors)
    # then refinement using a finer grid
    if idx<len(tau_vals)-1:
        tau_ub = tau_vals[idx+1]
    if idx>0:
        tau_lb = tau_vals[idx-1]
    tau_step = 0.01  #ms
    tau_vals = np.arange(tau_lb, tau_ub, tau_step)
    errors = np.zeros_like(tau_vals)
    for i, tau in enumerate(tau_vals):
        errors[i] = exp_mean_sq_dist(tau, f, r1_mod_normalized)
    idx = np.argmin(errors)
    tau = tau_vals[idx]   
    # or, alternatively, use an "off-the-shelf" optimization method  
#    sol = scipy.optimize.minimize_scalar(exp_mean_sq_dist, args=(f, r1_mod_normalized), 
#                                         bounds=(tau_lb, tau_ub), method='bounded', 
#                                         options={'disp':True, 'maxiter':500, 'xatol':1e-3})
#    tau = sol.x                            
    return tau
   
    
@njit    
def exp_mean_sq_dist(tau, *args):
    f, r1_mod_normalized = args
    exp_fdom = 1.0 / (1.0 + 1j*2.0*np.pi*f*tau)  # exp. function in Fourier domain
    error = np.sum(np.abs(exp_fdom - r1_mod_normalized)**2)  
    # correct normalization for mean squared error is not important for optim.
    return error
    


# LOAD / SAVE FUNCTIONS -------------------------------------------------------
    
def load(filepath, input_dict, quantities, param_dict):
    
    print('loading {} from file {}'.format(quantities, filepath))
    try:
        h5file = tables.open_file(filepath, mode='r')
        root = h5file.root
        
        for q in quantities:
            input_dict[q] = h5file.get_node(root, q).read()            
                   
        # loading parameters
        # only overwrite what is in the file, do not start params from scratch, 
        # otherwise: uncomment following line
        #param_dict = {} 
        for child in root.params._f_walknodes('Array'):
            param_dict[child._v_name] = child.read()[0]
        for group in root.params._f_walk_groups():
            if group != root.params: # walk group first yields the group itself, 
                                     # then its children
                param_dict[group._v_name] = {}
                for subchild in group._f_walknodes('Array'):
                    param_dict[group._v_name][subchild._v_name] = subchild.read()[0]            
        
        h5file.close()
    
    except IOError:
        warn('could not load quantities from file '+filepath)
    except:
        h5file.close()
    
    print('')
        

def save(filepath, output_dict, param_dict):
    
    print('saving {} into file {}'.format(output_dict.keys(), filepath))
    try:
        h5file = tables.open_file(filepath, mode='w')
        root = h5file.root
            
        for k in output_dict.keys():
            h5file.create_array(root, k, output_dict[k])
            print('created array {}'.format(k))
            
        h5file.create_group(root, 'params', 'Neuron model and numerics parameters')
        for name, value in param_dict.items():
            # for python2/3 compat.:
            # if isinstance(name, str):
            # name = name.encode(encoding='ascii') # do not allow unicode keys
            if isinstance(value, (int, float, bool)):
                h5file.create_array(root.params, name, [value])
            elif isinstance(value, str):
                h5file.create_array(root.params, name, 
                                    [value.encode(encoding='ascii')])
            elif isinstance(value, dict):
                params_sub = h5file.create_group(root.params, name)
                for nn, vv in value.items():
                    # for python2/3 compat.:
                    # if isinstance(nn, str):
                    # nn = nn.encode(encoding='ascii') # do not allow unicode keys
                    if isinstance(vv, str):
                        h5file.create_array(params_sub, nn, 
                                            [vv.encode(encoding='ascii')])
                    else:
                        h5file.create_array(params_sub, nn, [vv])
        h5file.close()
    
    except IOError:
        warn('could not write quantities into file {}'.format(filepath))
    except:
        h5file.close()

    print('')


# PLOTTING FUNCTION -----------------------------------------------------------
    
def plot_quantities(quantities_dict, quantity_names, sigmas_plot):
    
    mu_vals = quantities_dict['mu_vals']
    sigma_vals = quantities_dict['sigma_vals']   
    
    plt.figure()
    plt.suptitle('LNexp quantities')
    
    mu_lim = [-5, 5]
    inds_mu_plot = [i for i in range(len(mu_vals)) if \
                    mu_lim[0] <= mu_vals[i] <= mu_lim[1]]
    inds_sigma_plot = [np.argmin(np.abs(sigma_vals-sig)) for sig in sigmas_plot]
    N_sigma = len(inds_sigma_plot)    
    
    for k_j, j in enumerate(inds_sigma_plot):
        # color    
        rgb = [0, float(k_j+1)/(N_sigma), 0]
        linecolor = rgb
        
        if 'r_ss' in quantity_names:
            ax1 = plt.subplot(2, 1, 1)
            # labels
            if k_j in [0, N_sigma//2, N_sigma-1]:
                siglabel = \
                '$\sigma={0:.3}$ [mV/$\sqrt{{\mathrm{{ms}}}}$]'.format(sigma_vals[j])
                           
            else:
                siglabel = None
    
            plt.plot(mu_vals[inds_mu_plot], quantities_dict['r_ss'][inds_mu_plot,j], 
                     label=siglabel, color=linecolor)
    
            if k_j==0:
                plt.ylabel('$r_{\infty}$ [kHz]')
            if k_j==N_sigma-1:
                plt.legend(loc='best')
        
#        if 'V_mean_ss' in quantity_names:
#            plt.subplot(2, 1, 2, sharex=ax1)              
#            plt.plot(mu_vals[inds_mu_plot], 
#                     quantities_dict['V_mean_ss'][inds_mu_plot,j], 
#                     label=siglabel, color=linecolor)
#            if k_j==0:
#                plt.ylabel('$\langle V \\rangle_{\infty}$ [mV]')
            
        if 'tau_mu_exp' in quantity_names:
            plt.subplot(2, 1, 2, sharex=ax1)
            plt.plot(mu_vals[inds_mu_plot], 
                     quantities_dict['tau_mu_exp'][inds_mu_plot,j], 
                     label=siglabel, color=linecolor)
            plt.xlabel(r'$\mu$ [mV/ms]')
            if k_j==0:
                plt.ylabel(r'$\tau_{\mu}$ [ms]')          
       
    plt.show()


### ===========================================================================
    
'''
4th part: functions to calculate the ISI density via a finite volume method, adapted from: 
Augustin*, Ladenbauer*, Baumann, Obermayer, PLOS Comput. Biol. 2017
https://github.com/neuromethods/fokker-planck-based-spike-rate-models
(FVM implementation adjusted here to solve the first passage time problem)
'''


class Grid(object):  # this class implements the voltage discretization
   
    def __init__(self, V_0=-200., V_1=-40., V_r=-70., N_V=100):
        self.V_0 = V_0
        self.V_1 = V_1
        self.V_r = V_r
        self.N_V = int(N_V)
        # construct the grid object
        self.construct()

    def construct(self):
        self.V_centers = np.linspace(self.V_0, self.V_1, self.N_V)
        # shift V_centers by half of the grid spacing to the left
        # such that the last interface lies exactly on V_l
        self.V_centers -= (self.V_centers[-1] - self.V_centers[-2]) / 2.
        self.dV_centers = np.diff(self.V_centers)
        self.V_interfaces = np.zeros(self.N_V + 1)
        self.V_interfaces[1:-1] = self.V_centers[:-1] + 0.5 * self.dV_centers
        self.V_interfaces[0] = self.V_centers[0] - 0.5 * self.dV_centers[0]
        self.V_interfaces[-1] = self.V_centers[-1] + 0.5 * self.dV_centers[-1]
        self.dV_interfaces = np.diff(self.V_interfaces)
        self.dV = self.V_interfaces[2] - self.V_interfaces[1]
        self.ib = np.argmin(np.abs(self.V_centers - self.V_r))


@njit
def get_v_numba(L, Vi, DT, VT, taum, mu, EIF = True):
    # drift coeffs for EIF/LIF model
    # LIF model
    drift = np.empty(L)
    if not EIF:
        for i in xrange(L):
            drift[i] = mu - Vi[i] / taum
    # EIF model
    else:
        for i in xrange(L):
            drift[i] = ( - Vi[i] + DT * exp((Vi[i] - VT) / DT) ) / taum + mu
    return drift


@njit
def exp_vdV_D(v,dV,D): # helper function for diags_A
    return exp(-v*dV/D)
  

@njit
def matAdt_opt(mat,N,v,D,dV,dt):
    dt_dV = dt/dV

    for i in xrange(1,N-1):
        exp_vdV_D1 = exp_vdV_D(v[i],dV,D)
        if exp_vdV_D1 != 1.0: 
            mat[1,i] = -dt_dV*v[i]*exp_vdV_D1/(1.0-exp_vdV_D1) # diagonal
            mat[2,i-1] = dt_dV*v[i]/(1.0-exp_vdV_D1) # lower diagonal
        else:
            mat[1,i] = -dt_dV*D/dV # diagonal
            mat[2,i-1] = dt_dV*D/dV # lower diagonal
        exp_vdV_D2 = exp_vdV_D(v[i+1],dV,D)
        if exp_vdV_D2 != 1.0:
            mat[1,i] -= dt_dV*v[i+1]/(1.0-exp_vdV_D2) # diagonal  
            mat[0,i+1] = dt_dV*v[i+1]*exp_vdV_D2/(1.0-exp_vdV_D2) # upper diagonal
        else:
            mat[1,i] -= dt_dV*D/dV # diagonal
            mat[0,i+1] = dt_dV*D/dV # upper diagonal
            
    # boundary conditions
    exp_vdV_D1 = exp_vdV_D(v[1],dV,D)
    exp_vdV_D2 = exp_vdV_D(v[-1],dV,D)
    exp_vdV_D3 = exp_vdV_D(v[-2],dV,D)
    if exp_vdV_D1 != 1.0:
        tmp1 = v[1]/(1.0-exp_vdV_D1)
    else:
        tmp1 = D/dV
    if exp_vdV_D2 != 1.0:
        tmp2 = v[-1]/(1.0-exp_vdV_D2)
    else:
        tmp2 = D/dV
    if exp_vdV_D3 != 1.0:
        tmp3 = v[-2]/(1.0-exp_vdV_D3)
    else:
        tmp3 = D/dV
    
    mat[1,0] = -dt_dV*tmp1  # first diagonal
    mat[0,1] = dt_dV*tmp1*exp_vdV_D(v[1],dV,D)  # first upper  
    mat[2,-2] = dt_dV*tmp3  # last lower
    mat[1,-1] = -dt_dV * ( tmp3*exp_vdV_D(v[-2],dV,D) 
                          +tmp2*(1.0+exp_vdV_D(v[-1],dV,D)) )  # last diagonal


# initial probability density
def initial_p_distribution(grid,params):
    if params['fvm_v_init'] == 'normal':
        mean_gauss = params['fvm_normal_mean']
        sigma_gauss = params['fvm_normal_sigma']
        p_init = np.exp(-np.power((grid.V_centers - mean_gauss), 2) / 
                        (2 * sigma_gauss ** 2))
    elif params['fvm_v_init'] == 'delta':
        delta_peak_index = np.argmin(np.abs(grid.V_centers - 
                                            params['fvm_delta_peak']))
        p_init = np.zeros_like(grid.V_centers)
        p_init[delta_peak_index] = 1.
    elif params['fvm_v_init'] == 'uniform':
        # uniform dist on [Vr, Vs]
        p_init = np.zeros_like(grid.V_centers)
        p_init[grid.ib:] = 1.
    else:
        err_mes = ('Initial condition "{}" is not implemented!' + \
                   'See params dict for options.').format(params['fvm_v_init'])
        raise NotImplementedError(err_mes)
    # normalization with respect to the cell widths
    p_init =p_init/np.sum(p_init*grid.dV_interfaces)
    return p_init


@njit
def get_r_numba(v_end, dV, D, p_end):
    # calculation of rate/pISI
    tmp = exp((-v_end*dV)/D)
    if tmp != 1.0:
        r = v_end*((1.0+tmp)/(1.0-tmp))*p_end
    else:
        r = 2*D/dV * p_end
    return r


def pISI_fvm_sg(mu, sigma, params, fpt=True, rt=list()):
    # solves the Fokker Planck equation (first passage time problem)
    # using the Scharfetter-Gummel finite volume method

    dt = params['fvm_dt']
    T_ref = params['T_ref']
    DT = params['Delta_T']
    VT = params['V_T']
    taum = params['tau_m']

    EIF_model = True if params['neuron_model'] == 'EIF' else False

    # instance of the spatial grid class
    grid = Grid(V_0=params['V_lb'], V_1=params['V_s'], V_r=params['V_r'],
                N_V=params['N_centers_fvm'])

    pISI_vals = np.zeros_like(mu)

    dV = grid.dV
    Adt = np.zeros((3,grid.N_V))

    rc=0
    n_rt = len(rt)
    ones_mat = np.ones(grid.N_V)
    
    # drift coefficients
    v = get_v_numba(grid.N_V+1, grid.V_interfaces, DT, VT,
                    taum, mu[0], EIF = EIF_model)  
    # diffusion coefficient
    D = (sigma[0] ** 2) * 0.5  
    # create banded matrix A 
    matAdt_opt(Adt,grid.N_V,v,D,dV,dt)
    Adt *= -1.
    Adt[1,:] += ones_mat
                    
    for n in xrange(len(mu)):

        if rc<n_rt and rt[rc]<= n*dt < rt[rc]+dt:
            p = initial_p_distribution(grid, params)
            rc += 1
            
        if rc-1<n_rt and rt[rc-1]<= n*dt < rt[rc-1]+T_ref+dt:
            pISI_vals[n] = 0
        else:
            
            if n>0:
                toggle = False 
                if mu[n]!=mu[n-1]:
                    # drift coefficients
                    v = get_v_numba(grid.N_V+1, grid.V_interfaces, DT, VT,
                                    taum, mu[n], EIF = EIF_model)
                    toggle = True
                if sigma[n]!=sigma[n-1]:
                    # diffusion coefficient
                    D = (sigma[n] ** 2) * 0.5
                    toggle = True
                if toggle:
                    # create banded matrix A in each time step
                    matAdt_opt(Adt,grid.N_V,v,D,dV,dt)
                    Adt *= -1.
                    Adt[1,:] += ones_mat

            rhs = p.copy()   
            
            # solve the linear system
            p_new = solve_banded((1, 1), Adt, rhs)

            # compute rate / pISI
            pISI_vals[n] = get_r_numba(v[-1], dV, D, p_new[-1])
    
            p = p_new

    results = {'pISI_vals':pISI_vals}
    return results





def pISI0pISI1_deltaperts_fvm_sg(mu0, tpert_vec, sigma, params):
    # solves the Fokker Planck equation (first passage time problem)
    # for the the unperturbed system and the correction pISI1 for a small 
    # perturbation of mean input at time tpert, delta(t-tpert), using the 
    # Scharfetter-Gummel finite volume method
    epsilon = 1e-4
    dt = params['fvm_dt']
    T_ref = params['T_ref']
    DT = params['Delta_T']
    VT = params['V_T']
    taum = params['tau_m']

    EIF_model = True if params['neuron_model'] == 'EIF' else False

    # instance of the spatial grid class
    grid = Grid(V_0=params['V_lb'], V_1=params['V_s'], V_r=params['V_r'],
                N_V=params['N_centers_fvm'])

    pISI0 = np.zeros_like(params['t_grid'])
    pISI1_exc = np.zeros((len(tpert_vec), len(pISI0)))
    #pISI1_inh = np.zeros((len(tpert_vec), len(pISI0)))
    compute_pISI1 = True
    nt = len(pISI0)
    dV = grid.dV
    Adt = np.zeros((3,grid.N_V))
    ones_mat = np.ones(grid.N_V)

    # drift coefficients
    v = get_v_numba(grid.N_V+1, grid.V_interfaces, DT, VT,
                    taum, mu0, EIF = EIF_model)  
    # diffusion coefficient
    D = (sigma ** 2) * 0.5  
    # create banded matrix A 
    matAdt_opt(Adt,grid.N_V,v,D,dV,dt)  # changes the first argument
    Adt *= -1.
    Adt[1,:] += ones_mat
          
    p0 = initial_p_distribution(grid, params)
    pertcnt = 0
    for n in xrange(nt):          
        if n*dt < T_ref+dt:
            pISI0[n] = 0
        else:
            rhs = p0.copy()          
            # solve the linear system
            p0 = solve_banded((1, 1), Adt, rhs)
            # compute pISI0
            pISI0[n] = get_r_numba(v[-1], dV, D, p0[-1])
            
        idx = (n*dt<=tpert_vec) & (tpert_vec<(n+1)*dt)
        if any(idx) and compute_pISI1:
            prop_not_spiked = dV*np.sum(p0)
            # proportion of trials in which the neuron has not yet spiked 
            if prop_not_spiked > epsilon:
                # exc. perturbation: rightward shift in p
                p = p0.copy()
                p[1:] = p[:-1]  
                p[0] = 0              
                pISI = pISI0.copy()
                pISI[n] = get_r_numba(v[-1], dV, D, p[-1])
                # continue from tpert
                for k in range(n+1,nt):
                    rhs = p.copy()          
                    # solve the linear system
                    p = solve_banded((1, 1), Adt, rhs)
                    # compute pISI
                    pISI[k] = get_r_numba(v[-1], dV, D, p[-1])     
                pISI1_exc[pertcnt,:] = pISI.copy()               
                pertcnt += 1
            else:
                compute_pISI1 = False  # assuming tpert_vec is ordered
                
    for ip in range(pertcnt):
        pISI1_exc[ip,:] = (pISI1_exc[ip,:]-pISI0)/dV
         
    return pISI0, pISI1_exc #, pISI1_inh

