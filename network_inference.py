# -*- coding: utf-8 -*-
'''
example script for estimation of coupling strengths of a (possibly sub-sampled) 
network of leaky or exponential I&F neurons using method 1a, 
cf. Ladenbauer et al. 2018 (Results section 4, Fig 4A)
-- written by Josef Ladenbauer in 2018 

run time was <2 min. when using the finite volume method (use_fvm=True) and 
             <8 min. when using the Fourier method only (use_fvm=False) 
on an Intel i7-2600 quad-core PC using Python 2.7 (Anaconda distribution v. 5.0.1) 
'''

import inference_methods as im
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import multiprocessing
matplotlib.rc('text', usetex=True)
matplotlib.rc('xtick', labelsize=12) 
matplotlib.rc('ytick', labelsize=12) 
                                        
# SET PARAMETERS --------------------------------------------------------------
Nneurons_obs = 6  # number of neurons whose spike trains are observed
Nneurons_tot = 6  # number of neurons in total;
# note that the simulation code here is not optimized for large networks (N>50),
# for that purpose it is recommended to use simulation software, such as the 
# Python-based package Brian2 
                  
params = dict()
# neuron model parameters:
params['tau_m'] = 20.0  # ms, membrane time constant
params['V_r'] = -70.0  # mV, reset voltage
params['V_s'] = -40.0  # mV, spike voltage
params['Delta_T'] = 0.0  # mV, threshold slope factor (set 0 for LIF model, 
                         #                                >0 for EIF model)  
params['V_T'] = -50.0  # mV, effective threshold voltage (only used for EIF)
params['T_ref'] = 0.0  # ms, refractory duration

# input parameters:
np.random.seed(20)
mu_vals = -1.75 * (0.8 + 0.4*np.random.rand(Nneurons_tot))  # mV/ms, input mean
sigma_vals = 2.5 * (0.8 + 0.4*np.random.rand(Nneurons_tot))  # mV/sqrt(ms), input
                                                             # standard deviation

# parameters for data generation:
Jmat = 1.5*np.random.rand(Nneurons_tot,Nneurons_tot) - 0.75  # coupling strengths
Jmat -= np.diag(np.diag(Jmat))  # exclude autapses
d = 1.0  # ms, delay
input_cc = 0.0  # determines the correlation strength of external input fluctuations 
                # for each pair in the network 
t_end = 30e4  # ms, simulation duration ("recording time")
params['dt_sim'] = 0.05  # ms, simulation time step

# parameters for estimation (method 1a)
params['pISI_method'] = 'fourier'
f_max = 2000.0 # Hz, determines resolution (accuracy) of ISI density; 
               # 1k seems sufficient in many cases, for finer resolution try 2k or 4k
d_freq = 0.25 # Hz, spacing of frequency grid
d_V = 0.025  # mV, spacing of voltage grid
params['V_lb'] = -150.0  # mV, lower bound
params['V_vals'] = np.arange(params['V_lb'],params['V_s']+d_V/2,d_V)
params['freq_vals'] = np.arange(0.0, f_max+d_freq/2, d_freq)/1000  # kHz
params['V_r_idx'] = np.argmin(np.abs(params['V_vals']-params['V_r'])) 
                    # index of reset voltage on grid, this should be a grid point
                    
# parameters for estimation when using the finite volume method instead of the 
# Fourier method to calculate p_ISI^1, which is optional (and often faster)
use_fvm = True  # False: using Fourier method
if use_fvm:
    params['pISI_method'] = 'fvm' 
    d_V = 0.025  # mV, spacing of voltage grid
    params['V_lb'] = -150.0  # mV, lower bound
    params['V_vals'] = np.arange(params['V_lb'],params['V_s']+d_V/2,d_V)
    params['V_r_idx'] = np.argmin(np.abs(params['V_vals']-params['V_r'])) 
                        # index of reset voltage on grid, this should be a grid point
    params['neuron_model'] = 'LIF'
    params['integration_method'] = 'implicit'
    params['N_centers_fvm'] = 1000  # number of centers for voltage discretization
    params['fvm_v_init'] = 'delta'  # voltage density initialization
    params['fvm_delta_peak'] = params['V_r']  # location of initial density peak
    params['fvm_dt'] = 0.1  # ms, time step for finite volume method
                            # 0.1 seems ok, prev. def.: 0.05 ms 

sigma_init = 3.0  # initial sigma value (initial mu value will be determined by 
                  # sigma_init and empirical mean ISI)
N_tpert = 300  # determines spacing of (potential) perturbation times within ISIs
               # due to coupling; def.: 300            
J_bnds = (-3.0, 3.0)  # mV, min./max. coupling strength 
N_procs = int(5.0*multiprocessing.cpu_count()/6)  # number of parallel processes                
             
                
if __name__ == '__main__':
           
    # GENERATE SPIKE TRAINS from network model --------------------------------
    tgrid = np.arange(0, t_end+params['dt_sim']/2, params['dt_sim'])
            # time points for simulation
    
    V_init = params['V_r'] * np.ones(Nneurons_tot)

    start = time.time()
    randnvals = np.random.randn(Nneurons_tot,len(tgrid))
    randnvals_c = np.random.randn(len(tgrid)) # common noise
    print('')
    print('starting network simulation') 
    Spt_dummy, sp_counts, _ = \
        im.simulate_EIF_net_numba(tgrid, V_init, params['tau_m'], params['V_s'], 
                                  params['V_r'], params['V_T'], params['Delta_T'], 
                                  params['T_ref'], mu_vals, sigma_vals, Jmat, d, 
                                  input_cc,randnvals,randnvals_c)
    Spt_dict = {}
    for i_N in range(Nneurons_obs):
        if sp_counts[i_N]>0:
            Spt_dict[i_N] = Spt_dummy[i_N,Spt_dummy[i_N,:]>0]
        else:
            Spt_dict[i_N] = np.array([])
    
    del Spt_dummy
    
    print('network simulation took {dur}s'.format(
          dur=np.round(time.time() - start,2)))
    print('')   
    
    # ESTIMATE PARAMETERS from spike trains -----------------------------------
    start = time.time()
    N = len(Spt_dict.keys())
    args_fixed = (Spt_dict, d, sigma_init, N_tpert, J_bnds, Jmat, params)      
    arg_tuple_list = [(iN, N, args_fixed) for iN in range(N)]    
                                                              
    print('starting estimation using {} parallel processes'.format(N_procs))
    # e.g., one process for each neuron
    print('')
    print('likelihood optimization can take several minutes...')
    pool = multiprocessing.Pool(N_procs)
    if use_fvm:
        result = pool.imap_unordered(im.Jij_estim_wrapper_v1, arg_tuple_list)
    else:
        result = pool.imap_unordered(im.Jij_estim_wrapper_v2, arg_tuple_list)
    #result = map(im.Jij_estim_wrapper_v1, arg_tuple_list)  # single processing
    # Jij_estim_wrapper estimates all synaptic strengths for a given 
    # post-synaptic neuron (with number iN)

    D = {}
    D['J_true'] = Jmat
    D['mu_estim'] = np.zeros(N)*np.nan
    D['sigma_estim'] = np.zeros(N)*np.nan
    D['logl_uncoupled'] = np.zeros(N)*np.nan
    D['J_estim'] = np.zeros((N,N))
    D['logl_coupled'] = np.zeros((N,N))*np.nan
    
    finished = 0 
    for i_N, mu_estim, sigma_estim, logl_uncoupled, \
        J_estim_row, logl_coupled_row in result:
        finished += 1
        print(('{count} of {tot} estimation parts completed').
              format(count=finished, tot=N)) 
        D['mu_estim'][i_N] = mu_estim
        D['sigma_estim'][i_N] = sigma_estim
        D['logl_uncoupled'][i_N] = logl_uncoupled
        D['J_estim'][i_N,:] = J_estim_row
        D['logl_coupled'][i_N,:] = logl_coupled_row

    pool.close()
    Pcc = np.corrcoef(np.ravel(D['J_estim']),np.ravel(D['J_true']))
    print('')
    print('estimation took {dur}s, corr.-coeff. = {cc}'.format(
           dur=np.round(time.time() - start,2), cc=Pcc[0,1]) )
    print('')
             
    
    # PLOT --------------------------------------------------------------------
    plt.figure()
    plt.subplot(121)
    plt.imshow(D['J_true'], origin='upper', interpolation='nearest', 
               aspect='auto', vmin=-1.0, vmax=1.0, cmap=plt.cm.jet)
    plt.title('true coupling matrix')
    plt.colorbar()
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(D['J_estim'], origin='upper', interpolation='nearest', 
               aspect='auto', vmin=-1.0, vmax=1.0, cmap=plt.cm.jet)
    plt.title('estimated coupling matrix')
    plt.colorbar()
    plt.axis('off')

    plt.figure()
    ax = plt.subplot()
    idx = np.diag(np.ones(N))<1
    plt.plot(np.ravel(D['J_true'][idx]), np.ravel(D['J_estim'][idx]), 'ok', 
             markersize=4)
    plt.plot([-1, 1], [-1, 1], 'c--')
    plt.xlabel('true coupling strengths (mV)', fontsize=12)
    plt.ylabel('estimated coupling strengths (mV)', fontsize=12)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False) 
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
