# -*- coding: utf-8 -*-
'''
example script for estimation of coupling strengths and delays of a (possibly 
subsampled) network of leaky or exponential I&F neurons using method 1a, 
cf. Ladenbauer et al. 2019 (Results section 4)
-- written by Josef Ladenbauer in 2018/2019 

run time was <15 min on an Intel i7-2600 quad-core PC using Python 2.7 
(Anaconda distribution v. 5.3.0) 
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
Nneurons_tot = 6  # number of neurons in total
# note that the simulation code here is not optimized for larger networks 
# (i.e., Nneurons_tot > about 50), for that purpose it is recommended to use 
# simulation tools for spiking networks, e.g. the Python-based package Brian2 

params = dict()
# neuron model parameters:
params['tau_m'] = 20.0  # ms, membrane time constant
params['V_r'] = 0.0  # mV, reset voltage
params['V_s'] = 30.0  # mV, spike voltage
params['Delta_T'] = 0.0  # mV, threshold slope factor (set 0 for LIF model, 
                         #                                >0 for EIF model)  
params['V_T'] = 15.0  # mV, effective threshold voltage (only used for EIF)
params['T_ref'] = 0.0  # ms, refractory duration

# input parameters:
np.random.seed(20)
mu_vals = 1.75 * (0.8 + 0.4*np.random.rand(Nneurons_tot))  # mV/ms, input mean
sigma_vals = 2.5 * (0.8 + 0.4*np.random.rand(Nneurons_tot))  # mV/sqrt(ms), input
                                                             # standard deviation
                                                             
# parameters for data generation:
Jmat = 1.5*np.random.rand(Nneurons_tot,Nneurons_tot) - 0.75  # coupling strengths
Jmat -= np.diag(np.diag(Jmat))  # exclude autapses
delay = 1.0  # ms
input_cc = 0.0  # determines the correlation strength of external input fluctuations 
                # for each pair in the network 
t_end = 30e4  # ms, simulation duration ("recording time")
params['dt_sim'] = 0.05  # ms, simulation time step
    

# parameters for inference method
params['pISI_method'] = 'fvm' # numerical scheme based on finite volume method
#params['pISI_method'] = 'fourier' # numerical scheme based on Fourier transform
params['V_lb'] = -80.0  # mV, lower bound
if params['pISI_method'] == 'fvm': 
    params['neuron_model'] = 'LIF'
    params['integration_method'] = 'implicit'
    params['N_centers_fvm'] = 1000  # number of centers for voltage discretization
    params['fvm_v_init'] = 'delta'  # voltage density initialization
    params['fvm_delta_peak'] = params['V_r']  # location of initial density peak
    params['fvm_dt'] = 0.1  # ms, time step for finite volume method
                            # 0.1 seems ok, otherwise use, e.g., 0.05 ms
    # the next 3 lines are only required for im.find_mu_init, which uses a rapid 
    # computation of the mean ISI from the model
    d_V = 0.025  # mV, spacing of voltage grid
    params['V_vals'] = np.arange(params['V_lb'],params['V_s']+d_V/2,d_V)
    params['V_r_idx'] = np.argmin(np.abs(params['V_vals']-params['V_r'])) 
                        # index of reset voltage on grid, this should be a grid point
elif params['pISI_method'] == 'fourier':                         
    f_max = 2000.0 # Hz, determines resolution (accuracy) of ISI density
    d_freq = 0.25 # Hz, spacing of frequency grid
    d_V = 0.025  # mV, spacing of voltage grid
    params['V_vals'] = np.arange(params['V_lb'],params['V_s']+d_V/2,d_V)
    params['freq_vals'] = np.arange(0.0, f_max+d_freq/2, d_freq)/1000  # kHz
    params['V_r_idx'] = np.argmin(np.abs(params['V_vals']-params['V_r'])) 
                        # index of reset voltage on grid, this should be a grid point
params['ISI_min'] = 3.0  # ms                 
N_tpert = 500  # determines spacing of (potential) perturbation times within ISIs
               # due to coupling
params['d_grid'] = np.arange(0.5, 3, 0.5) # ms, delay values for optimization   
params['J_bnds'] = [-2, 2] # mV, minimal/maximal coupling strengths
params['J_init'] = np.array([0.25]) # mV, initial value for each estimated connection;
# each connection is estimated once for each value in this array, the estimate 
# according to the max. likelihood is returned 
sigma_init = 3.0  # initial sigma value (initial mu value will be determined by 
                  # sigma_init and empirical mean ISI)

# pseudo data by jittering presyn. spike times in order to assess bias in estimates
# of J and to compute z-scores
params['N_pseudo'] = 20  # 0 for no pseudo data 
params['pseudo_jitter_bnds'] = [-10, 10]  # ms

N_procs = min([int(5.0*multiprocessing.cpu_count()/6), 10])  # num. of parallel processes


if __name__ == '__main__':
    
    # GENERATE SPIKE TRAINS from network model --------------------------------
    tgrid = np.arange(0, t_end+params['dt_sim']/2, params['dt_sim'])
            # time points for simulation 

    V0_vals = params['V_r'] * np.ones(Nneurons_tot)

    start = time.time() 
    randnvals = np.random.randn(Nneurons_tot,len(tgrid))
    randnvals_c = np.random.randn(len(tgrid)) # common noise
    print('')
    print 'starting network simulation ...' 
    Spt_dummy, sp_counts, _ = \
        im.simulate_EIF_net_numba(tgrid, V0_vals, params['tau_m'], params['V_s'], 
                                  params['V_r'], params['V_T'], params['Delta_T'], 
                                  params['T_ref'], mu_vals, sigma_vals, Jmat, delay, 
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
    args_fixed = (Spt_dict, sigma_init, N_tpert, params)      
    arg_tuple_list = [(iN, N, args_fixed) for iN in range(N)]                                                           
    print('starting I&F estimation using {} parallel processes'.format(N_procs))
    # e.g., one process for each neuron
    print('')
    print('likelihood optimization can take several minutes...')
    pool = multiprocessing.Pool(N_procs)
    result = pool.imap_unordered(im.Jdij_estim_wrapper, arg_tuple_list)
    # Jdij_estim_wrapper estimates all synaptic strengths for a given 
    # postsynaptic neuron (with number iN)
    #result = map(im.Jdij_estim_wrapper, arg_tuple_list)  # single process for debugging
                        
    D = {}
    D['J_true'] = Jmat
    D['mu_estim'] = np.zeros(N)*np.nan
    D['sigma_estim'] = np.zeros(N)*np.nan
    D['loglike_uncoupled'] = np.zeros(N)*np.nan
    D['J_estim'] = np.zeros((N,N))
    D['J_z_estim'] = np.zeros((N,N))
    D['J_bias_estim'] = np.zeros((N,N))
    D['d_estim'] = np.zeros((N,N))
    D['loglike_coupled'] = np.zeros((N,N))*np.nan
                    
    finished = 0 
    for i_N, mu_estim, sigma_estim, logl_uncoupled, J_estim_row, J_estim_bias_row, \
        J_estim_z_row, d_estim_row, logl_coupled_row in result:
        finished += 1
        print(('{count} of {tot} estimation parts completed').
              format(count=finished, tot=N)) 
        D['mu_estim'][i_N] = mu_estim
        D['sigma_estim'][i_N] = sigma_estim
        D['loglike_uncoupled'][i_N] = logl_uncoupled
        D['J_estim'][i_N,:] = J_estim_row
        D['J_bias_estim'][i_N,:] = J_estim_bias_row
        D['J_z_estim'][i_N,:] = J_estim_z_row
        D['d_estim'][i_N,:] = d_estim_row
        D['loglike_coupled'][i_N,:] = logl_coupled_row
    pool.close()
    print('')
    print('estimation took {dur}s'.format(dur=np.round(time.time() - start,2)))
    print('')                   
                    

    # PLOT --------------------------------------------------------------------             
    idx = np.diag(np.ones(N))<1
    J_estim_vals = np.ravel(D['J_estim'][idx])
    biases = np.ravel(D['J_bias_estim'][idx])
    J_estim_vals -= np.mean(biases)
    J_true_vals = np.ravel(D['J_true'][idx])
    d_estim_vals = np.ravel(D['d_estim'][idx])
    mae_JestJtrue = np.mean(np.abs(J_estim_vals-J_true_vals))
    Pcc = np.corrcoef(J_estim_vals, J_true_vals)
    rho_JestJtrue = Pcc[0,1]
    mae_destdtrue = np.mean(np.abs(d_estim_vals-delay))
     
    plt.figure()       
    ax = plt.subplot(121)
    plt.plot(J_true_vals, J_estim_vals, 
            'ok', alpha=0.5, markersize=6, markeredgewidth=0)
    plt.plot([-0.8, 0.8], [-0.8, 0.8], '--', color='gray')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.xticks([-1, 0, 1])
    plt.yticks([-1, 0, 1])
    string = r'$ \varrho ={r}$, '.format(r=round(np.mean(rho_JestJtrue),2)) \
             + r'MAE$={}$~mV'.format(round(np.mean(mae_JestJtrue),3))
    plt.xlabel('true coupling strengths (mV)', fontsize=12)
    plt.ylabel('estimated coupling \n strengths (mV)', fontsize=12)
    plt.title(string, fontsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False) 
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
    ax = plt.subplot(122)    
    binedges = np.arange(0.25,2.251,0.5)
    plt.hist(d_estim_vals, bins=binedges, color='gray')
    plt.xlabel('estimated delays (ms)', fontsize=14)
    string = 'true delay = {}~ms \n'.format(delay) + r'MAE$={}$~ms'.format(
              round(np.mean(mae_destdtrue),3))
    plt.title(string, fontsize=14)
    plt.xlim([-0.5,6.5])
    plt.xticks(range(7))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False) 
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')