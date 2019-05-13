# -*- coding: utf-8 -*-
'''
example script for estimation of input parameters (mean mu, standard dev. sigma)
of a leaky or exponential I&F neuron (LIF/EIF) subject to fluctuating inputs 
using method 1, cf. Ladenbauer et al. 2019 (Results section 2)
-- written by Josef Ladenbauer in 2018/2019 

run time was <5 s on an Intel i7-8550U Laptop using Python 3.7 (Anaconda distrib.
v. 2019.03; also tested using Python 2.7, Anaconda distrib. v. 5.3.0)
'''

import inference_methods as im
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import scipy.optimize
#matplotlib.rc('text', usetex=True)
matplotlib.rc('xtick', labelsize=12) 
matplotlib.rc('ytick', labelsize=12) 
                                        
# SET PARAMETERS --------------------------------------------------------------
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
mu_true = 1.75  # mV/ms, input mean
sigma_true = 2.5 # mV/sqrt(ms), input standard deviation (noise intensity)

estimate_taum = False  # including taum in the estimation takes longer (~30 s)

# parameters for data generation:
params['dt_sim'] = 0.05  # ms, simulation time step
N_spk = 400  # number of spikes used for estimation
t_limits = [100, 1100] # ms, for plots
V_init = params['V_r']  # initial condition
np.random.seed(12)
           
# parameters for calculation of likelihood
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
    f_max = 1000.0 # Hz, determines resolution (accuracy) of ISI density; 1k seems
                   # sufficient in many cases, for finer resolution try 2k or 4k
    d_freq = 0.25 # Hz, spacing of frequency grid
    d_V = 0.01  # mV, spacing of voltage grid
    params['V_vals'] = np.arange(params['V_lb'],params['V_s']+d_V/2,d_V)
    params['freq_vals'] = np.arange(0.0, f_max+d_freq/2, d_freq)/1000  # kHz
    params['V_r_idx'] = np.argmin(np.abs(params['V_vals']-params['V_r'])) 
                        # index of reset voltage on grid, this should be a grid point
                                
sigma_init = 3.0  # initial sigma value (within reasonable range; initial mu
                  # value will be determined by sigma_init and empirical mean ISI)
#sigma_init = 1.5 + (sigma_true-1.5)*2*np.random.rand()  # e.g., randomized
                  
print('')
print('baseline spike rate should be > ~4 Hz on average (mean ISI < ~250ms)')
print('for current numerics parameters')
print('')


if __name__ == '__main__':
     
    # GENERATE SPIKE TRAIN from neuron model ----------------------------------
    tgrid = np.arange(0, 1e5+params['dt_sim']/2, params['dt_sim'])  
            # time points for simulation
    mu_vec = mu_true*np.ones(len(tgrid))
    sigma_vec = sigma_true*np.ones(len(tgrid))
    rand_vec = np.random.randn(len(tgrid))  # random numbers for input fluctuations
    
    start = time.time()  # for timing
    
    V_trace, Sp_times = im.simulate_EIF_numba(tgrid, V_init, params['tau_m'], 
                                        params['V_s'], params['V_r'], params['V_T'],
                                        params['Delta_T'], params['T_ref'],
                                        mu_vec, sigma_vec, rand_vec)
    
    print('simulation took {dur}s'.format(dur=np.round(time.time() - start,2)))
    print('')
    
    Spt_obs = Sp_times[:N_spk]
    ISIs = np.diff(Spt_obs)  # data for parameter estimation
    # extreme small or large ISIs in biological spike trains may cause problems, 
    # for robust estimation, e.g., omit the smallest and largest 5 %
    # and make sure all ISIs are larger than some reasonable minimum value:
    #ISIs_sorted = np.sort(ISIs)
    #ISIs_cropped = ISIs_sorted[int(0.025*len(ISIs)):int(0.975*len(ISIs))]
    #ISIs_cropped = ISIs_cropped[ISIs_cropped>2.5]
    #ISIs = ISIs_cropped.copy()
    
    ISImax = 1.1*np.max(ISIs)
    ISImean = np.mean(ISIs)
        
    # ESTIMATE PARAMETERS from spike train ------------------------------------
    start = time.time()
    if not estimate_taum:
        mu_init = im.find_mu_init(ISImean, args=(params,sigma_init))  
                  # determine initial mu value by matching the empirical mean ISI  
                  # for given sigma using the Fokker-Planck system for LIF/EIF 
                  # subject to white noise input (to calc. the steady-state spike rate)
        print('mu_init = {}'.format(np.round(mu_init,2)))
        init_vals = np.array([mu_init, sigma_init])
        
        sol = scipy.optimize.minimize(im.spiketrain_likel_musig, init_vals, 
                                      args=(params, ISIs, ISImean, ISImax), 
                                      method='nelder-mead', 
                                      options={'xatol':0.01, 'fatol':0.01})   
        print('')
        print(sol)
        print('')    
        mu_estim, sigma_estim = sol.x  
        
    else:
        # tau_m can be easily included in the estimation as follows:
        taum_init = 10.0  # ms, initial guess
        params_tmp = params.copy()
        params_tmp['tau_m'] = taum_init
        mu_init = im.find_mu_init(ISImean, args=(params_tmp,sigma_init))
        init_vals = np.array([mu_init, sigma_init, taum_init])
        sol = scipy.optimize.minimize(im.spiketrain_likel_musigtaum, init_vals, 
                                      args=(params, ISIs, ISImean, ISImax), 
                                      method='nelder-mead', 
                                      options={'xatol':0.01, 'fatol':0.01})
        print('')
        print(sol)
        print('')
        mu_estim, sigma_estim, taum_estim = sol.x 
    
    print('likelihood optimization took {dur}s'.format(
           dur=np.round(time.time() - start,2)))

    # COMPUTE ISI density, membrane voltage density for estimated parameters --
    args = (params, ISIs, ISImean, ISImax, mu_estim, sigma_estim)
    like, loglike, pISI_times, pISI_vals = im.calc_spiketrain_likelihood(args)    
    p_ss, r_ss, q_ss = im.EIF_steady_state_numba(params['V_vals'], params['V_r_idx'],
                                                 params['tau_m'], params['V_r'],
                                                 params['V_T'], params['Delta_T'], 
                                                 mu_estim, sigma_estim)  
    
    # PLOT --------------------------------------------------------------------
    plt.figure()
    ax = plt.subplot()
    width = 5.0  # ms, for ISI histogram
    bins = np.arange(0.0, ISImax, width) 
    ISIhist, binedges = np.histogram(ISIs, bins=bins)
    ISIhist_normed = ISIhist/(width*len(ISIs))
    plt.bar(binedges[:-1], ISIhist_normed, width=width, 
            color='black', linewidth=0, alpha=0.5, align='edge')  # ISI histogram
    plt.plot(pISI_times, pISI_vals, 'g', linewidth=3)  # pISI from max. likelihood
    plt.xlim([0, ISImax])
    plt.title('ISI histogram for {n_spikes} spikes'.format(n_spikes=N_spk), 
              fontsize=16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False) 
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    ax.xaxis.set_ticks_position('bottom')
    plt.xlabel('Interspike interval (ms)', fontsize=14)
     
    plt.figure()
    inds = (tgrid>=t_limits[0]) & (tgrid<=t_limits[1])
    plt.subplot2grid((1,7), (0,0), colspan=6)            
    plt.plot(tgrid[inds], V_trace[inds], color=(0.6,0.6,0.6))
    # visualize spike times:
    Spt_plot = Sp_times[(Sp_times>=t_limits[0]) & (Sp_times<=t_limits[1])]
    for i in range(len(Spt_plot)):
        plt.plot([Spt_plot[i], Spt_plot[i]], [params['V_s'], params['V_s']+2.5], 
                 'k', linewidth=2)
    plt.plot([50, 50], [-10, 10], 'k', linewidth=2)
    plt.plot([50, 250], [-10, -10], 'k',  linewidth=2)
    plt.title('Membrane voltage and observed spike times', fontsize=16)
    plt.ylim([-12, 40])
    plt.axis('off')

    ax = plt.subplot2grid((1,7), (0,6))
    width = 1.0 # mV, for voltage histogram
    bins = np.arange(params['V_vals'][0], params['V_vals'][-1], width) 
    tend_example = Spt_obs[-1]
    Vhist, binedges = np.histogram(V_trace[tgrid<tend_example], bins=bins, 
                                   density=True)
    ax.barh(binedges[:-1], Vhist, height=binedges[1]-binedges[0], color='k', 
            alpha=0.4, linewidth=0)  # voltage histogram       
    ax.plot(p_ss[params['V_vals']>=-10], params['V_vals'][params['V_vals']>=-10], 
            'g', linewidth=2)  # membrane voltage density from estim. parameters
    ax.set_ylim([-12, 40])
    ax.set_xlim([-0.003, 0.05])
    ax.set_axis_off()

    plt.show()