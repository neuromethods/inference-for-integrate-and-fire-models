# -*- coding: utf-8 -*-
'''
example script for estimation of input parameters (mean mu, standard dev. sigma)
of a leaky or exponential I&F neuron (LIF/EIF) subject to fluctuating inputs 
using method 1a, cf. Ladenbauer & Ostojic 2018 (Results section 2, Fig 1A)
-- written by Josef Ladenbauer in 2018 

run time was 14 s on an Intel i7-2600 quad-core PC using Python 2.7 
(Anaconda distribution v. 5.0.1) 
'''

import inference_methods as im
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import scipy.optimize
matplotlib.rc('text', usetex=True)
matplotlib.rc('xtick', labelsize=12) 
matplotlib.rc('ytick', labelsize=12) 
                                        
# SET PARAMETERS --------------------------------------------------------------
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
mu_true = -1.75  # mV/ms, input mean
sigma_true = 2.5 # mV/sqrt(ms), input standard deviation (noise intensity)

# parameters for data generation:
params['dt_sim'] = 0.05  # ms, simulation time step
N_spk = 400  # number of spikes used for estimation
t_limits = [100, 1100] # ms, for plots
V_init = params['V_r']  # initial condition
np.random.seed(12)

# parameters for calculation of likelihood (method 1a)
f_max = 1000.0 # Hz, determines resolution (accuracy) of ISI density; 1k seems
               # sufficient in many cases, for finer resolution try 2k or 4k
d_freq = 0.25 # Hz, spacing of frequency grid
d_V = 0.01  # mV, spacing of voltage grid
params['V_lb'] = -150.0  # mV, lower bound
params['V_vals'] = np.arange(params['V_lb'],params['V_s']+d_V/2,d_V)
params['freq_vals'] = np.arange(0.0, f_max+d_freq/2, d_freq)/1000  # kHz
params['V_r_idx'] = np.argmin(np.abs(params['V_vals']-params['V_r'])) 
                    # index of reset voltage on grid, this should be a grid point
sigma_init = 3.0  # initial sigma value (within reasonable range; initial mu
                  # value will be determined by sigma_init and empirical mean ISI)
#sigma_init = 1.5 + 4.5*np.random.rand()  # e.g., randomized
    
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
    mu_init = im.find_mu_init(ISImean, args=(params,sigma_init))  
              # determine initial mu value by matching the empirical mean ISI  
              # for given sigma using the Fokker-Planck system for LIF/EIF 
              # subject to white noise input (to calc. the steady-state spike rate)
    print('mu_init = {}'.format(np.round(mu_init,2)))
    init_vals = np.array([mu_init, sigma_init])
    
    sol = scipy.optimize.minimize(im.spiketrain_likel_musig, init_vals, 
                                  args=(params, ISIs, ISImean, ISImax), 
                                  method='nelder-mead', 
                                  options={'xatol':0.025, 'fatol':0.05})
    print('')
    print sol  
    print('')
    print('likelihood optimization took {dur}s'.format(
           dur=np.round(time.time() - start,2)))
    
    mu_estim, sigma_estim = sol.x  

    # COMPUTE ISI density, membrane voltage density for estimated parameters --
    args = (params, ISIs, ISImean, ISImax, mu_estim, sigma_estim)
    lval, loglval, pISI_times, pISI_vals = im.calc_spiketrain_likelihood(args)    
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
    plt.plot([50, 50], [-80, -60], 'k', linewidth=2)
    plt.plot([50, 250], [-80, -80], 'k',  linewidth=2)
    plt.title('Membrane voltage and observed spike times', fontsize=16)
    plt.ylim([-82, -35])
    plt.axis('off')

    ax = plt.subplot2grid((1,7), (0,6))
    width = 1.0 # mV, for voltage histogram
    bins = np.arange(params['V_vals'][0], params['V_vals'][-1], width) 
    tend_example = Spt_obs[-1]
    Vhist, binedges = np.histogram(V_trace[tgrid<tend_example], bins=bins, 
                                   density=True)
    ax.barh(binedges[:-1], Vhist, height=binedges[1]-binedges[0], color='k', 
            alpha=0.4, linewidth=0)  # voltage histogram       
    ax.plot(p_ss[params['V_vals']>=-80], params['V_vals'][params['V_vals']>=-80], 
            'g', linewidth=2)  # membrane voltage density from estim. parameters
    ax.set_ylim([-82, -35])
    ax.set_xlim([-0.003, 0.05])
    ax.set_axis_off()