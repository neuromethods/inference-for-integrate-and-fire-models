# -*- coding: utf-8 -*-
'''
example script for estimation of input perturbations of a leaky or exponential 
I&F neuron (LIF/EIF) subject to fluctuating inputs using method 2, 
cf. Ladenbauer & Ostojic 2018 (Results section 3, Fig 2A)
-- written by Josef Ladenbauer in 2018 

run time was about 2 min. on an Intel i7-2600 quad-core PC using Python 2.7 
(Anaconda distribution v. 5.0.1) 
'''

import inference_methods as im
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import scipy.optimize
import tables
from collections import OrderedDict
import os
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
mu = -1.75  # mV/ms, input mean
sigma = 2.5 # mV/sqrt(ms), input standard deviation (noise intensity)
# mean input perturbations are given by superposed alpha functions triggered at 
# known times
tau_true = 10.0  # ms, time constant of alpha function
J_true = 0.3  # mV/ms, peak value of mean input perturbation mu1(t)
d_true = 0.0  # ms, delay (estimation of delay not included here)

# note that when estimating the background parameters (mu and sigma) here it is 
# suggested to use method 1a and then re-estimate mu using method 2 with estimated 
# sigma before switching to method 2 for estimating the input perturbations

# parameters for data generation:
params['dt_sim'] = 0.05  # ms, simulation time step
t_start = 0.0  # ms
t_end = 1e5  # ms
t_limits = [5000, 6000]  # ms, for plots
V_init = params['V_r']  # initial condition    
np.random.seed(20)
# perturbation times v.1 (Gaussian distributed inter pert. intervals)
IPIs = 200.0 + 50.0*np.random.randn(10000)  
IPIs = IPIs[IPIs>0]
pert_times = np.cumsum(IPIs)
# perturbation times v.2 (Poisson-like with "refractory" duration)
#dp = 10.0  # ms
#mean_IPI = 90.0  # ms, for actual mean add dp
#IPIs = np.random.exponential(mean_IPI, 10000) + dp
#pert_times = np.cumsum(IPIs)

# parameters for calculation of likelihood (method 2)       
f_max = 1000.0 # Hz, frequency spacing for calculation of first order rate response,
               # which is needed for the derived spike rate model (LNexp);
               # 1k seems sufficient
d_freq = 0.25 # Hz, spacing of frequency grid
d_V = 0.005  # mV, spacing of voltage grid for calculations of steady-state and 
             # first order rate response (note: Vr should be a grid point)
params['V_lb'] = -150.0  # mV, lower bound
params['V_vals'] = np.arange(params['V_lb'],params['V_s']+d_V/2,d_V)
params['freq_vals'] = np.arange(d_freq, f_max+d_freq/2, d_freq)/1000  # kHz
params['V_r_idx'] = np.argmin(np.abs(params['V_vals']-params['V_r'])) 
                    # index of reset voltage on grid, this should be a grid point
params['d_mu'] = 1e-5 # mV/ms
params['d_sigma'] = 1e-5 # mV/sqrt(ms)

precalc_filename = 'quantities_cascade.h5' # pre-calc. quantities for derived 
                                           # spike rate model will be stored here 
precalc_folder = os.path.dirname(os.path.realpath(__file__)) 
                 # stores precalc files in the same directory as the script itself
J_init = 0.5*J_true + J_true*np.random.rand()  # initial J value
tau_init = 0.5*tau_true + tau_true*np.random.rand()  # initial tau value


if __name__ == '__main__':
    
    # GENERATE SPIKE TRAIN from neuron model ----------------------------------
    tgrid = np.arange(t_start, t_end+params['dt_sim']/2, params['dt_sim'])
            # time points for simulation
    
    rand_vec = np.random.randn(len(tgrid))
    pert_times = pert_times[pert_times<=tgrid[-1]]
        
    start = time.time()

    V_trace, mu_pert_trace, Sp_times = \
            im.simulate_EIF_pert_numba(tgrid, V_init, params['tau_m'], 
                                       params['V_s'], params['V_r'], params['V_T'],
                                       params['Delta_T'], params['T_ref'],
                                       mu, sigma, pert_times, J_true, tau_true, 
                                       d_true, rand_vec)
    print('')
    print('simulation took {dur}s'.format(dur=np.round(time.time() - start,2)))
    print('')
    
    # for plots (further below)
    inds = (tgrid>=t_limits[0]) & (tgrid<=t_limits[1])
    V_trace = V_trace[inds]
    mu_pert_trace = mu_pert_trace[inds]
    
    ISIs = np.diff(Sp_times)
    lastSpidx = [len(Sp_times)-1]  
    ISImin = np.min(ISIs) 
    ISImax = np.max(ISIs)
    # extreme small or large ISIs in biological spike trains may cause problems, 
    # for robust estimation, e.g., omit the smallest and largest 5 %
    #ISImin = np.percentile(ISIs, 2.5)  
    #ISImax = np.percentile(ISIs, 97.5)
        
        
    # ESTIMATE PARAMETERS from spike train ------------------------------------
    # pre-calculation of look-up quantities (time constant tau_mu, steady-state
    # spike rate r_ss) for the LNexp spike rate model derived from I&F neurons 
    # with white noise input, cf. Methods section 3 
    try:
        h5file = tables.open_file(precalc_folder+'/'+precalc_filename, mode='r')
        sigma_vals = np.array(h5file.root.sigma_vals)
        h5file.close()
        if np.abs(sigma-sigma_vals[0])>0.01:  
            precalculate = True
        else:
            precalculate = False
    except:
        precalculate = True
            
    if precalculate:   
        mu_vals = np.arange(mu-1.5, mu+1.5, 0.05)  
        sigma_vals = np.array([sigma])
        EIF_output_dict = OrderedDict()
        LN_quantities_dict = OrderedDict()
        EIF_output_names = ['r_ss', 'dr_ss_dmu', 'r1_mumod'] 
        LN_quantity_names = ['r_ss', 'tau_mu_exp']    
        params['N_procs'] = 1  # multi-proc. for len(sigma_vals)==1 is not included
        print('computing {}'.format(EIF_output_names))
        print('this may take a while for large numbers of mu & sigma values...')              
        EIF_output_dict, LN_quantities_dict = im.calc_EIF_output_and_cascade_quants(
                                                 mu_vals, sigma_vals, params, 
                                                 EIF_output_dict, EIF_output_names,
                                                 LN_quantities_dict, LN_quantity_names) 
        im.save(precalc_folder+'/'+precalc_filename, LN_quantities_dict, params)
        im.plot_quantities(LN_quantities_dict, LN_quantity_names, sigma_vals)
        print('LNexp quantities saved')
                                  
    dt = params['dt_sim']  # use same simulation time step as used for data 
                           # generation (not required)
    h5file = tables.open_file(precalc_folder+'/'+precalc_filename, mode='r')
    mu_vals = np.array(h5file.root.mu_vals)
    sigma_vals = np.array(h5file.root.sigma_vals)
    if np.abs(sigma-sigma_vals[0])>0.01 or mu>=mu_vals[-1] or mu<=mu_vals[0]:
        print ''
        print 'WARNING: precalc data does not seem to fit!'
        print 'mu-limits:', mu_vals[0], mu_vals[-1]
        print 'sigma:', sigma_vals
        print ''
    r_ss_array = np.array(h5file.root.r_ss)
    tau_mu_array = np.array(h5file.root.tau_mu_exp)
    tau_mu_array[tau_mu_array < dt] = dt
    h5file.close()
    
    # likelihood optimization using the derived (LNexp) spike rate model 
    start = time.time()
    args_fixed = (mu, pert_times, Sp_times, lastSpidx, d_true, tgrid,
                  mu_vals, r_ss_array, tau_mu_array, ISImin, ISImax)
    init_vals = np.array([J_init, tau_init])
    sol = scipy.optimize.minimize(im.spiketrain_likel_alpha, init_vals, 
                                  args=args_fixed, method='nelder-mead', 
                                  options={'xatol':0.01, 'fatol':0.005})
    print('')
    print sol  
    print('')
    print('likelihood optimization took {dur}s'.format(
          dur=np.round(time.time() - start,2)))

    J_estim = sol.x[0]
    tau_estim = sol.x[1]
    loglike = -1*sol.fun
    
          
    # PLOT -------------------------------------------------------------------- 
    plt.figure()
    ax = plt.subplot(212)
    # generate estimated mean input perturbation time series mu_1(t)
    tgrid = tgrid[inds]
    pert_times = pert_times[(pert_times>=tgrid[0]) & (pert_times<=tgrid[-1])]
    mu_pert_trace_estim = im.sim_only_mu_perturbation(tgrid, pert_times, 
                                                      J_estim, tau_estim)
    plt.plot(tgrid, mu+mu_pert_trace_estim, 'b')  # estimation 
    plt.plot(tgrid, mu+mu_pert_trace, 'c--')  # ground truth
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False) 
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.title('True and recovered mean input time series', fontsize=16)   
    plt.axis('off')
    
    plt.subplot(211)
    plt.plot(tgrid, V_trace, color=(0.6,0.6,0.6))
    # visualize spike times
    Spt_plot = Sp_times[(Sp_times>=t_limits[0]) & (Sp_times<=t_limits[1])]
    for i in range(len(Spt_plot)):
        plt.plot([Spt_plot[i], Spt_plot[i]], [params['V_s'], params['V_s']+2.5], 
                 'k', linewidth=2)
    plt.plot([t_limits[0]-50, t_limits[0]-50], [-80, -60], 'k', linewidth=2)
    plt.plot([t_limits[0]-50, t_limits[0]+150], [-80, -80], 'k',  linewidth=2)
    plt.title('Membrane voltage and observed spike times', fontsize=16)
    plt.ylim([-82, -35])
    plt.axis('off')