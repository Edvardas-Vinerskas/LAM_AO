# %%
%load_ext autoreload
%autoreload 2

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import sys
repo = '/home/evinerskas/LAM_OOPAO'
if repo not in sys.path:
    sys.path.insert(0, repo)




# %%
from PAPYRIIS_2stage_CNN_RL.OOPAO_PAPYRIIS_env import OOPAO_environment_PAPYRIIS


PAPYRIIS_env = OOPAO_environment_PAPYRIIS()

# %%

#TODO create a separate control theory project with all of theoretical control functions starting with leaky integrator
        #this could also be used for wind fitting and stuff like that
#TODO do an algorithm that can separate the wind layers (from SPIE poster)
#TODO in the future, you should compare all of the control algorithms against OMGI only
#TODO implement physics constraints into PO4AO
#TODO how to analyse CNNs
#TODO RAMA
#TODO it would be nice to rewrite all of the plotting functions so it is not so messy...
#TODO small chapter on integrator stability criterion in theory
#TODO estimate the optical gains (how to calculate them, use theory established for interaction matrices)
#TODO reformat all of your projects (once again ask claude)
#TODO implement the faster version of PO4AO on RAMA
#TODO how much can one reconstruct from 64 residuals frames?
    #if the correction is very good, you have less residual
    #how are you predicting in that case?
#TODO you can do a test where you change the t_boiling mid run and see how RL adapts 
    #compared to integrator and OMGI
    #this in theory would robustify the RL controller for onsky operations
#TODO you should give pseudo open loop to PO4AO innit
    #training PO4AO on pure noise?
    #remember that networks prefer frequencies with equal power (flat spectrum)
        #then you can finetune it further

#TODO check how atmosphere is mathematically simulated and incorporate into PINN
#TODO for your loss function inside simulations you can also calculate the loss value based on the real leftover residual
    #for this you would also need the gradient to flow from the residual?
#TODO do a package for all of functions for data analysis
#TODO how would linear prediction happen analitically?
    #you just shift the pixels and the new edge values is just new data that is NOT predicted correct?

#TODO discuss pixels4AO and bootstrapping
#TODO theoretical formulas for prediction limitation
#TODO in simulations calculate the expected error due to prediction
#TODO A differentiable version of OOPAO

# %%

#TODO summarise all of the results so far in your slides!!!!
    #TODO speak with claude whether boiling can be predicted in real life situations
        #reading papers here would be good (about boiling and prediction in difficult situations)


#TODO do a summary of first year notes from notebooks, notes app and your ideas in general
#TODO write goals for the week and write goals for the day!!!


# %%
from functions import *

loaddir = 'bench_sky_04_15/arcturus_v7'
RL_telemetry_1st = np.load(f'{loaddir}/2026-04-16T01_19_12_telemetry_data_RLiter50.npy', allow_pickle = True)
CL1st_OL_2nd     = np.load(f'{loaddir}/2026-04-16T01_27_23_telemetry_2nd_data_CL1OL2_noCRED2.npy', allow_pickle = True)


loaddir_sim = 'PAPYRIIS_2stage_CNN_RL/2026-09-23/PAPYRIIS_arcturus_v2_pushpull/results_2nd_stage_CL1OL2_r0_0.070_V0_4.049_L0_30.000_tboil_5.000_multi_layer.npz'
xxxx = np.load(f'{loaddir_sim}')

print(xxxx.files)
strehl = xxxx['all_2nd_stage_strehl']

def strehl_running_avg(strehl_array, kernel_size):

    strehl_mean = strehl_array.mean()

    kernel = np.ones(kernel_size) / kernel_size
    pad_left = len(kernel) // 2
    pad_right = len(kernel) - pad_left - 1

    strehl_padded = np.pad(strehl_array, (pad_left, pad_right), mode='constant', constant_values=strehl_mean)
    strehl_running = np.convolve(strehl_padded, kernel, mode='valid')
    
    return strehl_running

print(strehl.mean())
plt.figure()
plt.plot(strehl_running_avg(strehl, 10000))


# on sky 
# %%




