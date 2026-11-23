import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from po4ao_common.config import make_config

config = make_config({
    'MDP': {
        'n_history': 8,             # 64 frames of 2 x 76 x 76 would be 128 full-resolution channels per frame: too slow at kHz rates
        'obs_key': 'pix',
        'obs_gain': 0.1,            # WFS pixel values * obs_gain should be of the order of the DM-space residual (~0.03)
    },
    'replay_buffers': {
        'obs_half_precision': True, # 2 buffers x 20 episodes x 1000 frames x 2 x 76 x 76: 0.9 GB in float16, 1.8 GB in float32
    },
    'pix': {
        'dm_response': 'interaction_matrix',   # 'interaction_matrix': calibrated response with learned optical gain and delay
                                               # 'learned': the network sees the actions and learns the response
        'delay': 2,                 # initial guess of the loop delay in frames (learned from there)
        'max_lag': 3,
        'recon_weight': 10.0,       # dynamics training: weight of the prediction error measured after reconstruction
        'integrator_skip': True,   # policy = integrator on the reconstructed newest image + network (starting at zero)
        'reg_scale': 15.0 / 17.0,   # pupil diameter / extent of the DM grid: 15 of the 17 actuator pitches span the pupil
        'learn_registration': True,
        'skip_frames': 2,
        'loss_space': 'recon',      # 'recon': loss on the reconstructed prediction, 'pixel': on the raw predicted image
        'pixel_weight': 1.0,
    },
})
