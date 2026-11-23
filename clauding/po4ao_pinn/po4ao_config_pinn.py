import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from po4ao_common.config import make_config

config = make_config({
    'MDP': {
        'n_history': 8,             # must be >= n_frames + max_lag
    },
    'training': {
        'dynamics_rollout': 1,      # keep 1: see the warning in po4ao_common/config.py
    },
    'pinn': {
        'mode': 'kernel',           # 'kernel': free linear flow kernels, 'wind': explicit wind vector per layer
        'policy': 'conv',           # 'conv': original PO4AO policy (residuals + changes of command),
                                    # 'frozen_flow': predictor on the POL wavefront, 'pol': po4ao_pol policy
        'delay': 2,                 # initial guess of the loop delay in frames (the DM response is learned from there)
        'max_lag': 4,               # the DM response may spread over lags 1..max_lag
        'free_lags': False,         # True: one free 3 x 3 kernel per lag instead of (delay weights) x (one kernel)
        'n_frames': 3,              # POL frames used by the flow operator ('kernel' mode)
        'kernel': 5,                # flow kernel size, in actuators
        'n_layers': 3,              # 'wind' mode
        'residual': True,           # add the small penalised CNN
        'res_filters': 16,
        'res_frames': 4,
        'res_weight': 0.1,          # lambda of the penalty on the CNN output
        'res_causal': True,         # withhold from the CNN the actions that cannot have reached the WFS yet
    },
})
