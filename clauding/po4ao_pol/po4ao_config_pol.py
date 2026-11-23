import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from po4ao_common.config import make_config

config = make_config({
    'MDP': {
        'n_history': 16,            # with the total command as input a short history is enough (64 in the original)
    },
    'pol': {
        'input_mode': 'increments+total',   # residuals + increments + the total command c_{t-1}
                                            # 'totals': residuals + history of total commands (badly conditioned, see models)
        'policy_cmd': True,         # False: the policy keeps the original inputs, only the dynamics sees the total command
        'linear_skip': True,        # output = Linear(inputs) + CNN(inputs), see po4ao_models_pol.py
        'linear_kernel': 3,
        'cmd_gain': 1.0,            # scale applied to the command channels before the networks
    },
})
