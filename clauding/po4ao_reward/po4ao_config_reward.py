import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from po4ao_common.config import make_config

config = make_config({
    'MDP': {
        'n_history': 16,            # the original networks; 64 on PAPYRUS, 16 here to keep the comparison runs short
    },
    'reward': {
        'kind': 'l2',               # 'l2', 'log', 'log_modal', 'log_pixel', 'strehl', 'huber', 'modal'  (po4ao_rewards.py)
        'eps': 1e-4,                # noise floor of the squared residual, in command units^2
        'action_penalty': 'increment',   # 'increment', 'smooth', 'command', 'none'
        'pessimism': 0.0,           # weight of the ensemble-variance penalty
        'discount': 1.0,
        'strehl_scale': 1.0,        # set by the run script from the simulator; on the bench: (rad rms per command unit)^2
    },
})
