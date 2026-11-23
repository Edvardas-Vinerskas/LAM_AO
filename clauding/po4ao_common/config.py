"""Default configuration, same layout as po4ao_config_PAPYRUS.py. The variants copy it and override entries."""

import copy

default_config = {
    'RL': {
        'iterations': 20,               # policy episodes (200 on the bench)
        'episode_length': 1000,
        'warmup_episodes': 10,
        'max_sigma': 0.01,              # exploration noise at the start of the warm-up (command units)
        'min_sigma': 0,
        'loss_function_penalty': 0.1,
    },
    'training': {
        'dynamics_grad_steps': 40,
        'policy_grad_steps': 20,
        'dynamics_grad_steps_warmup': 400,
        'policy_grad_steps_warmup': 200,
        # > 1 trains the dynamics on its own multi-step predictions. Leave it at 1 on closed-loop data: the
        # recorded actions are a function of the true observations (a = gain * o in the warm-up), so in a
        # multi-step roll-out the model learns to read the true observation back from the action instead of
        # trusting its own prediction, and its response to an action comes out wrong (dynamics_benchmark.py).
        'dynamics_rollout': 1,
        'lr_dynamics': 1e-3,
        'lr_policy': 1e-3,
        'dynamics_grad_clip': 0.5,
        'policy_grad_clip': None,       # the original does not clip the policy gradient
    },
    'MDP': {
        'n_history': 64,
        'planning_horizon': 4,
        'data_shape': 17,               # DM grid
        'obs_key': 'dm',                # 'dm': reconstructed residual on the DM grid, 'pix': WFS image
        'obs_gain': 1.0,                # multiplies the observation (to bring pixel values to the command scale)
    },
    'replay_buffers': {
        'replay_size': 20,              # episodes
        'warmup_memory': 20,
        'train_warmup_percent': 0.2,
        'obs_half_precision': False,    # store observations in float16 (pixel space)
    },
    'integrator': {
        'gain': 0.4,                    # warm-up only
        'leak': 0.99,                   # single source: the bench/simulator and the planning use the same value
        'command_clamp': 0.8,
        'n_modes': 195,
        'integrator': False,            # use the integrator as the policy
    },
    'NN_models': {
        'filters_per_layer': 32,
        'training_batch': 16,
        'n_ensemble': 5,
        'action_clamp': 1.0,
    },
}


def make_config(overrides=None):
    cfg = copy.deepcopy(default_config)
    for section, entries in (overrides or {}).items():
        cfg.setdefault(section, {}).update(entries)
    return cfg
