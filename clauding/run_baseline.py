"""Reference runs on the toy simulator: the plain integrator and the unchanged PO4AO algorithm.

    python run_baseline.py                  # PO4AO, n_history = 64 as on PAPYRUS
    python run_baseline.py --n_history 16
    python run_baseline.py --integrator     # integrator gain scan on the same turbulence

Every variant script (po4ao_pix, po4ao_pol, po4ao_pinn, po4ao_reward) uses the same simulator seed, so
the turbulence is identical and the numbers can be compared episode by episode.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from po4ao_common.config import make_config
from po4ao_common.core import PO4AO, l2_loss
from po4ao_common.models import ConvDynamics, ConvPolicy, Ensemble
from po4ao_common.runner import run_sim, integrator_reference, save_history, summarize
from po4ao_common.toy_env import ToyAOEnv

HERE = os.path.dirname(os.path.abspath(__file__))


def make_env(cfg, seed=0, **kw):
    return ToyAOEnv(n_act=cfg['MDP']['data_shape'], n_modes=cfg['integrator']['n_modes'],
                    leak=cfg['integrator']['leak'], clamp=cfg['integrator']['command_clamp'], seed=seed, **kw)


def build_baseline(env, cfg, loss_fn=l2_loss):
    n, nf = cfg['MDP']['n_history'], cfg['NN_models']['filters_per_layer']
    shape = (env.n_act, env.n_act)
    dynamics = Ensemble(lambda: ConvDynamics(env.mask, n, nf), cfg['NN_models']['n_ensemble'])
    policy = ConvPolicy(env.mask, env.KL_projection, n, nf, cfg['NN_models']['action_clamp'])
    return PO4AO(cfg, dynamics, policy, (1,) + shape, shape, loss_fn, str(env.device))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_history', type=int, default=64)
    ap.add_argument('--iterations', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--integrator', action='store_true')
    ap.add_argument('--net_seed', type=int, default=None,
                    help='seed of the networks and mini-batches only (same turbulence): run-to-run scatter')
    args = ap.parse_args()

    cfg = make_config({'MDP': {'n_history': args.n_history}, 'RL': {'iterations': args.iterations}})
    torch.manual_seed(args.seed if args.net_seed is None else args.net_seed)
    n_ep = cfg['RL']['warmup_episodes'] + cfg['RL']['iterations']

    if args.integrator:
        out = {}
        for gain in [0.4, 0.7, 1.0, 1.1, 1.2, 1.3, 1.4]:
            v = integrator_reference(make_env(cfg, args.seed), n_ep, cfg['RL']['episode_length'], gain)
            out[gain] = v
            print('integrator gain %.1f: residual variance, last 10 episodes %.4f rad^2 (all policy-phase episodes %.4f)'
                  % (gain, np.mean(v[-10:]), np.mean(v[cfg['RL']['warmup_episodes']:])))
        import json
        os.makedirs(os.path.join(HERE, 'runs'), exist_ok=True)
        with open(os.path.join(HERE, 'runs', 'integrator_seed%d.json' % args.seed), 'w') as f:
            json.dump(out, f)
    else:
        env = make_env(cfg, args.seed)
        agent = build_baseline(env, cfg)
        label = 'baseline_n%d%s' % (args.n_history, '' if args.net_seed is None else '_net%d' % args.net_seed)
        history = run_sim(env, agent, cfg, label=label)
        save_history(history, cfg, os.path.join(HERE, 'runs', '%s_seed%d.json' % (label, args.seed)))
        print(label, summarize(history))
