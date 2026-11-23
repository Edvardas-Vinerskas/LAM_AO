"""PO4AO with alternative planning losses. The networks are the original ones; only the loss changes.

    python po4ao_reward.py --kind l2                 # the original
    python po4ao_reward.py --kind log
    python po4ao_reward.py --kind log_modal --eps 1e-5
    python po4ao_reward.py --kind strehl
    python po4ao_reward.py --kind l2 --pessimism 1.0
    python po4ao_reward.py --kind l2 --action_penalty smooth
    python po4ao_reward.py --bench --kind log        # PAPYRUS, through DAO (see po4ao_common/bench_dao.py)
"""

import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
from po4ao_common.runner import run_sim, run_bench, save_history, summarize
from po4ao_config_reward import config
from po4ao_rewards import Reward
from run_baseline import build_baseline, make_env


def make_reward(env, cfg, modes=None):
    """modes: (n_valid, n_modes) orthonormal modal basis in actuator space. Default: an orthonormal basis
    of the controlled subspace taken from the projector the policy already uses (F = M2C pinv(M2C))."""
    r = cfg['reward']
    if modes is None:
        evals, evecs = torch.linalg.eigh(env.KL_projection.double().cpu())
        modes = evecs[:, evals > 0.5].float()
    return Reward(r['kind'], r['eps'], r['action_penalty'], r['pessimism'], r['discount'],
                  mask=env.mask.cpu(), modes=modes, strehl_scale=r['strehl_scale'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', default=config['reward']['kind'])
    ap.add_argument('--eps', type=float, default=config['reward']['eps'])
    ap.add_argument('--action_penalty', default=config['reward']['action_penalty'])
    ap.add_argument('--pessimism', type=float, default=config['reward']['pessimism'])
    ap.add_argument('--penalty', type=float, default=config['RL']['loss_function_penalty'])
    ap.add_argument('--n_history', type=int, default=config['MDP']['n_history'])
    ap.add_argument('--iterations', type=int, default=config['RL']['iterations'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--bench', action='store_true')
    args = ap.parse_args()
    config['reward'].update({'kind': args.kind, 'eps': args.eps, 'action_penalty': args.action_penalty,
                             'pessimism': args.pessimism})
    config['RL'].update({'loss_function_penalty': args.penalty, 'iterations': args.iterations})
    config['MDP']['n_history'] = args.n_history
    torch.manual_seed(args.seed)

    if args.bench:
        from po4ao_common.bench_dao import PapyrusBench
        env = PapyrusBench(config)
        agent = build_baseline(env, config, make_reward(env, config))
        run_bench(env, agent, config, savedir='~/PO4AO/logs/reward_%s' % args.kind)
    else:
        env = make_env(config, args.seed)
        # in the simulator the conversion from DM-space rms to radians of wavefront is known
        config['reward']['strehl_scale'] = float(env.cmd_scale) ** 2
        agent = build_baseline(env, config, make_reward(env, config))
        label = 'reward_%s_eps%g_%s_pess%g_pen%g' % (args.kind, args.eps, args.action_penalty, args.pessimism, args.penalty)
        history = run_sim(env, agent, config, label=label)
        save_history(history, config, os.path.join(HERE, 'runs', '%s_seed%d.json' % (label, args.seed)))
        print(label, summarize(history))
