"""PO4AO with the total DM command as input (pseudo-open-loop information).

    python po4ao_pol.py                           # toy simulator: residuals + increments + total command
    python po4ao_pol.py --n_history 8
    python po4ao_pol.py --no_policy_cmd           # only the dynamics sees the total command
    python po4ao_pol.py --input_mode totals       # the direct "swap increments for totals" version
    python po4ao_pol.py --bench                   # PAPYRUS, through DAO (see po4ao_common/bench_dao.py)

The bench already owns the total command (prev_commands in the original step()); it is returned with
every observation as obs.cmd and stored in the replay buffer next to the action.
"""

import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
from po4ao_common.core import PO4AO, l2_loss
from po4ao_common.models import Ensemble
from po4ao_common.runner import run_sim, run_bench, save_history, summarize
from po4ao_config_pol import config
from po4ao_models_pol import PolDynamics, PolPolicy


def build(env, cfg, loss_fn=l2_loss):
    n, nf, p = cfg['MDP']['n_history'], cfg['NN_models']['filters_per_layer'], cfg['pol']
    shape = (env.n_act, env.n_act)
    dynamics = Ensemble(lambda: PolDynamics(env.mask, n, nf, p['linear_skip'], p['linear_kernel'], p['cmd_gain'],
                                            p['input_mode']),
                        cfg['NN_models']['n_ensemble'])
    policy = PolPolicy(env.mask, env.KL_projection, n, nf, cfg['NN_models']['action_clamp'], p['linear_skip'],
                       p['linear_kernel'], cfg['integrator']['gain'], p['cmd_gain'], p['input_mode'], p['policy_cmd'])
    return PO4AO(cfg, dynamics, policy, (1,) + shape, shape, loss_fn, str(env.device))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_history', type=int, default=config['MDP']['n_history'])
    ap.add_argument('--iterations', type=int, default=config['RL']['iterations'])
    ap.add_argument('--input_mode', default=config['pol']['input_mode'], choices=['increments+total', 'totals'])
    ap.add_argument('--no_policy_cmd', action='store_true')
    ap.add_argument('--no_linear_skip', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--bench', action='store_true')
    args = ap.parse_args()
    config['MDP']['n_history'] = args.n_history
    config['RL']['iterations'] = args.iterations
    config['pol'].update({'input_mode': args.input_mode, 'policy_cmd': not args.no_policy_cmd,
                          'linear_skip': not args.no_linear_skip})
    torch.manual_seed(args.seed)

    if args.bench:
        from po4ao_common.bench_dao import PapyrusBench
        env = PapyrusBench(config)
        run_bench(env, build(env, config), config, savedir='~/PO4AO/logs/pol')
    else:
        from run_baseline import make_env
        env = make_env(config, args.seed)
        label = 'pol_n%d_%s%s%s' % (args.n_history, args.input_mode.replace('+', '_'),
                                    '' if config['pol']['policy_cmd'] else '_dynonly',
                                    '' if config['pol']['linear_skip'] else '_noskip')
        history = run_sim(env, build(env, config), config, label=label)
        save_history(history, config, os.path.join(HERE, 'runs', '%s_seed%d.json' % (label, args.seed)))
        print(label, summarize(history))
