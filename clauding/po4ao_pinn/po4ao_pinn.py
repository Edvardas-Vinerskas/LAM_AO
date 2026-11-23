"""PO4AO with a frozen-flow (physics-informed) dynamics model.

    python po4ao_pinn.py                         # toy simulator, free flow kernels, original PO4AO policy
    python po4ao_pinn.py --policy frozen_flow    # the policy is a predictor on the open-loop wavefront too
    python po4ao_pinn.py --mode wind             # explicit wind vectors: prints the identified wind
    python po4ao_pinn.py --no_residual           # purely linear dynamics, no CNN at all
    python po4ao_pinn.py --bench                 # PAPYRUS, through DAO (see po4ao_common/bench_dao.py)
"""

import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'po4ao_pol'))
sys.path.insert(0, HERE)
from po4ao_common.core import PO4AO, l2_loss
from po4ao_common.models import ConvPolicy, Ensemble
from po4ao_common.runner import run_sim, run_bench, save_history, summarize
from po4ao_config_pinn import config
from po4ao_models_pinn import FrozenFlowDynamics, FrozenFlowPolicy
from po4ao_models_pol import PolPolicy


def build(env, cfg, loss_fn=l2_loss):
    n, p = cfg['MDP']['n_history'], cfg['pinn']
    nf, clamp = cfg['NN_models']['filters_per_layer'], cfg['NN_models']['action_clamp']
    shape = (env.n_act, env.n_act)
    dynamics = Ensemble(lambda: FrozenFlowDynamics(env.mask, n, p['delay'], p['max_lag'], p['n_frames'], p['kernel'],
                                                   p['mode'], p['n_layers'], p['residual'], p['res_filters'],
                                                   p['res_frames'], p['res_weight'], p['free_lags'], p['res_causal']),
                        cfg['NN_models']['n_ensemble'])
    if p['policy'] == 'conv':            # the original PO4AO policy: residuals and changes of command only
        policy = ConvPolicy(env.mask, env.KL_projection, n, nf, clamp)
    elif p['policy'] == 'frozen_flow':   # predictor on the POL wavefront, same structure as the dynamics
        policy = FrozenFlowPolicy(env.mask, env.KL_projection, n, cfg['integrator']['leak'], p['delay'], p['max_lag'],
                                  p['n_frames'], p['kernel'], clamp, p['residual'], p['res_filters'], p['res_frames'])
    elif p['policy'] == 'pol':           # linear + CNN on residuals and total commands (po4ao_pol)
        policy = PolPolicy(env.mask, env.KL_projection, n, nf, clamp, True, 3, cfg['integrator']['gain'])
    else:
        raise ValueError(p['policy'])
    return PO4AO(cfg, dynamics, policy, (1,) + shape, shape, loss_fn, str(env.device))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default=config['pinn']['mode'], choices=['kernel', 'wind'])
    ap.add_argument('--policy', default=config['pinn']['policy'], choices=['conv', 'frozen_flow', 'pol'])
    ap.add_argument('--n_history', type=int, default=config['MDP']['n_history'])
    ap.add_argument('--n_frames', type=int, default=config['pinn']['n_frames'])
    ap.add_argument('--no_residual', action='store_true')
    ap.add_argument('--res_filters', type=int, default=config['pinn']['res_filters'])
    ap.add_argument('--res_frames', type=int, default=config['pinn']['res_frames'])
    ap.add_argument('--res_weight', type=float, default=config['pinn']['res_weight'])
    ap.add_argument('--iterations', type=int, default=config['RL']['iterations'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--bench', action='store_true')
    args = ap.parse_args()
    config['pinn'].update({'mode': args.mode, 'policy': args.policy, 'n_frames': args.n_frames,
                           'residual': not args.no_residual, 'res_filters': args.res_filters,
                           'res_frames': args.res_frames, 'res_weight': args.res_weight})
    config['MDP']['n_history'] = args.n_history
    config['RL']['iterations'] = args.iterations
    torch.manual_seed(args.seed)

    if args.bench:
        from po4ao_common.bench_dao import PapyrusBench
        env = PapyrusBench(config)
        run_bench(env, build(env, config), config, savedir='~/PO4AO/logs/pinn')
    else:
        from run_baseline import make_env
        env = make_env(config, args.seed)
        agent = build(env, config)
        label = 'pinn_%s_%s_n%d_m%d%s' % (args.mode, args.policy, args.n_history, args.n_frames,
                                          '_cnn%dx%d_w%g' % (args.res_filters, args.res_frames, args.res_weight)
                                          if config['pinn']['residual'] else '_linear')
        history = run_sim(env, agent, config, label=label)
        save_history(history, config, os.path.join(HERE, 'runs', '%s_seed%d.json' % (label, args.seed)))
        print(label, summarize(history))
        print('model 0:', agent.dynamics.models[0].describe())
        print('true wind of the simulator, pitch/frame (vx, vy), per layer:',
              [(round(float(x) / env.pitch, 3), round(float(y) / env.pitch, 3)) for x, y in zip(env.vx, env.vy)],
              'weights', [round(float(c), 2) for c in env.cn2])
