"""PO4AO in wavefront-sensor pixel space.

    python po4ao_pix.py                             # toy simulator (2-channel slopes image, 76 pixels across)
    python po4ao_pix.py --loss_space pixel          # loss on the raw predicted image, for comparison
    python po4ao_pix.py --dm_response learned       # the network learns the response to the DM by itself
    python po4ao_pix.py --no_integrator_skip        # policy = plain network
    python po4ao_pix.py --bench                     # PAPYRUS, through DAO (see po4ao_common/bench_dao.py)

The warm-up still uses the integrator on the reconstructed residual (obs.dm): the standard pipeline keeps
running, the networks simply read the WFS image (obs.pix) instead of its reconstruction.
"""

import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
from po4ao_common.core import PO4AO
from po4ao_common.models import Ensemble
from po4ao_common.runner import run_sim, run_bench, save_history, summarize
from po4ao_config_pix import config
from po4ao_models_pix import PixDynamics, PixPolicy, ReconLoss


def build(env, cfg, inner_loss=None):
    n, nf, p = cfg['MDP']['n_history'], cfg['NN_models']['filters_per_layer'], cfg['pix']
    gain = cfg['MDP']['obs_gain']
    recon, pix_mask = env.reconstructor()            # dm[valid] = recon @ pix[pix_mask]
    int_mat = env.interaction_matrix() * gain if p['dm_response'] == 'interaction_matrix' else None
    dynamics = Ensemble(lambda: PixDynamics(pix_mask, n, nf, p['reg_scale'], p['learn_registration'], p['skip_frames'],
                                            p['dm_response'], int_mat, env.mask, p['delay'], p['max_lag'],
                                            recon / gain, p['recon_weight']),
                        cfg['NN_models']['n_ensemble'])
    policy = PixPolicy(n, pix_mask, env.mask, env.KL_projection, nf, cfg['NN_models']['action_clamp'], p['reg_scale'],
                       p['learn_registration'], recon / gain if p['integrator_skip'] else None,
                       cfg['integrator']['gain'])
    loss = ReconLoss(recon, pix_mask, env.mask, gain, p['loss_space'], p['pixel_weight'], inner_loss)
    return PO4AO(cfg, dynamics, policy, tuple(pix_mask.shape), (env.n_act, env.n_act), loss, str(env.device))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_history', type=int, default=config['MDP']['n_history'])
    ap.add_argument('--loss_space', default=config['pix']['loss_space'], choices=['recon', 'pixel'])
    ap.add_argument('--dm_response', default=config['pix']['dm_response'], choices=['interaction_matrix', 'learned'])
    ap.add_argument('--no_integrator_skip', action='store_true')
    ap.add_argument('--iterations', type=int, default=config['RL']['iterations'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--bench', action='store_true')
    args = ap.parse_args()
    config['MDP']['n_history'] = args.n_history
    config['pix'].update({'loss_space': args.loss_space, 'dm_response': args.dm_response,
                          'integrator_skip': not args.no_integrator_skip})
    config['RL']['iterations'] = args.iterations
    torch.manual_seed(args.seed)

    if args.bench:
        from po4ao_common.bench_dao import PapyrusBench
        env = PapyrusBench(config)
        run_bench(env, build(env, config), config, savedir='~/PO4AO/logs/pix')
    else:
        from run_baseline import make_env
        env = make_env(config, args.seed)
        agent = build(env, config)
        label = 'pix_n%d_%s_%s%s' % (args.n_history, args.loss_space, args.dm_response,
                                     '' if config['pix']['integrator_skip'] else '_noskip')
        history = run_sim(env, agent, config, label=label)
        save_history(history, config, os.path.join(HERE, 'runs', '%s_seed%d.json' % (label, args.seed)))
        print(label, summarize(history))
        print('model 0:', agent.dynamics.models[0].describe())
