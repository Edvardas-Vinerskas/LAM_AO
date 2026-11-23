"""Off-line comparison of the dynamics models on the toy simulator. Three questions per model:

1. PREDICTION. Trained on the PO4AO warm-up (recording A: integrator gain 0.4 + exploration noise), how well
   does it predict a held-out warm-up recording, 1 to 4 frames ahead, open loop, with the recorded commands?
2. TRANSFER. Same question on recording B, made with a different controller (integrator gain 1.0, no
   exploration noise). PO4AO's planning asks the model about controllers it has not seen.
3. RESPONSE TO ACTIONS. The policy is trained on the model's derivative with respect to the action, which
   prediction error does not measure. The simulator is branched at 30 instants: the same four actions are
   applied in both branches, except for a small extra poke on the first one. The difference between the
   branches is the true response to the poke (the WFS noise is identical in both and cancels). The model is
   asked for the same difference.
        response gain    projection of the predicted response on the true one, two frames after the poke
                         (1 = right amplitude; the loop delay is two frames)
        response error   |predicted - true|^2 / |true|^2, 2, 3 and 4 frames after the poke
        early response   what the model predicts one frame after the poke, where the truth is zero
   For the pixel-space models the response is compared after reconstruction to DM space.

Prediction scores are error variance over target variance (0 = perfect, 1 = no better than predicting zero).

    python dynamics_benchmark.py                 # DM-space models
    python dynamics_benchmark.py --pix           # adds the pixel-space models (about 2 GB of GPU memory)
"""

import argparse
import copy
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ['', 'po4ao_pol', 'po4ao_pinn', 'po4ao_pix']:
    sys.path.insert(0, os.path.join(HERE, d))
from po4ao_common.config import make_config
from run_baseline import make_env, build_baseline

PIX_GAIN = 0.1
N_HIST = 64         # longest history any model needs


@torch.no_grad()
def record(cfg, seed, n_frames, gain, sigma0):
    env = make_env(cfg, seed)
    obs = env.flatten()
    out = {'dm': [], 'pix': [], 'act': [], 'cmd': []}
    for t in range(n_frames):
        action = gain * obs.dm
        sigma = sigma0 * (1 - t / n_frames)
        if sigma > 0:
            action = action + env.sample_noise(sigma)
        nxt = env.step(action)
        out['dm'].append(obs.dm.unsqueeze(0))
        out['pix'].append((obs.pix * PIX_GAIN).half())
        out['act'].append(action)
        out['cmd'].append(nxt.cmd)
        obs = nxt
    return {k: torch.stack(v) for k, v in out.items()}, env


@torch.no_grad()
def record_response(cfg, seed, n_probes=30, gain=0.6, poke=0.01, steps=4, spacing=120):
    """Histories, the four actions, and the true response to a poke on the first of them, at n_probes instants."""
    env = make_env(cfg, seed)
    obs = env.flatten()
    hist = {'dm': [], 'pix': [], 'act': [], 'cmd': []}
    out = {k: [] for k in ['dm', 'pix', 'act', 'cmd', 'actions', 'poke', 'resp_dm', 'resp_pix']}
    t = 0
    while len(out['poke']) < n_probes:
        if t >= N_HIST + 200 and (t - N_HIST) % spacing == 0:
            # histories as the policy has them at this instant: o up to now, a and c up to the previous frame
            out['dm'].append(torch.stack(hist['dm'][-(N_HIST - 1):] + [obs.dm.unsqueeze(0)]))
            out['pix'].append(torch.stack(hist['pix'][-(N_HIST - 1):] + [(obs.pix * PIX_GAIN).half()]))
            out['act'].append(torch.stack(hist['act'][-(N_HIST - 1):]))
            out['cmd'].append(torch.stack(hist['cmd'][-(N_HIST - 1):]))
            delta = env.sample_noise(poke)
            snap = env.snapshot()
            # branch 1: closed loop without the poke; its actions are replayed in branch 2
            o, acts, ref = obs, [], []
            for k in range(steps):
                acts.append(gain * o.dm)
                o = env.step(acts[-1])
                ref.append((o.dm.unsqueeze(0), o.pix * PIX_GAIN))
            env.restore(snap)
            poked = []
            for k in range(steps):
                o = env.step(acts[k] + (delta if k == 0 else 0))
                poked.append((o.dm.unsqueeze(0), o.pix * PIX_GAIN))
            env.restore(snap)
            out['actions'].append(torch.stack(acts))
            out['poke'].append(delta)
            out['resp_dm'].append(torch.stack([p[0] - r[0] for p, r in zip(poked, ref)]))
            out['resp_pix'].append(torch.stack([p[1] - r[1] for p, r in zip(poked, ref)]).half())
        action = gain * obs.dm
        nxt = env.step(action)
        hist['dm'].append(obs.dm.unsqueeze(0))
        hist['pix'].append((obs.pix * PIX_GAIN).half())
        hist['act'].append(action)
        hist['cmd'].append(nxt.cmd)
        for v in hist.values():
            del v[:-N_HIST]
        obs = nxt
        t += 1
    return {k: torch.stack(v) for k, v in out.items()}


@torch.no_grad()
def response_test(agent, R, key):
    n = agent.n_history
    obs_hist = R[key][:, -n:].float()
    act_prev, cmd_prev = R['act'][:, -(n - 1):], R['cmd'][:, -(n - 1):]
    steps = R['actions'].shape[1]
    base = [R['actions'][:, k:k + 1] for k in range(steps)]
    poked = [base[0] + R['poke'].unsqueeze(1)] + base[1:]
    p0 = agent.rollout(obs_hist, act_prev, cmd_prev, base)
    p1 = agent.rollout(obs_hist, act_prev, cmd_prev, poked)
    pred = [b - a for a, b in zip(p0, p1)]
    if key == 'pix':        # compare after reconstruction: that is what the planning loss reads
        pred = [agent.loss_fn.reconstruct(p) for p in pred]
    true = R['resp_dm'].float()                                       # (M, steps, 1, h, w)
    ref = true[:, 1].pow(2).sum()
    gain = float((pred[1] * true[:, 1]).sum() / ref)
    err = [float((pred[k] - true[:, k]).pow(2).sum() / true[:, k].pow(2).sum()) for k in range(1, steps)]
    early = float(pred[0].pow(2).sum() / ref)
    return gain, err, early


def evaluate(agent, data, key, n_batches=40):
    agent.replay.state.zero_()
    agent.replay.stamp.fill_(-1)
    agent.replay.fill(data[key], data['act'], data['cmd'])
    return agent.dynamics_error(n_batches)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pix', action='store_true')
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--only', default='', help='only the models whose name starts with this, e.g. pinn')
    args = ap.parse_args()
    torch.manual_seed(0)

    base = make_config()
    A, env = record(base, 0, 10000, 0.4, 0.01)
    A_test, _ = record(base, 1, 4000, 0.4, 0.002)
    B, _ = record(base, 2, 4000, 1.0, 0.0)
    R = record_response(base, 3)

    # the variant folders are on sys.path (see above): these are po4ao_pol/po4ao_pol.py etc.
    from po4ao_config_pol import config as cfg_pol
    import po4ao_pol
    from po4ao_config_pinn import config as cfg_pinn
    import po4ao_pinn

    def variant(cfg, **over):
        cfg = copy.deepcopy(cfg)
        for sec, d in over.items():
            cfg[sec].update(d)
        cfg['replay_buffers'].update({'replay_size': 10, 'warmup_memory': 10})
        return cfg

    runs = [
        ('baseline, 64 frames', 'dm', build_baseline, variant(base)),
        ('baseline, 16 frames', 'dm', build_baseline, variant(base, MDP={'n_history': 16})),
        ('baseline, 8 frames', 'dm', build_baseline, variant(base, MDP={'n_history': 8})),
        ('pol: totals instead of increments, 16 fr.', 'dm', po4ao_pol.build, variant(cfg_pol, pol={'input_mode': 'totals'})),
        ('pol: increments + total, CNN only, 16 fr.', 'dm', po4ao_pol.build, variant(cfg_pol, pol={'linear_skip': False})),
        ('pol: increments + total, 16 frames', 'dm', po4ao_pol.build, variant(cfg_pol)),
        ('pol: increments + total, 8 frames', 'dm', po4ao_pol.build, variant(cfg_pol, MDP={'n_history': 8})),
        ('pinn: frozen-flow kernels + CNN, 8 frames', 'dm', po4ao_pinn.build, variant(cfg_pinn)),
        ('pinn: same, CNN sees the newest action', 'dm', po4ao_pinn.build, variant(cfg_pinn, pinn={'res_causal': False})),
        ('pinn: same, free DM-response lags', 'dm', po4ao_pinn.build, variant(cfg_pinn, pinn={'free_lags': True})),
        ('pinn: frozen-flow kernels only (linear)', 'dm', po4ao_pinn.build, variant(cfg_pinn, pinn={'residual': False})),
        ('pinn: linear, free DM-response lags', 'dm', po4ao_pinn.build, variant(cfg_pinn, pinn={'residual': False, 'free_lags': True})),
        ('pinn: explicit wind vectors + CNN', 'dm', po4ao_pinn.build, variant(cfg_pinn, pinn={'mode': 'wind'})),
    ]
    if args.pix:
        from po4ao_config_pix import config as cfg_pix
        import po4ao_pix
        runs.append(('pix: network learns the DM response', 'pix', po4ao_pix.build, variant(cfg_pix, pix={'dm_response': 'learned'})))
        runs.append(('pix: DM response = interaction matrix', 'pix', po4ao_pix.build, variant(cfg_pix, pix={'recon_weight': 0.0})))
        runs.append(('pix: same + reconstructed-error loss', 'pix', po4ao_pix.build, variant(cfg_pix)))
    if args.only:
        runs = [r for r in runs if r[0].startswith(args.only)]

    print('%-45s %7s | %-23s | %-23s | %5s | %-17s | %5s' % (
        'model', 'params', 'A held-out, steps 1..4', 'B other ctrl, steps 1..4', 'gain', 'resp. err 2..4', 'early'))
    for name, key, build, cfg in runs:
        agent = build(env, cfg)
        agent.replay_warmup.fill(A[key], A['act'], A['cmd'])
        agent.train_dynamics(args.steps, warmup_only=True)
        n_par = sum(p.numel() for p in agent.dynamics.models[0].parameters())
        ea = evaluate(agent, A_test, key)
        eb = evaluate(agent, B, key)
        gain, err, early = response_test(agent, R, key)
        print('%-45s %7d | %s | %s | %5.2f | %s | %5.3f' % (
            name, n_par, ' '.join('%.3f' % e for e in ea), ' '.join('%.3f' % e for e in eb), gain,
            ' '.join('%.3f' % e for e in err), early))
        if hasattr(agent.dynamics.models[0], 'describe'):
            print('    ', agent.dynamics.models[0].describe())
        del agent
        torch.cuda.empty_cache()
