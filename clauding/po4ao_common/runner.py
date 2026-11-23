"""Loops around the PO4AO agent.

    run_sim      synchronous: episode, then training, then episode... for the simulator
                 (equivalent to a trainer that always finishes within one episode)
    run_bench    asynchronous, as po4ao_PAPYRUS_upd.py: the control loop never waits for the trainer,
                 a second process trains on the shared replay buffers and publishes the policy weights

Both take an environment with flatten() / step(action) / sample_noise(sigma) (toy_env.ToyAOEnv or
bench_dao.PapyrusBench) and an agent (core.PO4AO).
"""

import copy
import json
import os
import time

import numpy as np
import torch


def get_obs(obs, cfg):
    """The observation the networks see, (C, H, W)."""
    if cfg['MDP']['obs_key'] == 'dm':
        return obs.dm.unsqueeze(0)
    return obs.pix * cfg['MDP']['obs_gain']


@torch.no_grad()
def run_episode(env, agent, cfg, mode, obs, sigma=0.0, store_warmup=False, policy=None):
    """
    mode 'warmup'      integrator + exploration noise, starts from a flat DM (as the original)
         'integrator'  plain integrator
         'policy'      the policy
    :return next observation, metrics of the episode
    """
    gain = cfg['integrator']['gain']
    if mode == 'warmup':
        obs = env.flatten()
        agent.replay.new_segment()
        agent.replay_warmup.new_segment()
        agent.reset_history()

    reward, res_var, strehl, latency = 0.0, [], [], []
    for _ in range(cfg['RL']['episode_length']):
        o = get_obs(obs, cfg)
        agent.push_obs(o)
        if mode == 'policy':
            t0 = time.perf_counter()
            action = agent.act(policy)
            if agent.device != 'cpu':
                torch.cuda.synchronize()
            latency.append(time.perf_counter() - t0)
        else:
            action = gain * obs.dm
            if sigma > 0:
                action = action + env.sample_noise(sigma)

        next_obs = env.step(action)
        agent.push_action(action, next_obs.cmd)
        agent.replay.append(o, action, next_obs.cmd)
        if store_warmup:
            agent.replay_warmup.append(o, action, next_obs.cmd)

        reward += float((obs.dm ** 2).sum())
        info = getattr(env, 'info', None)
        if info:
            res_var.append(info['res_var'])
            strehl.append(info['strehl'])
        obs = next_obs

    metrics = {'mode': mode, 'reward': reward}
    if res_var:
        metrics['res_var'] = float(np.mean(res_var))
        metrics['strehl'] = float(np.mean(strehl))
    if latency:
        metrics['latency_ms'] = 1e3 * float(np.median(latency))
    return obs, metrics


def run_sim(env, agent, cfg, log=print, label=''):
    """Warm-up, pre-training, then policy episodes with training after each one. Returns the episode log."""
    history = []
    sigma0 = cfg['RL']['max_sigma']
    sigma = sigma0
    n_warm = cfg['RL']['warmup_episodes']
    obs = env.flatten()

    for i in range(n_warm):
        obs, m = run_episode(env, agent, cfg, 'warmup', obs, sigma, store_warmup=sigma >= cfg['RL']['min_sigma'])
        history.append(m)
        log('%s warm-up %2d  sigma %.4f  res_var %.4f  SR %.3f' % (label, i, sigma, m.get('res_var', np.nan), m.get('strehl', np.nan)))
        sigma = max(0.0, sigma - sigma0 / n_warm)

    t0 = time.time()
    dyn = agent.train_dynamics(cfg['training']['dynamics_grad_steps_warmup'], warmup_only=True)
    pol = agent.train_policy(cfg['training']['policy_grad_steps_warmup'], warmup_only=True)
    log('%s pre-training %.1f s  dyn %.4f  pol %.4f  (x1000)' % (label, time.time() - t0, 1e3 * dyn, 1e3 * pol))

    mode = 'integrator' if cfg['integrator']['integrator'] else 'policy'
    obs = env.flatten()
    agent.replay.new_segment()
    for i in range(cfg['RL']['iterations']):
        obs, m = run_episode(env, agent, cfg, mode, obs)
        t0 = time.time()
        m['dyn_loss'] = agent.train_dynamics(cfg['training']['dynamics_grad_steps'])
        m['pol_loss'] = agent.train_policy(cfg['training']['policy_grad_steps'])
        m['train_s'] = time.time() - t0
        history.append(m)
        log('%s episode %2d  res_var %.4f  SR %.3f  dyn %.4f  pol %.4f  policy %.2f ms  training %.1f s'
            % (label, i, m.get('res_var', np.nan), m.get('strehl', np.nan), 1e3 * m['dyn_loss'], 1e3 * m['pol_loss'],
               m.get('latency_ms', np.nan), m['train_s']))
    return history


@torch.no_grad()
def integrator_reference(env, n_episodes, episode_length, gain):
    """Plain integrator on the same environment (same seed = same turbulence), per-episode residual variance."""
    obs = env.flatten()
    out = []
    for _ in range(n_episodes):
        v = []
        for _ in range(episode_length):
            obs = env.step(gain * obs.dm)
            v.append(env.info['res_var'])
        out.append(float(np.mean(v)))
    return out


def summarize(history, n_last=10):
    pol = [m for m in history if m['mode'] != 'warmup']
    last = pol[-n_last:]
    out = {'res_var_last': float(np.mean([m['res_var'] for m in last])) if 'res_var' in last[0] else None,
           'strehl_last': float(np.mean([m['strehl'] for m in last])) if 'strehl' in last[0] else None,
           'res_var_first': pol[0].get('res_var'),
           'res_var_worst': float(np.max([m['res_var'] for m in pol])) if 'res_var' in pol[0] else None,
           'latency_ms': float(np.median([m['latency_ms'] for m in pol if 'latency_ms' in m])) if 'latency_ms' in pol[0] else None}
    return out


def save_history(history, cfg, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump({'config': cfg, 'history': history, 'summary': summarize(history)}, f, indent=1)


# ======================================================================================== bench (asynchronous)
def _trainer(agent, cfg, start_q, done_q):
    """Runs in its own process. Trains on the shared buffers every time the control loop asks for it."""
    print('training process started')
    while True:
        msg = start_q.get()
        if msg == 'quit':
            return
        t0 = time.time()
        dyn = agent.train_dynamics(cfg['training']['dynamics_grad_steps'])
        pol = agent.train_policy(cfg['training']['policy_grad_steps'])
        if agent.device != 'cpu':
            torch.cuda.synchronize()
        done_q.put((dyn, pol, time.time() - t0))


def run_bench(env, agent, cfg, savedir=None, log=print):
    """
    Asynchronous version for the real loop. The agent's networks and buffers are shared with a trainer
    process (spawn); the control loop uses a private copy of the policy that is refreshed when a training
    round has finished. Structure and defaults follow po4ao_PAPYRUS_upd.main().

    Not exercised on hardware by the author of this file: on the toy simulator it was only run on cpu
    (sharing CUDA tensors between processes is not available on Windows).
    """
    import torch.multiprocessing as mp

    if savedir is not None:
        savedir = os.path.expanduser(savedir)       # the original passed '~/...' straight to os.makedirs
        os.makedirs(savedir, exist_ok=True)
        with open(os.path.join(savedir, 'config.json'), 'w') as f:
            json.dump(cfg, f, indent=1)

    history = []
    sigma0 = cfg['RL']['max_sigma']
    sigma = sigma0
    n_warm = cfg['RL']['warmup_episodes']
    obs = env.flatten()
    for i in range(n_warm):
        obs, m = run_episode(env, agent, cfg, 'warmup', obs, sigma, store_warmup=sigma >= cfg['RL']['min_sigma'])
        history.append(m)
        log('warm-up %d  reward %.3f' % (i, m['reward']))
        sigma = max(0.0, sigma - sigma0 / n_warm)

    dyn = agent.train_dynamics(cfg['training']['dynamics_grad_steps_warmup'], warmup_only=True)
    pol = agent.train_policy(cfg['training']['policy_grad_steps_warmup'], warmup_only=True)
    log('pre-training  dyn %.4f  pol %.4f  (x1000)' % (1e3 * dyn, 1e3 * pol))

    # private copy for the control loop, shared originals for the trainer
    policy_rt = copy.deepcopy(agent.policy).eval()
    for p in policy_rt.parameters():
        p.requires_grad_(False)
    agent.dynamics.share_memory()
    agent.policy.share_memory()
    agent.replay.share_memory()
    agent.replay_warmup.share_memory()

    ctx = mp.get_context('spawn')
    start_q, done_q = ctx.Queue(), ctx.Queue()
    trainer = ctx.Process(target=_trainer, args=(agent, cfg, start_q, done_q), daemon=True)
    trainer.start()

    training = False
    obs = env.flatten()
    agent.replay.new_segment()
    mode = 'integrator' if cfg['integrator']['integrator'] else 'policy'
    try:
        for i in range(cfg['RL']['iterations']):
            t0 = time.time()
            obs, m = run_episode(env, agent, cfg, mode, obs, policy=policy_rt)
            if training:
                try:
                    m['dyn_loss'], m['pol_loss'], m['train_s'] = done_q.get(False)
                    policy_rt.load_state_dict(agent.policy.state_dict())
                    training = False
                except Exception:
                    pass
            if not training:
                start_q.put('train')
                training = True
            history.append(m)
            log('iteration %d (%.2f s)  reward %.3f  policy %.2f ms' % (i, time.time() - t0, m['reward'], m.get('latency_ms', np.nan)))
    finally:
        start_q.put('quit')
        trainer.join(timeout=30)
        if savedir is not None:
            agent.save(os.path.join(savedir, 'models_final.pt'))
            with open(os.path.join(savedir, 'history.json'), 'w') as f:
                json.dump(history, f, indent=1)
    return history
