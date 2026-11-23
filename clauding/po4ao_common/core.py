"""The PO4AO agent: same algorithm as po4ao_PAPYRUS_upd.py, written once for all the variants.

    1. dynamics: ensemble of models trained to predict the next observation from the recent
       observations and commands (supervised, on the replay buffers),
    2. policy: trained by back-propagating the planning loss through `planning_horizon` steps of the
       learned dynamics (ensemble mean), starting from histories drawn from the replay buffers,
    3. control: the policy maps the current history to a change of DM command.

What a variant provides: the two networks (see models.py for the calling convention) and the loss.

    loss_fn(pred, action, ctx) -> scalar
        pred    (B, C, H, W)   predicted next observation (ensemble mean)
        action  (B, 1, h, w)   the action that led to it
        ctx     dict: 'ens' (B, n_models, C, H, W) individual predictions, 'cmd' (B, 1, h, w) total
                command after the action, 'prev_action' (B, 1, h, w), 'k' step in the horizon, 'penalty'
"""

import torch

from .replay import Replay


def l2_loss(pred, action, ctx):
    """The original PO4AO loss: predicted residual + penalty on the action."""
    return pred.pow(2).mean() + ctx['penalty'] * action.pow(2).mean()


class PO4AO:

    def __init__(self, cfg, dynamics, policy, obs_shape, act_shape, loss_fn=l2_loss, device='cuda:0',
                 make_buffers=True):
        self.cfg = cfg
        self.device = device
        self.dynamics = dynamics.to(device)
        self.policy = policy.to(device)
        self.loss_fn = loss_fn
        self.obs_shape, self.act_shape = tuple(obs_shape), tuple(act_shape)

        self.n_history = cfg['MDP']['n_history']
        self.horizon = cfg['MDP']['planning_horizon']
        self.batch_size = cfg['NN_models']['training_batch']
        self.leak = cfg['integrator']['leak']
        self.clamp = cfg['integrator']['command_clamp']
        self.penalty = cfg['RL']['loss_function_penalty']
        self.warmup_percent = cfg['replay_buffers']['train_warmup_percent']
        self.dyn_rollout = cfg['training']['dynamics_rollout']
        self.dyn_clip = cfg['training']['dynamics_grad_clip']
        self.pol_clip = cfg['training']['policy_grad_clip']

        self.dynamics_optimizer = torch.optim.Adam(self.dynamics.parameters(), lr=cfg['training']['lr_dynamics'])
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=cfg['training']['lr_policy'])

        if make_buffers:
            ep = cfg['RL']['episode_length']
            dtype = torch.float16 if cfg['replay_buffers']['obs_half_precision'] else torch.float32
            self.replay = Replay(obs_shape, act_shape, cfg['replay_buffers']['replay_size'] * ep, device, dtype)
            self.replay_warmup = Replay(obs_shape, act_shape, cfg['replay_buffers']['warmup_memory'] * ep, device, dtype)
        self.reset_history()

    # ==================================================================== training
    def _sample(self, window, warmup_only=False):
        use_warmup = warmup_only or (not self.replay.can_sample(window)) or float(torch.rand(1)) < self.warmup_percent
        buf = self.replay_warmup if (use_warmup and self.replay_warmup.can_sample(window)) else self.replay
        return buf.sample(window, self.batch_size)

    def train_dynamics(self, n_steps, warmup_only=False):
        """Supervised training of every model of the ensemble, each on its own mini-batches."""
        n, K = self.n_history, self.dyn_rollout
        self.dynamics.train()
        loss = torch.zeros(())
        for _ in range(n_steps):
            self.dynamics_optimizer.zero_grad()
            loss = 0
            for model in self.dynamics.models:
                obs, act, cmd = self._sample(n + K, warmup_only)
                obs_hist = obs[:, :n]
                for k in range(K):
                    pred = model(obs_hist, act[:, k:k + n], cmd[:, k:k + n])
                    if hasattr(model, 'prediction_loss'):     # a model may weight its errors (pixel space does)
                        loss = loss + model.prediction_loss(pred, obs[:, n + k]) / K
                    else:
                        loss = loss + (obs[:, n + k] - pred).pow(2).mean() / K
                    if k < K - 1:        # feed the prediction back: trains what the planning actually uses
                        obs_hist = torch.cat([obs_hist[:, 1:], pred.unsqueeze(1)], dim=1)
                if hasattr(model, 'regularization'):
                    loss = loss + model.regularization()
            loss.backward()
            if self.dyn_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.dynamics.parameters(), self.dyn_clip)
            self.dynamics_optimizer.step()
        return float(loss) / len(self.dynamics.models)

    def train_policy(self, n_steps, warmup_only=False):
        """Back-propagation of the planning loss through the learned dynamics."""
        n = self.n_history
        self.policy.train()
        for p in self.dynamics.parameters():
            p.requires_grad_(False)
        loss = torch.zeros(())
        for _ in range(n_steps):
            self.policy_optimizer.zero_grad()
            obs_hist, act, cmd = self._sample(n, warmup_only)
            act_prev, cmd_prev = act[:, :-1], cmd[:, :-1]
            prev_action = act[:, -2:-1]
            loss = 0
            for k in range(self.horizon):
                action = self.policy(obs_hist, act_prev, cmd_prev)
                # the bench integrates the action itself: the same leak and clamp are applied here
                new_cmd = (self.leak * cmd_prev[:, -1:] + action).clamp(-self.clamp, self.clamp)
                act_hist = torch.cat([act_prev, action], dim=1)
                cmd_hist = torch.cat([cmd_prev, new_cmd], dim=1)
                ens = self.dynamics(obs_hist, act_hist, cmd_hist)
                pred = ens.mean(dim=1)
                ctx = {'ens': ens, 'cmd': new_cmd, 'prev_action': prev_action, 'k': k,
                       'horizon': self.horizon, 'penalty': self.penalty}
                loss = loss + self.loss_fn(pred, action, ctx)
                obs_hist = torch.cat([obs_hist[:, 1:], pred.unsqueeze(1)], dim=1)
                act_prev, cmd_prev, prev_action = act_hist[:, 1:], cmd_hist[:, 1:], action
            loss.backward()
            if self.pol_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.pol_clip)
            self.policy_optimizer.step()
        for p in self.dynamics.parameters():
            p.requires_grad_(True)
        return float(loss)

    @torch.no_grad()
    def rollout(self, obs_hist, act_prev, cmd_prev, actions):
        """Open-loop predictions of the ensemble mean for a given list of actions, each (B, 1, h, w)."""
        preds = []
        for action in actions:
            new_cmd = (self.leak * cmd_prev[:, -1:] + action).clamp(-self.clamp, self.clamp)
            act_hist = torch.cat([act_prev, action], dim=1)
            cmd_hist = torch.cat([cmd_prev, new_cmd], dim=1)
            pred = self.dynamics(obs_hist, act_hist, cmd_hist).mean(dim=1)
            preds.append(pred)
            obs_hist = torch.cat([obs_hist[:, 1:], pred.unsqueeze(1)], dim=1)
            act_prev, cmd_prev = act_hist[:, 1:], cmd_hist[:, 1:]
        return preds

    @torch.no_grad()
    def dynamics_error(self, n_batches=20, steps=None):
        """Open-loop prediction error of the ensemble mean on the replay buffer, per step ahead,
        normalised by the variance of the target (1 = no better than predicting zero)."""
        n = self.n_history
        K = steps or self.horizon
        err = torch.zeros(K)
        ref = torch.zeros(K)
        for _ in range(n_batches):
            obs, act, cmd = self.replay.sample(n + K, self.batch_size)
            obs_hist = obs[:, :n]
            for k in range(K):
                pred = self.dynamics(obs_hist, act[:, k:k + n], cmd[:, k:k + n]).mean(dim=1)
                err[k] += float((obs[:, n + k] - pred).pow(2).mean())
                ref[k] += float(obs[:, n + k].pow(2).mean())
                obs_hist = torch.cat([obs_hist[:, 1:], pred.unsqueeze(1)], dim=1)
        return (err / ref).tolist()

    # ==================================================================== control
    def reset_history(self):
        n = self.n_history
        self.obs_hist = torch.zeros((1, n) + self.obs_shape, device=self.device)
        self.act_prev = torch.zeros((1, n - 1) + self.act_shape, device=self.device)
        self.cmd_prev = torch.zeros((1, n - 1) + self.act_shape, device=self.device)

    @torch.no_grad()
    def push_obs(self, obs):
        """obs (C, H, W): the newest observation o_t."""
        self.obs_hist = torch.cat([self.obs_hist[:, 1:], obs.reshape((1, 1) + self.obs_shape)], dim=1)

    @torch.no_grad()
    def push_action(self, action, cmd):
        """action a_t and the total command c_t it produced, (h, w) each."""
        self.act_prev = torch.cat([self.act_prev[:, 1:], action.reshape((1, 1) + self.act_shape)], dim=1)
        self.cmd_prev = torch.cat([self.cmd_prev[:, 1:], cmd.reshape((1, 1) + self.act_shape)], dim=1)

    @torch.no_grad()
    def act(self, policy=None):
        """Action for the history as it stands after push_obs. Returns (h, w)."""
        policy = policy if policy is not None else self.policy
        return policy(self.obs_hist, self.act_prev, self.cmd_prev)[0, 0]

    # ==================================================================== checkpoints
    def save(self, path, buffers=False):
        d = {'dynamics': self.dynamics.state_dict(), 'policy': self.policy.state_dict(), 'config': self.cfg}
        if buffers:
            d['replay'] = self.replay.state_dict()
            d['replay_warmup'] = self.replay_warmup.state_dict()
        torch.save(d, path)

    def load(self, path, buffers=False):
        d = torch.load(path, map_location=self.device)
        self.dynamics.load_state_dict(d['dynamics'])
        self.policy.load_state_dict(d['policy'])
        if buffers and 'replay' in d:
            self.replay.load_state_dict(d['replay'])
            self.replay_warmup.load_state_dict(d['replay_warmup'])
