"""Planning losses (negative rewards) for the PO4AO policy training.

The original loss, summed over the planning horizon, is

    L = mean(pred^2) + penalty * mean(action^2)

pred = predicted next residual (DM space), action = change of command. Everything here keeps that
interface (see po4ao_common/core.py: loss_fn(pred, action, ctx) -> scalar), so a Reward object can be
passed to any of the variants.

Residual terms (`kind`)
    'l2'           the original. Since Strehl ~ exp(-sigma^2), this already IS -log(Strehl) up to a scale.
    'log'          mean_b log(eps + msr_b), msr_b = mean-square residual of sample b. The logarithm is taken
                   per sample because a log around the whole batch loss would do nothing: Adam is invariant
                   to the scale of the loss, and d log(x) = dx / x is only a rescaling. Per sample it is a
                   re-weighting by 1 / (eps + msr_b): well-corrected moments count as much as bad ones.
    'log_modal'    mean_i eps * log(1 + m_i^2 / eps), m = modal coefficients of the residual. Weight
                   1 / (eps + m_i^2) per mode: no mode is ignored because another one is larger.
                   This is the closest to "log contrast" for high-contrast imaging.
    'log_pixel'    same on actuators instead of modes (Cauchy / Lorentzian loss).
    'strehl'       mean_b 1 - exp(-s * msr_b): the Marechal Strehl itself. Bounded: gradients fade for
                   very bad samples, the opposite emphasis to 'log'.
    'huber'        quadratic below `eps`, linear above (robust to outliers such as cosmic hits or bad frames).
    'modal'        mean_i w_i m_i^2 with user weights (e.g. a band of spatial frequencies = a dark-hole region).

`eps` is where the logarithm stops rewarding: set it to the noise floor (variance the residual cannot go
below). log(eps + x) with eps > 0 answers both objections to a plain log: it accepts x = 0, and the
eps * log(1 + x / eps) form is >= 0, zero at x = 0 and equal to x (the L2 loss) for x << eps.

Action terms (`action_penalty`)
    'increment'    penalty * mean(a_t^2), the original
    'smooth'       penalty * mean((a_t - a_{t-1})^2): penalises jitter, not the tracking of fast turbulence
    'command'      penalty * mean(c_t^2): keeps the stroke low (acts like a leak)
    'none'

Other terms
    pessimism      beta * variance of the ensemble predictions: the policy is pushed away from regions
                   where the models disagree (MOPO-style penalty), which is where a learned model is wrong.
    discount       weight gamma^k on step k of the horizon.
"""

import torch


class Reward:

    def __init__(self, kind='l2', eps=1e-4, action_penalty='increment', pessimism=0.0, discount=1.0,
                 mask=None, modes=None, mode_weights=None, strehl_scale=1.0):
        """
        mask           (h, w) valid actuators            (needed by the modal kinds)
        modes          (n_valid, n_modes) orthonormal modal basis in actuator space (M2C, or its Q factor)
        mode_weights   (n_modes,) for kind 'modal'
        strehl_scale   (radians of wavefront rms per unit of DM-space rms)^2, for kind 'strehl'
        """
        self.kind, self.eps, self.action_penalty = kind, eps, action_penalty
        self.pessimism, self.discount, self.strehl_scale = pessimism, discount, strehl_scale
        self.mask = None if mask is None else torch.as_tensor(mask).bool()
        self.modes = None if modes is None else torch.as_tensor(modes).float()
        self.mode_weights = None if mode_weights is None else torch.as_tensor(mode_weights).float()

    def _modal(self, pred):
        vec = pred[:, 0][:, self.mask.to(pred.device)]                     # (B, n_valid)
        return vec @ self.modes.to(pred.device)                             # (B, n_modes)

    def residual_term(self, pred):
        kind, eps = self.kind, self.eps
        if kind == 'l2':
            return pred.pow(2).mean()
        if kind == 'log':
            msr = pred.pow(2).flatten(1).mean(1)
            return torch.log(eps + msr).mean()
        if kind == 'log_pixel':
            return eps * torch.log1p(pred.pow(2) / eps).mean()
        if kind == 'log_modal':
            return eps * torch.log1p(self._modal(pred).pow(2) / eps).mean()
        if kind == 'strehl':
            msr = pred.pow(2).flatten(1).mean(1)
            return (1.0 - torch.exp(-self.strehl_scale * msr)).mean()
        if kind == 'huber':
            a = pred.abs()
            return torch.where(a < eps, 0.5 * a.pow(2), eps * (a - 0.5 * eps)).mean()
        if kind == 'modal':
            return (self._modal(pred).pow(2) * self.mode_weights.to(pred.device)).mean()
        raise ValueError(kind)

    def action_term(self, action, ctx):
        if self.action_penalty == 'increment':
            return action.pow(2).mean()
        if self.action_penalty == 'smooth':
            return (action - ctx['prev_action']).pow(2).mean()
        if self.action_penalty == 'command':
            return ctx['cmd'].pow(2).mean()
        if self.action_penalty == 'none':
            return 0.0
        raise ValueError(self.action_penalty)

    def __call__(self, pred, action, ctx):
        loss = self.residual_term(pred) + ctx['penalty'] * self.action_term(action, ctx)
        if self.pessimism > 0:
            loss = loss + self.pessimism * ctx['ens'].var(dim=1, unbiased=False).mean()
        if self.discount != 1.0:
            loss = loss * self.discount ** ctx['k']
        return loss
