"""Baseline PO4AO networks (port of po4ao_models_PAPYRUS_upd.py) with the calling convention shared by
all the variants in clauding/:

    dynamics(obs_hist, act_hist, cmd_hist) -> next observation        (B, C, H, W)
        obs_hist (B, n, C, H, W)   o_{t-n+1} ... o_t          (C = 1 in DM space)
        act_hist (B, n, h, w)      a_{t-n+1} ... a_t          changes of command
        cmd_hist (B, n, h, w)      c_{t-n+1} ... c_t          total commands, c_t = clamp(leak c_{t-1} + a_t)

    policy(obs_hist, act_prev, cmd_prev) -> a_t                        (B, 1, h, w)
        obs_hist (B, n, C, H, W)   o_{t-n+1} ... o_t
        act_prev (B, n-1, h, w)    a_{t-n+1} ... a_{t-1}
        cmd_prev (B, n-1, h, w)    c_{t-n+1} ... c_{t-1}

The baseline ignores the total commands. Changes with respect to the original, all small:
  * the action is zeroed outside the valid actuators (the original left the raw convolution output
    there: never sent to the DM, but stored in the replay buffer, fed back to both networks and
    counted in the action penalty),
  * no in-place write on the network output.
"""

import torch
from torch import nn


def conv_net(c_in, c_out, n_filt, n_layers=3):
    layers = [nn.Conv2d(c_in, n_filt, 3, padding=1), nn.LeakyReLU()]
    for _ in range(n_layers - 2):
        layers += [nn.Conv2d(n_filt, n_filt, 3, padding=1), nn.LeakyReLU()]
    layers += [nn.Conv2d(n_filt, c_out, 3, padding=1)]
    return nn.Sequential(*layers)


class ActionHead(nn.Module):
    """Clamp, keep the valid actuators only, project on the controlled modes (F = M2C pinv(M2C))."""

    def __init__(self, mask, F, clamp=1.0):
        super().__init__()
        self.register_buffer('mask', torch.as_tensor(mask).bool())
        self.register_buffer('F', torch.as_tensor(F).float())
        self.clamp = clamp

    def forward(self, raw):
        # raw (B, 1, h, w)
        vec = raw.clamp(-self.clamp, self.clamp)[:, 0][:, self.mask]            # (B, n_valid)
        vec = vec @ self.F.T
        out = torch.zeros_like(raw)
        out[:, 0][:, self.mask] = vec
        return out


class ConvDynamics(nn.Module):

    def __init__(self, mask, n_history, n_filt=32):
        super().__init__()
        self.register_buffer('mask', torch.as_tensor(mask).bool())
        self.net = conv_net(2 * n_history, 1, n_filt)

    def forward(self, obs_hist, act_hist, cmd_hist=None):
        feats = torch.cat([obs_hist[:, :, 0], act_hist], dim=1)
        return self.net(feats) * self.mask


class ConvPolicy(nn.Module):

    def __init__(self, mask, F, n_history, n_filt=32, clamp=1.0):
        super().__init__()
        self.net = conv_net(2 * n_history - 1, 1, n_filt)
        self.head = ActionHead(mask, F, clamp)

    def forward(self, obs_hist, act_prev, cmd_prev=None):
        feats = torch.cat([obs_hist[:, :, 0], act_prev], dim=1)
        return self.head(self.net(feats))


class Ensemble(nn.Module):
    """A list of dynamics models. forward returns (B, n_models, C, H, W)."""

    def __init__(self, make_model, n_models=5):
        super().__init__()
        self.models = nn.ModuleList([make_model() for _ in range(n_models)])

    def forward(self, obs_hist, act_hist, cmd_hist=None):
        return torch.stack([m(obs_hist, act_hist, cmd_hist) for m in self.models], dim=1)
