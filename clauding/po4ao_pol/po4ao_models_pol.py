"""PO4AO with pseudo-open-loop information: the networks also see the TOTAL DM command.

Inputs (DM space, 17 x 17 images):
    dynamics   o_{t-n+1..t},  a_{t-n+1..t},    c_{t-1}     -> o_{t+1}
    policy     o_{t-n+1..t},  a_{t-n+1..t-1},  c_{t-1}     -> a_t   (still a change of command: the bench
                                                                    interface and its safety clamp are unchanged)

o = residual seen by the WFS, a = change of command, c = total command held by the integrator.
With c_{t-1} and the increments every past command is known, so the open-loop (pseudo-open-loop, POL)
wavefront  o_j + c_{j-delay}  is a linear combination of the inputs for every frame of the history.

Two design points, both learned the hard way on the simulator (dynamics_benchmark.py shows them):

  * Keep the increments. Replacing them by the history of total commands (input_mode='totals', the most
    direct reading of the idea) carries the same information but is badly conditioned: consecutive total
    commands are almost equal, and the effect of one action is then a difference of nearly identical
    inputs. The dynamics trained that way under-estimated the response of the loop to an action by a
    third, and the policy optimised through it over-drove the mirror. With the increments as inputs the
    response is identified from the exploration noise directly, as in the original PO4AO.

  * Add a linear path. The commands are 10 to 30 times larger than the residual and the output is a
    small difference of large numbers (o_{t+1} = POL_{t+1} - c_{t+1-delay}). A bias-free linear
    convolution represents that difference exactly, a LeakyReLU network only approximately. Each network is
    output = Linear(inputs) + CNN(inputs). For the policy the linear path starts as the integrator
    (a_t = gain * o_t) and the last CNN layer at zero: the policy starts as the warm-up controller.

policy_cmd=False keeps the total command out of the policy (it then has the original PO4AO inputs).
A policy that sees c can move the pole of the integrator (a term +k*c in the action is a leak of
0.99 + k), and a slow drift is invisible to a 4-step planning horizon.
"""

import torch
from torch import nn

from po4ao_common.models import ActionHead, conv_net


def _features(obs_hist, acts, cmd, input_mode, cmd_gain):
    o = obs_hist[:, :, 0]
    if input_mode == 'totals':                      # o history + total-command history
        return torch.cat([o, cmd_gain * cmd], dim=1)
    if cmd is None:                                 # o history + increments (the original inputs)
        return torch.cat([o, acts], dim=1)
    return torch.cat([o, acts, cmd_gain * cmd], dim=1)


class PolDynamics(nn.Module):

    def __init__(self, mask, n_history, n_filt=32, linear_skip=True, linear_kernel=3, cmd_gain=1.0,
                 input_mode='increments+total'):
        super().__init__()
        self.register_buffer('mask', torch.as_tensor(mask).bool())
        self.cmd_gain, self.input_mode = cmd_gain, input_mode
        c_in = 2 * n_history + (1 if input_mode == 'increments+total' else 0)
        self.net = conv_net(c_in, 1, n_filt)
        self.lin = None
        if linear_skip:
            self.lin = nn.Conv2d(c_in, 1, linear_kernel, padding=linear_kernel // 2, bias=False)
            nn.init.zeros_(self.lin.weight)

    def forward(self, obs_hist, act_hist, cmd_hist):
        if self.input_mode == 'totals':
            feats = _features(obs_hist, None, cmd_hist, 'totals', self.cmd_gain)
        else:
            # c_{t-1}: the command before the current action (the current action is already in act_hist)
            feats = _features(obs_hist, act_hist, cmd_hist[:, -2:-1], self.input_mode, self.cmd_gain)
        out = self.net(feats)
        if self.lin is not None:
            out = out + self.lin(feats)
        return out * self.mask


class PolPolicy(nn.Module):

    def __init__(self, mask, F, n_history, n_filt=32, clamp=1.0, linear_skip=True, linear_kernel=3,
                 init_gain=0.4, cmd_gain=1.0, input_mode='increments+total', policy_cmd=True):
        super().__init__()
        self.cmd_gain, self.input_mode, self.policy_cmd = cmd_gain, input_mode, policy_cmd
        c_in = 2 * n_history - 1 + (1 if (input_mode == 'increments+total' and policy_cmd) else 0)
        self.net = conv_net(c_in, 1, n_filt)
        self.lin = None
        if linear_skip:
            self.lin = nn.Conv2d(c_in, 1, linear_kernel, padding=linear_kernel // 2, bias=False)
            nn.init.zeros_(self.lin.weight)
            with torch.no_grad():
                # channel n_history - 1 is the newest residual o_t: start as the integrator
                self.lin.weight[0, n_history - 1, linear_kernel // 2, linear_kernel // 2] = init_gain
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        self.head = ActionHead(mask, F, clamp)

    def forward(self, obs_hist, act_prev, cmd_prev):
        if self.input_mode == 'totals':
            feats = _features(obs_hist, None, cmd_prev, 'totals', self.cmd_gain)
        else:
            feats = _features(obs_hist, act_prev, cmd_prev[:, -1:] if self.policy_cmd else None,
                              self.input_mode, self.cmd_gain)
        out = self.net(feats)
        if self.lin is not None:
            out = out + self.lin(feats)
        return self.head(out)
