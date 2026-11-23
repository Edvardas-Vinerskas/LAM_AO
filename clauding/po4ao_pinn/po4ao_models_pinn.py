"""Physics-informed PO4AO networks: frozen flow enforced on the pseudo-open-loop wavefront.

Frozen flow says the TURBULENT wavefront at t+1 is the wavefront at t translated by v*dt (one
translation per layer). It says nothing simple about the residual, which is turbulence minus DM. So the
physics has to be applied to the pseudo-open-loop (POL) wavefront, and the dynamics is built as

    1. what the WFS saw of the DM at frame j     d_j     = sum_{lag=1..L} G_lag * c_{j-lag}
    2. POL wavefront                             pol_j   = o_j + d_j
    3. frozen-flow prediction                    pol_t+1 = sum_{i<m} K_i * pol_{t-i}
    4. back to the residual                      o_t+1   = pol_t+1 - d_t+1  +  r(o, c)

`*` is a convolution on the 17 x 17 grid. Every operator in 1-3 is linear and bias-free:

    G   G_lag = w_lag * g: non-negative weights over the lags that sum to one (loop delay, also fractional)
        times one 3 x 3 kernel (optical gain, small mis-registration).
        Initialised to "the DM is seen `delay` frames later with unit gain".
    K   the flow operator. A translation is a convolution with a shifted delta, a sum of layers is a
        convolution with the sum of their kernels, so one convolution kernel represents multi-layer
        frozen flow. Initialised to the identity (persistence). Two parameterisations:
          'kernel'  free k x k kernels on the m most recent POL frames (the spatial-domain twin of
                    predictive Fourier control: one temporal filter per spatial frequency),
          'wind'    K = alpha * sum_l w_l * Shift(vx_l, vy_l): three numbers per layer, the shift built
                    with a cubic interpolation kernel so that it is differentiable in the wind vector.
    r   a small CNN with a zero-initialised last layer: it starts at exactly zero and is penalised by
        lambda * mean(r^2) (the APHYNITY recipe), so it only takes what the physics cannot explain
        (WFS non-linearity, edge effects, vibrations).

The same operator K acts on the measured residual and on the DM contribution because both are parts
of one wavefront: that sharing is the physical constraint, and it is what an unconstrained network
has to discover from data.

The policy has the same structure:  target command = sum_i P_i * pol_{t-i},  a_t = target - leak * c_{t-1} + r.
P initialised to the identity is the pseudo-open-loop controller "apply the last open-loop wavefront";
training moves P towards a predictor.
"""

import torch
import torch.nn.functional as Fn
from torch import nn

from po4ao_common.models import ActionHead, conv_net


def nearest_valid_index(mask):
    """For every pixel of the grid, flat index of the nearest valid actuator (itself if valid)."""
    mask = torch.as_tensor(mask).bool().cpu()
    h, w = mask.shape
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    pts = torch.stack([yy.flatten(), xx.flatten()], 1).float()
    valid = torch.nonzero(mask.flatten()).squeeze(1)
    d = torch.cdist(pts, pts[valid])
    return valid[d.argmin(1)]


def cubic_weights(d, radius):
    """Keys cubic-convolution weights that translate a signal by d pixels (differentiable in d).
    Returns (..., 2 * radius + 1); use as cross-correlation weights."""
    j = torch.arange(-radius, radius + 1, device=d.device, dtype=d.dtype)
    x = (j + d.unsqueeze(-1)).abs()
    a = -0.5
    w1 = (a + 2) * x ** 3 - (a + 3) * x ** 2 + 1
    w2 = a * x ** 3 - 5 * a * x ** 2 + 8 * a * x - 4 * a
    return torch.where(x <= 1, w1, torch.where(x < 2, w2, torch.zeros_like(x)))


class _PolBase(nn.Module):
    """Shared by the dynamics and the policy: DM response G and the POL wavefront.

    G is constrained to what a DM can do: G_lag = w_lag * g, with w = softmax over the lags (non-negative,
    sums to one: a possibly fractional delay) and g one 3 x 3 kernel (gain and sub-actuator registration).
    A free kernel per lag fits the warm-up better but for the wrong reason: under the integrator the
    commands are a function of the past residuals, so free lags become a hidden copy of the residual
    history (closed-loop identification bias) and the model degrades when the controller changes.
    Set free_lags=True to see it.
    """

    def __init__(self, mask, delay, max_lag, free_lags=False):
        super().__init__()
        assert 1 <= delay <= max_lag
        self.max_lag, self.free_lags = max_lag, free_lags
        self.register_buffer('mask', torch.as_tensor(mask).bool())
        self.register_buffer('nearest', nearest_valid_index(mask))
        if free_lags:
            gain = torch.zeros(1, max_lag, 3, 3)
            gain[0, max_lag - delay, 1, 1] = 1.0                             # a window c_{j-L} ... c_{j-1}: lag L first
        else:
            logit = torch.full((max_lag,), -2.0)
            logit[delay - 1] = 2.0                                           # ~95 % of the weight on `delay`
            self.lag_logit = nn.Parameter(logit)
            gain = torch.zeros(1, 1, 3, 3)
            gain[0, 0, 1, 1] = 1.0
        self.dm_gain = nn.Parameter(gain)

    def extend(self, x):
        """Fill the grid outside the valid actuators with the nearest valid value, so that a kernel that
        looks upwind of the pupil edge finds the edge value instead of zero."""
        return x.flatten(-2)[..., self.nearest].reshape(x.shape)

    def dm_seen(self, cmd, frames):
        """What the WFS saw of the DM at the frames j of `frames`: sum_lag G_lag * c_{j-lag}.
        cmd (B, T, h, w) with index = time. Returns (B, len(frames), h, w)."""
        L = self.max_lag
        if self.free_lags:
            return torch.cat([Fn.conv2d(cmd[:, j - L:j], self.dm_gain, padding=1) for j in frames], dim=1)
        w = torch.softmax(self.lag_logit, 0).flip(0).reshape(1, L, 1, 1)
        mixed = torch.stack([(cmd[:, j - L:j] * w).sum(1) for j in frames], dim=1)      # (B, len, h, w)
        B, n = mixed.shape[:2]
        return Fn.conv2d(mixed.reshape(B * n, 1, *mixed.shape[2:]), self.dm_gain, padding=1).reshape(mixed.shape)

    def describe_dm(self):
        if self.free_lags:
            g = self.dm_gain[0].sum(dim=(1, 2)).flip(0).tolist()
            return 'DM seen by the WFS: gain per lag 1..%d: %s' % (self.max_lag, ' '.join('%.2f' % v for v in g))
        w = torch.softmax(self.lag_logit, 0).tolist()
        return 'DM seen by the WFS: gain %.2f, weight of lags 1..%d: %s' % (
            float(self.dm_gain.sum()), self.max_lag, ' '.join('%.2f' % v for v in w))


class FrozenFlowDynamics(_PolBase):

    def __init__(self, mask, n_history, delay=2, max_lag=4, n_frames=3, kernel=5, mode='kernel',
                 n_layers=3, residual=True, res_filters=16, res_frames=4, res_weight=0.1, free_lags=False,
                 res_causal=True):
        super().__init__(mask, delay, max_lag, free_lags)
        assert n_history >= n_frames + max_lag, 'n_history must be at least n_frames + max_lag'
        self.n, self.m, self.k, self.mode = n_history, n_frames, kernel, mode
        self.res_frames, self.res_weight = res_frames, res_weight
        self.res_skip = (delay - 1) if res_causal else 0
        assert n_history >= res_frames + self.res_skip
        if mode == 'kernel':
            self.flow = nn.Parameter(torch.zeros(1, n_frames, kernel, kernel))
            with torch.no_grad():
                self.flow[0, -1, kernel // 2, kernel // 2] = 1.0
        elif mode == 'wind':
            self.wind = nn.Parameter(0.05 * torch.randn(n_layers, 2))       # (vx, vy) in actuator pitches per frame
            self.layer_logit = nn.Parameter(torch.zeros(n_layers))
            self.alpha_logit = nn.Parameter(torch.tensor(5.0))               # sigmoid -> AR coefficient just below 1
        else:
            raise ValueError(mode)
        self.res_net = None
        if residual:
            self.res_net = conv_net(2 * res_frames, 1, res_filters)
            nn.init.zeros_(self.res_net[-1].weight)
            nn.init.zeros_(self.res_net[-1].bias)
        self._res = None

    def flow_kernel(self):
        """(1, frames, k, k) cross-correlation weights of the flow operator."""
        if self.mode == 'kernel':
            return self.flow
        r = self.k // 2
        wx = cubic_weights(self.wind[:, 0], r)                               # (layers, k)
        wy = cubic_weights(self.wind[:, 1], r)
        w = torch.softmax(self.layer_logit, 0).reshape(-1, 1, 1) * wy.unsqueeze(2) * wx.unsqueeze(1)
        return torch.sigmoid(self.alpha_logit) * w.sum(0).reshape(1, 1, self.k, self.k)

    def predict_pol(self, pol):
        K = self.flow_kernel()
        pol = self.extend(pol[:, -K.shape[1]:])
        r = self.k // 2
        return Fn.conv2d(Fn.pad(pol, (r, r, r, r), mode='replicate'), K)

    def forward(self, obs_hist, act_hist, cmd_hist):
        o = obs_hist[:, :, 0]
        n, m = self.n, self.m
        d = self.dm_seen(cmd_hist, range(n - m, n + 1))                      # frames t-m+1 ... t+1
        pol = o[:, n - m:] + d[:, :m]
        out = self.predict_pol(pol) - d[:, m:]
        if self.res_net is not None:
            f, skip = self.res_frames, self.res_skip
            # The CNN gets the increments, not the totals: consecutive total commands are almost equal,
            # and a network fed with them mis-estimates the effect of a single action.
            # Causality: an action cannot reach the WFS before `delay` frames, so the newest delay - 1
            # actions are withheld from the CNN (under the integrator a_t = gain * o_t, and a network
            # that sees a_t learns to use it as a second copy of o_t: a response that does not exist).
            self._res = self.res_net(torch.cat([o[:, -f:], act_hist[:, n - f - skip:n - skip]], dim=1))
            out = out + self._res
        return out * self.mask

    def regularization(self):
        if self._res is None:
            return 0.0
        return self.res_weight * self._res.pow(2).mean()

    def __getstate__(self):
        # the cached CNN output belongs to an autograd graph and cannot be sent to the training process
        state = self.__dict__.copy()
        state['_res'] = None
        return state

    @torch.no_grad()
    def describe(self):
        """Human-readable state of the physical part (for logs)."""
        s = self.describe_dm()
        if self.mode == 'wind':
            w = torch.softmax(self.layer_logit, 0).tolist()
            v = self.wind.tolist()
            s += ' | layers (weight, vx, vy [pitch/frame]): ' + ' '.join('(%.2f, %+.3f, %+.3f)' % (w[i], v[i][0], v[i][1]) for i in range(len(w)))
        else:
            K = self.flow.sum(1)[0]
            r = self.k // 2
            ax = torch.arange(-r, r + 1, device=K.device, dtype=K.dtype)
            tot = K.sum()
            # a cross-correlation kernel with its weight at offset -v moves the signal by +v
            s += ' | flow kernel: sum %.3f, centroid -> mean shift (%+.3f, %+.3f) pitch/frame' % (
                float(tot), -float((K.sum(0) * ax).sum() / tot), -float((K.sum(1) * ax).sum() / tot))
        return s


class FrozenFlowPolicy(_PolBase):

    def __init__(self, mask, F, n_history, leak, delay=2, max_lag=4, n_frames=3, kernel=5, clamp=1.0,
                 residual=True, res_filters=16, res_frames=4):
        super().__init__(mask, delay, max_lag)
        assert n_history >= n_frames + max_lag
        self.n, self.m, self.k, self.leak = n_history, n_frames, kernel, leak
        self.res_frames = res_frames
        self.pred = nn.Parameter(torch.zeros(1, n_frames, kernel, kernel))
        with torch.no_grad():
            self.pred[0, -1, kernel // 2, kernel // 2] = 1.0
        self.res_net = None
        if residual:
            self.res_net = conv_net(2 * res_frames - 1, 1, res_filters)
            nn.init.zeros_(self.res_net[-1].weight)
            nn.init.zeros_(self.res_net[-1].bias)
        self.head = ActionHead(mask, F, clamp)

    def forward(self, obs_hist, act_prev, cmd_prev):
        o = obs_hist[:, :, 0]
        n, m, r = self.n, self.m, self.k // 2
        # cmd_prev holds c_{t-n+1} ... c_{t-1}: index = time index of the observations, the last one missing
        d = self.dm_seen(cmd_prev, range(n - m, n))                          # frames t-m+1 ... t
        pol = self.extend(o[:, n - m:] + d)
        target = Fn.conv2d(Fn.pad(pol, (r, r, r, r), mode='replicate'), self.pred)
        raw = target - self.leak * cmd_prev[:, -1:]
        if self.res_net is not None:
            f = self.res_frames
            raw = raw + self.res_net(torch.cat([o[:, -f:], act_prev[:, -(f - 1):]], dim=1))
        return self.head(raw)
