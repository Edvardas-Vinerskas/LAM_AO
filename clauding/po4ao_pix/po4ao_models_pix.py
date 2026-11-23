"""PO4AO in wavefront-sensor pixel space.

    dynamics   WFS image history (n x C x Hp x Wp) + DM command history (n x h x w)  ->  next WFS image (C x Hp x Wp)
    policy     WFS image history + past actions                                       ->  DM action (h x w)

C is the number of WFS channels (2 for x/y slopes maps, 4 for the four pyramid pupil images, 1 for a
Zernike WFS), Hp x Wp the pupil sampling of the WFS (76 on PAPYRUS) and h x w the DM grid (17).

What differs from the DM-space networks, and why (each point was a failure on the simulator first):

  * Registration. The DM grid and the WFS pupil image are two samplings of the same pupil with different
    extents (17 actuators cover a little more than the 76-pixel pupil). `Registration` resamples one onto
    the other with an affine transform (scale, shift, rotation: 6 numbers) initialised from the geometry.

  * Resolution. The networks work at half the WFS resolution inside (stride-2 stem): a frozen-flow shift
    of ~0.5 pixel/frame and an actuator footprint of ~5 pixels need a receptive field of tens of pixels.

  * The response to the DM is not left to the network (dm_response='interaction_matrix'). Most of the
    variance of a WFS image is signal the DM cannot produce, so a network trained on pixel error learns
    the effect of an action slowly and only near the actions it has seen; a policy optimised through it
    then finds actions the model is wrong about. Instead the calibrated interaction matrix IM is used:

        d_j     = g * IM @ sum_lag w_lag c_{j-lag}        DM as seen by the WFS (g: optical gain, w: delay; learned)
        pol_j   = o_j + d_j                               pseudo-open-loop image
        pol_t+1 = pol_t + network(o history, pol history) turbulence evolution: the only thing learned freely
        o_t+1   = pol_t+1 - d_t+1

    The actions reach the prediction only through the physical term d, so the response to any action has
    the right shape by construction. dm_response='learned' is the free version (network sees the actions).

  * The policy starts as the integrator (integrator_skip): action = gain * reconstructor(newest image)
    + network(...), with the last network layer at zero. A randomly initialised pixel policy is a much
    worse starting point than the warm-up controller.

  * Loss. The planning loss is NOT the predicted image squared (that would weight every mode by the
    square of the WFS sensitivity to it, and ask the policy to cancel signal the DM cannot produce). The
    predicted image goes through the linear reconstructor first: see ReconLoss.
"""

import torch
import torch.nn.functional as Fn
from torch import nn

from po4ao_common.models import ActionHead


class PyramidFrames:
    """Raw pyramid camera frame -> network input: the four pupil images, flux-normalised and reference-subtracted.

        s_q(x) = I_q(x) / N  -  I_ref,q(x) / N_ref,        q = 1..4,  x in the valid pixels of the pupil
        N      = mean of I_1 + I_2 + I_3 + I_4 over the valid pixels        (OOPAO 'slopesMaps' normalisation)

    The division by N removes photometric fluctuations (seeing, transparency, star flux), which would
    otherwise look like wavefront signal; the reference I_ref (camera frame for a flat DM, recorded with
    the calibration) removes the static pattern. output='slopes' gives the two slopes maps of the classical
    processing instead, Sx = (I1 - I2 + I4 - I3) / N - ref and Sy = (I1 - I4 + I2 - I3) / N - ref (same
    quadrant convention as OOPAO Pyramid.signalProcessing).

    `scale` multiplies the result to bring it to the size of the DM-space residual (it replaces
    cfg['MDP']['obs_gain'] for raw frames; set it with fit_scale on warm-up frames). The interaction
    matrix and the reconstructor given to PixDynamics / PixPolicy / ReconLoss must be calibrated on frames
    processed by this same object, scale included.

        quadrants  four (row, col) top-left corners of the pupil images on the camera, in the order 1..4
        size       pupil image size in pixels (76 on PAPYRUS)
        valid      (size, size) bool, valid pixels of one pupil image (same for the four)
        reference  raw camera frame for a flat wavefront, or None (no reference subtraction)
        dark       raw dark frame subtracted before anything else, or None
    """

    def __init__(self, quadrants, size, valid, reference=None, dark=None, output='pupils', scale=1.0, device='cpu'):
        assert output in ('pupils', 'slopes') and len(quadrants) == 4
        self.quadrants, self.size, self.output, self.scale = [tuple(q) for q in quadrants], size, output, scale
        self.device = torch.device(device)
        self.valid = torch.as_tensor(valid).bool().to(self.device)
        self.dark = None if dark is None else torch.as_tensor(dark).float().to(self.device)
        self.ref = None
        if reference is not None:
            self.ref = self._normalised(torch.as_tensor(reference).float().to(self.device))

    @property
    def mask(self):
        """(C, size, size) bool: valid pixels of the output (the pix_mask of the pixel-space networks)."""
        return self.valid.unsqueeze(0).expand(4 if self.output == 'pupils' else 2, -1, -1).clone()

    def _normalised(self, frame):
        if self.dark is not None:
            frame = frame - self.dark
        n = self.size
        I = torch.stack([frame[r:r + n, c:c + n] for r, c in self.quadrants]) * self.valid      # (4, n, n)
        N = I.sum(0)[self.valid].mean().clamp_min(1e-12)
        if self.output == 'pupils':
            return I / N
        I1, I2, I3, I4 = I
        return torch.stack([I1 - I2 + I4 - I3, I1 - I4 + I2 - I3]) / N

    def __call__(self, frame):
        s = self._normalised(torch.as_tensor(frame).float().to(self.device))
        if self.ref is not None:
            s = s - self.ref
        return self.scale * s * self.valid

    def fit_scale(self, frames, target_rms):
        """Set `scale` so that the rms over the valid pixels of these frames equals target_rms (for example
        the rms of the DM-space residual recorded at the same time)."""
        self.scale = 1.0
        rms = torch.stack([self(f)[self.mask].pow(2).mean() for f in frames]).mean().sqrt()
        self.scale = float(target_rms / rms.clamp_min(1e-12))
        return self.scale


class Registration(nn.Module):
    """Affine resampling between the DM grid and the WFS pupil image.

    scale = (pupil diameter) / (extent of the DM grid) = 15 / 17 when 15 of the 17 actuator pitches span the pupil.
    """

    def __init__(self, scale, learn=True):
        super().__init__()
        theta = torch.tensor([[scale, 0.0, 0.0], [0.0, scale, 0.0]])
        theta_inv = torch.tensor([[1.0 / scale, 0.0, 0.0], [0.0, 1.0 / scale, 0.0]])
        self.theta = nn.Parameter(theta, requires_grad=learn)               # pixel coordinates -> DM coordinates
        self.theta_inv = nn.Parameter(theta_inv, requires_grad=learn)       # DM coordinates -> pixel coordinates

    def _warp(self, x, theta, size):
        grid = Fn.affine_grid(theta.unsqueeze(0).expand(x.shape[0], 2, 3), (x.shape[0], x.shape[1]) + tuple(size),
                              align_corners=False)
        return Fn.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    def to_pix(self, dm_maps, size):
        """(B, K, h, w) on the DM grid -> (B, K, *size) on the pupil image."""
        return self._warp(dm_maps, self.theta, size)

    def to_dm(self, pix_maps, size):
        """(B, K, Hp, Wp) on the pupil image -> (B, K, *size) on the DM grid."""
        return self._warp(pix_maps, self.theta_inv, size)


class PixDynamics(nn.Module):

    def __init__(self, pix_mask, n_history, n_filt=32, reg_scale=15.0 / 17.0, learn_registration=True,
                 skip_frames=2, dm_response='interaction_matrix', int_mat=None, dm_mask=None, delay=2, max_lag=3,
                 recon=None, recon_weight=0.0):
        """
        pix_mask       (C, Hp, Wp) bool, valid WFS pixels (the prediction is zero elsewhere)
        dm_response    'interaction_matrix' or 'learned' (see the module doc)
        int_mat        (n_valid_pix, n_valid_act) calibrated interaction matrix, already multiplied by obs_gain
        dm_mask        (h, w) bool, valid actuators
        delay          initial guess of the loop delay in frames; max_lag: the DM may be seen at lags 1..max_lag
        recon          (n_valid_act, n_valid_pix) reconstructor, already divided by obs_gain
        recon_weight   training loss = pixel MSE + recon_weight * MSE after reconstruction. The reconstructor
                       selects the part of the image the DM acts on and the planning loss reads; in the pixel
                       MSE that part is a small fraction, and the optical gain is identified slowly from it.
        """
        super().__init__()
        pix_mask = torch.as_tensor(pix_mask)
        self.register_buffer('pix_mask', pix_mask.float())
        self.recon_weight = recon_weight if recon is not None else 0.0
        if self.recon_weight > 0:
            self.register_buffer('recon', torch.as_tensor(recon).float(), persistent=False)
            self.register_buffer('rec_idx', torch.nonzero(pix_mask.flatten()).squeeze(1), persistent=False)
        C = pix_mask.shape[0]
        self.C, self.skip_frames, self.dm_response, self.max_lag = C, skip_frames, dm_response, max_lag
        if dm_response == 'interaction_matrix':
            assert int_mat is not None and dm_mask is not None and n_history > max_lag
            self.register_buffer('int_mat', torch.as_tensor(int_mat).float(), persistent=False)
            self.register_buffer('dm_mask', torch.as_tensor(dm_mask).bool(), persistent=False)
            self.register_buffer('pix_idx', torch.nonzero(pix_mask.flatten()).squeeze(1), persistent=False)
            logit = torch.full((max_lag,), -2.0)
            logit[delay - 1] = 2.0
            self.lag_logit = nn.Parameter(logit)
            self.optical_gain = nn.Parameter(torch.tensor(1.0))
            n_pol = n_history - max_lag
            self.obs_stem = nn.Sequential(nn.Conv2d((n_history + n_pol) * C, n_filt, 3, stride=2, padding=1), nn.LeakyReLU())
            self.body = nn.Sequential(nn.Conv2d(n_filt, n_filt, 3, padding=1), nn.LeakyReLU(),
                                      nn.Conv2d(n_filt, n_filt, 3, padding=2, dilation=2), nn.LeakyReLU())
            self.skip = nn.Conv2d(2 * skip_frames * C, C, 3, padding=1)
        else:
            self.reg = Registration(reg_scale, learn_registration)
            self.obs_stem = nn.Sequential(nn.Conv2d(n_history * C, n_filt, 3, stride=2, padding=1), nn.LeakyReLU())
            self.act_stem = nn.Sequential(nn.Conv2d(n_history, n_filt, 3, padding=1), nn.LeakyReLU())
            self.body = nn.Sequential(nn.Conv2d(2 * n_filt, n_filt, 3, padding=1), nn.LeakyReLU(),
                                      nn.Conv2d(n_filt, n_filt, 3, padding=2, dilation=2), nn.LeakyReLU())
            self.skip = nn.Conv2d(skip_frames * (C + 1), C, 3, padding=1)
        self.up = nn.ConvTranspose2d(n_filt, C, 4, stride=2, padding=1)

    # ---------------------------------------------------------------- physical DM response
    def dm_seen(self, cmd, frames, shape):
        """Image of the DM as the WFS saw it at the frames j of `frames`: g * IM @ sum_lag w_lag c_{j-lag}.
        cmd (B, T, h, w) with index = time. Returns (B, len(frames), C, Hp, Wp)."""
        L = self.max_lag
        w = torch.softmax(self.lag_logit, 0).flip(0).reshape(1, L, 1, 1)      # a window c_{j-L} ... c_{j-1}: lag L first
        mixed = torch.stack([(cmd[:, j - L:j] * w).sum(1) for j in frames], dim=1)        # (B, K, h, w)
        sig = mixed[:, :, self.dm_mask] @ self.int_mat.T                                   # (B, K, n_valid_pix)
        img = torch.zeros(sig.shape[0], sig.shape[1], self.pix_mask.numel(), device=sig.device)
        img[:, :, self.pix_idx] = sig
        return self.optical_gain * img.reshape(sig.shape[:2] + tuple(shape))

    def forward(self, obs_hist, act_hist, cmd_hist):
        n = obs_hist.shape[1]
        Hp, Wp = obs_hist.shape[-2:]
        s, C = self.skip_frames, self.C
        if self.dm_response == 'interaction_matrix':
            d = self.dm_seen(cmd_hist, range(self.max_lag, n + 1), obs_hist.shape[2:])    # frames L ... n (n = t+1)
            pol = obs_hist[:, self.max_lag:] + d[:, :-1]                                  # (B, n - L, C, Hp, Wp)
            x = torch.cat([obs_hist.flatten(1, 2), pol.flatten(1, 2)], dim=1)
            coarse = self.up(self.body(self.obs_stem(x)))[..., :Hp, :Wp]
            fine = self.skip(torch.cat([obs_hist[:, -s:].flatten(1, 2), pol[:, -s:].flatten(1, 2)], dim=1))
            out = pol[:, -1] + coarse + fine - d[:, -1]
        else:
            x = obs_hist.flatten(1, 2)                                                    # (B, n * C, Hp, Wp)
            f_obs = self.obs_stem(x)
            f_act = self.act_stem(self.reg.to_pix(act_hist, f_obs.shape[-2:]))
            coarse = self.up(self.body(torch.cat([f_obs, f_act], dim=1)))[..., :Hp, :Wp]
            fine = self.skip(torch.cat([x[:, -s * C:], self.reg.to_pix(act_hist[:, -s:], (Hp, Wp))], dim=1))
            out = coarse + fine
        return out * self.pix_mask

    def prediction_loss(self, pred, target):
        err = pred - target
        loss = err.pow(2).mean()
        if self.recon_weight > 0:
            loss = loss + self.recon_weight * (err.flatten(1)[:, self.rec_idx] @ self.recon.T).pow(2).mean()
        return loss

    def describe(self):
        if self.dm_response != 'interaction_matrix':
            return 'DM response learned by the network'
        return 'DM response from the interaction matrix: optical gain %.2f, weight of lags 1..%d: %s' % (
            float(self.optical_gain), self.max_lag,
            ' '.join('%.2f' % v for v in torch.softmax(self.lag_logit, 0).tolist()))


class PixPolicy(nn.Module):

    def __init__(self, n_history, pix_mask, mask, F, n_filt=32, clamp=1.0, reg_scale=15.0 / 17.0,
                 learn_registration=True, recon=None, init_gain=0.4):
        """
        recon   (n_valid_act, n_valid_pix) reconstructor already divided by obs_gain. When given, the policy is
                action = gain * recon @ newest image + network(...), the network starting at zero: it starts
                as the integrator of the warm-up. None: a plain network (random start).
        """
        super().__init__()
        pix_mask = torch.as_tensor(pix_mask)
        mask = torch.as_tensor(mask).bool()
        self.dm_shape = tuple(mask.shape)
        self.reg = Registration(reg_scale, learn_registration)
        self.stem = nn.Sequential(nn.Conv2d(n_history * pix_mask.shape[0], n_filt, 3, stride=2, padding=1), nn.LeakyReLU(),
                                  nn.Conv2d(n_filt, n_filt, 3, stride=2, padding=1), nn.LeakyReLU())
        self.net = nn.Sequential(nn.Conv2d(n_filt + n_history - 1, n_filt, 3, padding=1), nn.LeakyReLU(),
                                 nn.Conv2d(n_filt, n_filt, 3, padding=1), nn.LeakyReLU(),
                                 nn.Conv2d(n_filt, 1, 3, padding=1))
        self.has_skip = recon is not None
        if self.has_skip:
            self.register_buffer('recon', torch.as_tensor(recon).float(), persistent=False)
            self.register_buffer('pix_idx', torch.nonzero(pix_mask.flatten()).squeeze(1), persistent=False)
            self.register_buffer('mask', mask, persistent=False)
            self.gain = nn.Parameter(torch.tensor(float(init_gain)))
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        self.head = ActionHead(mask, F, clamp)

    def forward(self, obs_hist, act_prev, cmd_prev=None):
        f = self.stem(obs_hist.flatten(1, 2))                                # (B, n_filt, Hp / 4, Wp / 4)
        f = self.reg.to_dm(f, self.dm_shape)                                 # features on the DM grid
        raw = self.net(torch.cat([f, act_prev], dim=1))
        if self.has_skip:
            vec = obs_hist[:, -1].flatten(1)[:, self.pix_idx] @ self.recon.T        # reconstructed newest residual
            integ = torch.zeros_like(raw)
            integ[:, 0][:, self.mask] = self.gain * vec
            raw = raw + integ
        return self.head(raw)


class ReconLoss:
    """Planning loss for pixel-space predictions: reconstruct, then the usual PO4AO loss.

        residual = recon @ pred[valid pixels] / obs_gain          (command units, valid actuators)
        loss     = mean(residual^2) + penalty * mean(action^2)

    which is the DM-space loss of the original evaluated on the predicted image. `inner` may be any
    loss from po4ao_reward (it receives the reconstructed residual as a (B, 1, h, w) image).
    space = 'pixel' is the naive alternative, mean(pred^2) * pixel_weight, kept for comparison.
    """

    def __init__(self, recon, pix_mask, dm_mask, obs_gain=1.0, space='recon', pixel_weight=1.0, inner=None):
        self.recon = torch.as_tensor(recon).float()                          # (n_valid, n_sig)
        self.sig_idx = torch.nonzero(torch.as_tensor(pix_mask).flatten()).squeeze(1)
        self.dm_mask = torch.as_tensor(dm_mask).bool()
        self.obs_gain, self.space, self.pixel_weight, self.inner = obs_gain, space, pixel_weight, inner

    def reconstruct(self, pred):
        """(B, C, Hp, Wp) -> (B, 1, h, w) residual in command units."""
        dev = pred.device
        vec = (pred.flatten(1)[:, self.sig_idx.to(dev)] / self.obs_gain) @ self.recon.to(dev).T
        out = torch.zeros((pred.shape[0], 1) + tuple(self.dm_mask.shape), device=dev)
        out[:, 0][:, self.dm_mask.to(dev)] = vec
        return out

    def __call__(self, pred, action, ctx):
        if self.space == 'pixel':
            return self.pixel_weight * pred.pow(2).mean() + ctx['penalty'] * action.pow(2).mean()
        res = self.reconstruct(pred)
        if self.inner is not None:
            return self.inner(res, action, ctx)
        return res.pow(2).mean() + ctx['penalty'] * action.pow(2).mean()
