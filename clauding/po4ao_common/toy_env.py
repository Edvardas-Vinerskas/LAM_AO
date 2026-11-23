"""Toy single-conjugate AO loop for developing PO4AO variants without the bench or OOPAO.

It is NOT an end-to-end simulator (no diffraction, the WFS is a geometric gradient sensor with a
saturation that mimics the optical gains of a pyramid). What it does reproduce is what the controller cares about:

  * multi-layer frozen-flow von Karman turbulence (exact sub-pixel translation by Fourier shift,
    optional slow "boiling" so that the screens never repeat),
  * a DM with Gaussian influence functions on an n_act x n_act grid (17 x 17, 241 valid, as PAPYRUS),
  * a WFS sampled n_pix across the pupil (76, as PAPYRUS) giving a 2-channel slopes image,
  * the bench's own leaky integration of the action, command clamp and loop delay,
  * the bench's linear reconstruction to DM space (interaction matrix of the controlled modes, pinv).

Interface (same as bench_dao.PapyrusBench):

    obs = env.flatten()            # zero the DM, read a frame
    obs = env.step(action)         # action = change of command, (n_act, n_act) image
    obs.dm   (n_act, n_act)        reconstructed residual in command units (what PO4AO sees today),
                                   sign such that  action = gain * obs.dm  is the integrator
    obs.pix  (2, n_pix, n_pix)     WFS slopes image (x and y), zero outside the pupil
    obs.cmd  (n_act, n_act)        total command now held by the integrator (after this action)
    env.info                       ground truth of the frame just measured (residual variance, Strehl)

Time line of step(a_t):  c_t = clamp(leak * c_{t-1} + a_t);  the atmosphere moves one frame;
the WFS measures  phi_atm(t+1) - DM(c_{t+1-delay}).  delay = 2 is the usual two-frame loop.
"""

import math
from collections import deque, namedtuple

import numpy as np
import torch

Obs = namedtuple('Obs', ['dm', 'pix', 'cmd'])


class ToyAOEnv:

    def __init__(self, n_act=17, n_pix=76, n_modes=195, diameter=1.52, r0=0.15, L0=20.0,
                 wind_speed=(8.0, 15.0, 30.0), wind_dir=(20.0, 110.0, 200.0), cn2=(0.6, 0.3, 0.1),
                 frequency=1000.0, delay=2, leak=0.99, clamp=0.8, cmd_scale=25.0,
                 modulation=5.0, noise_std=0.05, boiling=2e-5, coupling=0.35,
                 screen_size=512, seed=0, device=None):
        """
        r0            Fried parameter at the WFS wavelength [m]
        wind_speed    [m/s] per layer, wind_dir [deg], cn2 = fraction of the turbulence per layer
        delay         loop delay in frames (>= 1)
        cmd_scale     radians of phase at the peak of an actuator for a command of 1
                      (chosen so that commands live in about [-0.8, 0.8] like the DM voltages on the bench)
        modulation    WFS saturation, in lambda/D of tilt (None = perfectly linear gradient sensor)
        noise_std     WFS noise, rad/pixel rms on each slope
        boiling       per-frame innovation variance fraction of each screen (0 = pure frozen flow)
        """
        self.device = torch.device(device if device is not None else ('cuda:0' if torch.cuda.is_available() else 'cpu'))
        dev = self.device
        self.n_act, self.n_pix, self.n_modes = n_act, n_pix, n_modes
        self.delay, self.leak, self.clamp, self.cmd_scale = int(delay), leak, clamp, cmd_scale
        self.noise_std, self.boiling = noise_std, boiling
        self.frequency = frequency
        self.pix_size = diameter / n_pix
        # two generators: the turbulence must not depend on how many noise samples the controller consumed,
        # so that two controllers run with the same seed see exactly the same atmosphere
        self.gen = torch.Generator(device=dev)
        self.gen.manual_seed(seed)
        self.gen_noise = torch.Generator(device=dev)
        self.gen_noise.manual_seed(seed + 12345)

        # ---------------------------------------------------------------- pupil
        ax = (torch.arange(n_pix, device=dev) - (n_pix - 1) / 2)
        yy, xx = torch.meshgrid(ax, ax, indexing='ij')
        self.pupil = (xx ** 2 + yy ** 2) <= (n_pix / 2) ** 2
        self.n_pup = int(self.pupil.sum())

        # ---------------------------------------------------------------- deformable mirror
        # the pupil spans (n_act - 2) pitches: the outer ring of actuators sits just outside the pupil
        self.pitch = n_pix / (n_act - 2)
        aax = (torch.arange(n_act, device=dev) - (n_act - 1) / 2)
        ay, axx = torch.meshgrid(aax, aax, indexing='ij')
        self.mask = (axx ** 2 + ay ** 2) <= ((n_act - 1) / 2 + 0.75) ** 2      # 241 valid actuators for n_act = 17
        self.dm_x, self.dm_y = torch.where(self.mask)
        self.n_valid = int(self.mask.sum())
        cx = axx[self.mask] * self.pitch
        cy = ay[self.mask] * self.pitch
        d2 = (xx.reshape(-1, 1) - cx.reshape(1, -1)) ** 2 + (yy.reshape(-1, 1) - cy.reshape(1, -1)) ** 2
        self.IF = torch.exp(math.log(coupling) * d2 / self.pitch ** 2) * cmd_scale      # (n_pix^2, n_valid) rad / unit command
        IFp = self.IF[self.pupil.flatten()]                                             # (n_pup, n_valid)
        self.IFp = IFp

        # controlled modes: eigenmodes of the piston-removed DM geometric covariance, smooth ones first
        IFc = (IFp - IFp.mean(0, keepdim=True)).double()
        evals, evecs = torch.linalg.eigh(IFc.T @ IFc)
        order = torch.argsort(evals, descending=True)
        self.M2C = evecs[:, order[:n_modes]].float().contiguous()                       # (n_valid, n_modes), orthonormal
        self.KL_projection = (self.M2C @ self.M2C.T).contiguous()                       # = M2C @ pinv(M2C)
        self.fit_proj = torch.linalg.pinv(IFc).float()                                  # phase (pupil) -> best DM command

        # ---------------------------------------------------------------- wavefront sensor
        # slopes = finite differences between neighbouring pupil pixels, on the n_pix grid
        p = self.pupil
        self.valid_x = torch.zeros_like(p)
        self.valid_x[:, :-1] = p[:, :-1] & p[:, 1:]
        self.valid_y = torch.zeros_like(p)
        self.valid_y[:-1, :] = p[:-1, :] & p[1:, :]
        self.sig_mask = torch.stack([self.valid_x, self.valid_y])                       # (2, n_pix, n_pix)
        self.n_sig = int(self.sig_mask.sum())
        self.sig_idx = torch.nonzero(self.sig_mask.flatten()).squeeze(1)
        self.pup_idx = torch.nonzero(self.pupil.flatten()).squeeze(1)
        self.sig_mask_f = self.sig_mask.float()
        self.mod = None if modulation is None else 2 * math.pi * modulation / n_pix     # rad / pixel

        # interaction matrix of the controlled modes (push-pull, linear regime, no noise) and reconstructor
        amp = 1e-3
        IM = torch.zeros(self.n_sig, n_modes, device=dev, dtype=torch.float64)
        for i in range(n_modes):
            ph = (self.IF @ self.M2C[:, i]).reshape(n_pix, n_pix) * amp
            IM[:, i] = ((self._slopes(ph, linear=True) - self._slopes(-ph, linear=True))[self.sig_mask] / (2 * amp)).double()
        # WFS signal (valid slopes) -> command on the valid actuators.  obs.dm = recon @ signal
        self.recon = (self.M2C.double() @ torch.linalg.pinv(IM)).float().contiguous()   # (n_valid, n_sig)
        # calibrated response of the WFS image to a command in the controlled subspace: signal = int_mat @ command
        self.int_mat = (IM @ self.M2C.double().T).float().contiguous()                  # (n_sig, n_valid)

        # ---------------------------------------------------------------- atmosphere
        N = screen_size
        self.N = N
        f = torch.fft.fftfreq(N, d=self.pix_size, device=dev)                           # cycles / m
        fy, fx = torch.meshgrid(f, f, indexing='ij')
        psd = 0.0229 * r0 ** (-5.0 / 3.0) * (fx ** 2 + fy ** 2 + 1.0 / L0 ** 2) ** (-11.0 / 6.0)
        psd[0, 0] = 0.0
        df = 1.0 / (N * self.pix_size)
        self.amp_f = (N * torch.sqrt(psd) * df).float()                                 # screen = Re ifft2(N * white * amp_f)
        self.kx = torch.fft.fftfreq(N, device=dev).reshape(1, N)                        # cycles / pixel
        self.ky = torch.fft.fftfreq(N, device=dev).reshape(N, 1)
        self.cn2 = torch.tensor(cn2, device=dev, dtype=torch.float32)
        self.cn2 = self.cn2 / self.cn2.sum()
        ws = torch.tensor(wind_speed, device=dev, dtype=torch.float32)
        wd = torch.deg2rad(torch.tensor(wind_dir, device=dev, dtype=torch.float32))
        # pixels per frame
        self.vx = ws * torch.cos(wd) / frequency / self.pix_size
        self.vy = ws * torch.sin(wd) / frequency / self.pix_size
        self.n_layers = len(wind_speed)
        self.layer_amp = (N * self.amp_f).unsqueeze(0) * torch.sqrt(self.cn2).reshape(-1, 1, 1)   # (L, N, N)
        self.F = self._new_screens_f()                                                  # (L, N, N) complex
        self.pos = torch.zeros(self.n_layers, 2, device=dev)                            # accumulated shift per layer [pixels]

        # ---------------------------------------------------------------- loop state
        self.cmd = torch.zeros(n_act, n_act, device=dev)
        self.queue = deque([torch.zeros(self.n_valid, device=dev) for _ in range(self.delay - 1)])
        self.info = {}
        self.t = 0

    # ==================================================================== atmosphere
    def _new_screens_f(self):
        # complex white noise shaped by the von Karman spectrum, one screen per layer. The field is not
        # Hermitian: its real part (the only part used) is a screen with the right spectrum, and translating
        # the complex field translates its real part, so no FFT is needed to draw a screen.
        w = torch.randn(2, self.n_layers, self.N, self.N, device=self.device, generator=self.gen)
        return torch.complex(w[0], w[1]) * self.layer_amp

    def _advance(self):
        self.pos[:, 0] += self.vx
        self.pos[:, 1] += self.vy
        if self.boiling > 0:
            self.F.mul_(math.sqrt(1.0 - self.boiling)).add_(self._new_screens_f(), alpha=math.sqrt(self.boiling))
        self.t += 1

    def atm_phase(self):
        """Turbulent phase over the n_pix x n_pix grid [rad] at the current time."""
        ramp = torch.exp(-2j * math.pi * (self.kx.unsqueeze(0) * self.pos[:, 0].reshape(-1, 1, 1)
                                          + self.ky.unsqueeze(0) * self.pos[:, 1].reshape(-1, 1, 1)))
        screen = torch.fft.ifft2((self.F * ramp).sum(0)).real
        return screen[:self.n_pix, :self.n_pix]

    # ==================================================================== wavefront sensor
    def _slopes(self, phase, linear=False):
        s = torch.zeros(2, self.n_pix, self.n_pix, device=self.device)
        s[0, :, :-1] = phase[:, 1:] - phase[:, :-1]
        s[1, :-1, :] = phase[1:, :] - phase[:-1, :]
        if self.mod is not None and not linear:
            # Saturating response: unit gain for small local tilts, saturation at the modulation radius m.
            # Because the residual always carries high spatial frequencies the DM cannot correct, the
            # sensor works part-way into the saturation and its sensitivity to a DM poke drops below the
            # calibrated one: the "optical gain" of a pyramid on sky (about 0.7 with the defaults). tanh is
            # a stand-in for that behaviour, not a model of a pyramid (the purely geometric response of a
            # modulated pyramid, arcsin, has a gain ABOVE one before it clips, which real sensors do not show).
            s = self.mod * torch.tanh(s / self.mod)
        return s * self.sig_mask_f

    def _measure(self, applied):
        atm = self.atm_phase()
        res = atm - (self.IF @ applied).reshape(self.n_pix, self.n_pix)
        pix = self._slopes(res)
        if self.noise_std > 0:
            pix = pix + self.noise_std * torch.randn(pix.shape, device=self.device, generator=self.gen_noise) * self.sig_mask_f
        dm = torch.zeros(self.n_act, self.n_act, device=self.device)
        dm[self.dm_x, self.dm_y] = self.recon @ pix.flatten()[self.sig_idx]
        rp = res.flatten()[self.pup_idx]
        ap = atm.flatten()[self.pup_idx]
        var, atm_var = torch.stack([rp.var(unbiased=False), ap.var(unbiased=False)]).tolist()
        self.info = {'res_var': var, 'strehl': math.exp(-var), 'atm_var': atm_var}
        self._last_atm = atm
        return Obs(dm, pix, self.cmd.clone())

    def fitting_var(self):
        """Residual variance [rad^2] of the best possible DM fit of the last measured turbulent phase."""
        ap = self._last_atm[self.pupil]
        ap = ap - ap.mean()
        fit = ap - self.IFp @ (self.fit_proj @ ap)
        fit = fit - fit.mean()
        return float((fit ** 2).mean())

    # ==================================================================== loop
    @torch.no_grad()
    def flatten(self):
        self.cmd.zero_()
        self.queue = deque([torch.zeros(self.n_valid, device=self.device) for _ in range(self.delay - 1)])
        return self._measure(torch.zeros(self.n_valid, device=self.device))

    @torch.no_grad()
    def step(self, action):
        act = action.reshape(self.n_act, self.n_act) * self.mask
        self.cmd = (self.leak * self.cmd + act).clamp(-self.clamp, self.clamp)
        self.queue.append(self.cmd[self.dm_x, self.dm_y].clone())
        applied = self.queue.popleft()
        self._advance()
        return self._measure(applied)

    @torch.no_grad()
    def sample_noise(self, sigma):
        """Exploration noise of the warm-up: random signs on the actuators, filtered by the controlled modes."""
        signs = torch.sign(torch.randn(self.n_valid, device=self.device, generator=self.gen_noise))
        out = torch.zeros(self.n_act, self.n_act, device=self.device)
        out[self.dm_x, self.dm_y] = self.KL_projection @ (sigma * signs)
        return out

    # what a pixel-space loss needs to get back to DM space:  dm[valid] = recon @ pix[sig_mask]
    def reconstructor(self):
        return self.recon, self.sig_mask

    # calibrated interaction matrix in pixel space:  pix[sig_mask] = interaction_matrix() @ cmd[valid]
    def interaction_matrix(self):
        return self.int_mat

    # ==================================================================== branching (for response tests)
    def snapshot(self):
        """Full state, random generators included: after restore() the same actions give the same frames."""
        return {'F': self.F.clone(), 'pos': self.pos.clone(), 'cmd': self.cmd.clone(),
                'queue': [q.clone() for q in self.queue], 't': self.t,
                'gen': self.gen.get_state(), 'gen_noise': self.gen_noise.get_state()}

    def restore(self, snap):
        self.F = snap['F'].clone()
        self.pos = snap['pos'].clone()
        self.cmd = snap['cmd'].clone()
        self.queue = deque([q.clone() for q in snap['queue']])
        self.t = snap['t']
        self.gen.set_state(snap['gen'])
        self.gen_noise.set_state(snap['gen_noise'])
