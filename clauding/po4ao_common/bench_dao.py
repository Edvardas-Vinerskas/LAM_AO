"""PAPYRUS through DAO shared memory, with the interface of toy_env.ToyAOEnv.

NOT TESTED: it was written off the bench, from po4ao_PAPYRUS_upd.py. The DM-space part (step, flatten,
sample_noise) is a transcription of the original functions with three changes:
  * the command sent to the DM is the clamped one (the original stored the clamped command but sent the
    unclamped one),
  * leak and clamp come from the config (the original hard-coded 0.99 and 0.8),
  * the total command is returned with every observation (obs.cmd).

Pixel space needs three things this file cannot know and that must be supplied in cfg['bench']:
    'pix_reader'   callable() -> torch tensor (C, Hp, Wp) on the control device: the WFS frame that belongs
                   to the residual just read, cropped to the pupil image(s), reference-subtracted and
                   flux-normalised exactly as the pipeline does before its reconstructor.
                   For a pyramid, po4ao_pix.po4ao_models_pix.PyramidFrames does this on the raw camera
                   frame: pix_reader = lambda: pyr(camera_shm.get_data()) gives the four pupil images
                   (C = 4), divided by the mean flux and minus the flat-wavefront reference.
    'pix_mask'     (C, Hp, Wp) bool array, valid pixels (PyramidFrames.mask)
    'pix_recon'    (241, n_valid_pixels) matrix: commands = pix_recon @ frame[pix_mask]
                   (M2C @ command matrix of the pipeline), used only by the planning loss
    'pix_im'       (n_valid_pixels, 241) matrix: frame[pix_mask] = pix_im @ commands, the calibrated
                   interaction matrix expressed on the actuators (interaction matrix @ pinv(M2C))
"""

from collections import namedtuple

import numpy as np
import torch

Obs = namedtuple('Obs', ['dm', 'pix', 'cmd'])

SHM = {
    'm2c': '/tmp/papyrus_M2C.im.shm',
    'dm_map': '/tmp/dm241Map.im.shm',
    'res_wf': '/tmp/papyrus_res_wf.im.shm',
    'dm_cmd': '/tmp/dmCmd03.im.shm',
}


class PapyrusBench:

    def __init__(self, cfg, device='cuda:0', sem_nb=5):
        import dao
        b = cfg.get('bench', {})
        names = dict(SHM, **b.get('shm', {}))
        self.device = torch.device(device)
        self.sem_nb = sem_nb
        self.leak = cfg['integrator']['leak']
        self.clamp = cfg['integrator']['command_clamp']
        nmodes = cfg['integrator']['n_modes']

        m2c = dao.shm(names['m2c']).get_data()
        valid = dao.shm(names['dm_map']).get_data()
        self.frame_shm = dao.shm(names['res_wf'])
        self.dm_shm = dao.shm(names['dm_cmd'])

        self.mask = torch.from_numpy(np.asarray(valid).astype(bool)).to(self.device)
        self.n_act = self.mask.shape[0]
        self.dm_x, self.dm_y = torch.where(self.mask)          # row-major, same order as np.where
        self.n_valid = int(self.mask.sum())
        proj = m2c[:, :nmodes] @ np.linalg.pinv(m2c[:, :nmodes])
        self.KL_projection = torch.from_numpy(np.asarray(proj)).float().to(self.device)

        self.cmd = torch.zeros(self.n_act, self.n_act, device=self.device)
        self.pix_reader = b.get('pix_reader')
        self._pix_mask = b.get('pix_mask')
        self._pix_recon = b.get('pix_recon')
        self._pix_im = b.get('pix_im')

    def _read(self):
        dm = torch.zeros(self.n_act, self.n_act, device=self.device)
        dm[self.dm_x, self.dm_y] = torch.from_numpy(
            self.frame_shm.get_data(check=True, semNb=self.sem_nb)).flatten().float().to(self.device)
        pix = self.pix_reader() if self.pix_reader is not None else None
        return Obs(-dm, pix, self.cmd.clone())

    @torch.no_grad()
    def flatten(self):
        self.cmd.zero_()
        self.dm_shm.set_data(self.cmd[self.dm_x, self.dm_y].cpu().numpy())
        return self._read()

    @torch.no_grad()
    def step(self, action):
        self.cmd = (self.leak * self.cmd + action.reshape(self.n_act, self.n_act) * self.mask).clamp(-self.clamp, self.clamp)
        self.dm_shm.set_data(self.cmd[self.dm_x, self.dm_y].cpu().numpy())
        return self._read()

    @torch.no_grad()
    def sample_noise(self, sigma):
        signs = torch.sign(torch.randn(self.n_valid, device=self.device))
        out = torch.zeros(self.n_act, self.n_act, device=self.device)
        out[self.dm_x, self.dm_y] = self.KL_projection @ (sigma * signs)
        return out

    def reconstructor(self):
        if self._pix_recon is None or self._pix_mask is None:
            raise RuntimeError("pixel space needs cfg['bench']['pix_recon'] and ['pix_mask'] (see the module doc)")
        return (torch.as_tensor(self._pix_recon).float().to(self.device),
                torch.as_tensor(self._pix_mask).bool().to(self.device))

    def interaction_matrix(self):
        if self._pix_im is None:
            raise RuntimeError("cfg['bench']['pix_im'] is missing (or set cfg['pix']['use_interaction_matrix'] = False)")
        return torch.as_tensor(self._pix_im).float().to(self.device)
