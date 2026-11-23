"""Replay buffer for PO4AO.

Differences with EfficientExperienceReplay (po4ao_util_PAPYRUS.py):

  * observations and actions may have different shapes (WFS pixels in, DM commands out),
  * the total DM command is stored next to the action (needed by the pseudo-open-loop variants),
  * next_states is not stored (it was never read back: the next state is the next entry),
  * every entry carries a time stamp, and sample() only returns windows whose stamps are consecutive.
    The original sampler can return a window that straddles the write pointer once the buffer has
    wrapped (newest frames followed by the oldest ones) or, if episodes have unequal lengths, two
    episodes; such a window teaches the dynamics a jump that never happened.
"""

import torch


class Replay:

    def __init__(self, obs_shape, act_shape, max_size, device='cuda:0', obs_dtype=torch.float32):
        self.max_size = int(max_size)
        self.device = device
        self.obs = torch.zeros((self.max_size,) + tuple(obs_shape), dtype=obs_dtype, device=device)
        self.act = torch.zeros((self.max_size,) + tuple(act_shape), device=device)
        self.cmd = torch.zeros((self.max_size,) + tuple(act_shape), device=device)
        self.stamp = torch.full((self.max_size,), -1, dtype=torch.long)       # cpu
        # [length, write pointer, clock]: a tensor so that it can be shared with the training process
        self.state = torch.zeros(3, dtype=torch.long)

    def share_memory(self):
        for t in (self.obs, self.act, self.cmd):
            if not t.is_cuda:           # CUDA tensors are shared by handle when the object is sent to the process
                t.share_memory_()
        self.stamp.share_memory_()
        self.state.share_memory_()
        return self

    def __len__(self):
        return int(self.state[0])

    def new_segment(self):
        """Call when the time series is interrupted (DM flattened, loop opened, frames dropped...)."""
        self.state[2] += 1000000

    def append(self, obs, act, cmd):
        i = int(self.state[1])
        self.obs[i] = obs
        self.act[i] = act
        self.cmd[i] = cmd
        self.stamp[i] = self.state[2]
        self.state[2] += 1
        self.state[1] = (i + 1) % self.max_size
        self.state[0] = min(int(self.state[0]) + 1, self.max_size)

    def fill(self, obs, act, cmd):
        """Append a whole recorded sequence (T, ...) as one segment."""
        self.new_segment()
        for i in range(len(obs)):
            self.append(obs[i], act[i], cmd[i])

    def can_sample(self, window):
        return len(self) > 4 * window

    def sample(self, window, batch_size):
        """
        :return obs (B, window, *obs_shape) float32, act (B, window, *act_shape), cmd (B, window, *act_shape),
                consecutive frames, oldest first
        """
        n = len(self)
        starts = []
        n_found = 0
        for _ in range(100):
            s = torch.randint(0, n, (4 * batch_size,))
            e = (s + window - 1) % self.max_size
            ok = (self.stamp[e] - self.stamp[s] == window - 1) & (self.stamp[s] >= 0)
            if n < self.max_size:
                ok &= (s + window - 1) < n
            s = s[ok]
            starts.append(s)
            n_found += len(s)
            if n_found >= batch_size:
                break
        if n_found < batch_size:
            raise RuntimeError('replay buffer: no contiguous window of %d frames' % window)
        starts = torch.cat(starts)[:batch_size]
        idx = ((starts.unsqueeze(1) + torch.arange(window).unsqueeze(0)) % self.max_size).to(self.obs.device)
        return self.obs[idx].float(), self.act[idx], self.cmd[idx]

    def state_dict(self):
        return {'obs': self.obs, 'act': self.act, 'cmd': self.cmd, 'stamp': self.stamp, 'state': self.state}

    def load_state_dict(self, d):
        self.obs.copy_(d['obs'])
        self.act.copy_(d['act'])
        self.cmd.copy_(d['cmd'])
        self.stamp.copy_(d['stamp'])
        self.state.copy_(d['state'])
