"""Does a finer grid help to predict frozen flow? A small numerical study (numpy, cpu, ~10 s).

Two separate questions are tested on Kolmogorov screens translated by a fraction of an actuator pitch d.

1. REPRESENTATION. Take the part of the wavefront the DM can correct (spatial frequencies below 1/(2d)).
   Predict its translated version with the best local linear filter (a k x k convolution fitted by least
   squares), on a grid sampled at d (DM grid, critical sampling), d/2 and d/4.
   A translation of a band-limited signal is a sinc interpolation: on a critically sampled grid the ideal
   kernel is wide, so a compact kernel cannot do it; on an oversampled grid it can.

2. INFORMATION. Give the predictor the full-band wavefront (everything the WFS resolves, up to 1/(2 d/4))
   instead of the DM band only, and ask it for the same target (the translated DM-band wavefront).
   Under frozen flow each spatial frequency translates on its own and, for Gaussian turbulence, the
   frequencies are independent: the high frequencies carry no information on the low ones.

Output: fraction of the "do nothing" (persistence) error that remains after prediction; lower is better.
"""

import numpy as np


def screens(n, n_screens, rng, cutoff=None):
    """Periodic Kolmogorov screens (n x n), optionally low-passed at `cutoff` (cycles/pixel, square band)."""
    f = np.fft.fftfreq(n)
    fx, fy = np.meshgrid(f, f)
    k = np.sqrt(fx ** 2 + fy ** 2)
    k[0, 0] = 1.0
    amp = k ** (-11.0 / 6.0)
    amp[0, 0] = 0.0
    out = np.fft.fft2(rng.standard_normal((n_screens, n, n))) * amp
    if cutoff is not None:
        out = out * ((np.abs(fx) <= cutoff) & (np.abs(fy) <= cutoff))
    return out, fx, fy


def shift(F, fx, fy, dx, dy):
    return F * np.exp(-2j * np.pi * (fx * dx + fy * dy))


def fit_and_score(x_train, y_train, x_test, y_test, k):
    """Best k x k linear filter from x to y (least squares on patches), residual variance on the test set."""
    r = k // 2

    def patches(x):
        cols = [np.roll(np.roll(x, -i, axis=1), -j, axis=2).reshape(-1) for i in range(-r, r + 1) for j in range(-r, r + 1)]
        return np.stack(cols, axis=1)

    A = patches(x_train)
    w = np.linalg.lstsq(A, y_train.reshape(-1), rcond=None)[0]
    return np.mean((patches(x_test) @ w - y_test.reshape(-1)) ** 2)


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    over = 4                       # fine grid: 4 samples per actuator pitch (about the PAPYRUS WFS / DM ratio)
    n = 256                        # fine pixels -> 64 actuators across
    dm_cut = 1.0 / (2 * over)      # DM Nyquist frequency in cycles per fine pixel

    F_dm, fx, fy = screens(n, 12, rng, cutoff=dm_cut)
    rng = np.random.default_rng(0)
    F_full, _, _ = screens(n, 12, rng)          # same random numbers: F_dm is the low-passed F_full

    print('remaining fraction of the persistence error (0 = perfect prediction, 1 = no better than doing nothing)\n')
    for shift_d in [0.1, 0.2, 0.5]:
        s = shift_d * over
        tgt = np.fft.ifft2(shift(F_dm, fx, fy, s, 0.6 * s)).real          # translated DM-band wavefront
        src = np.fft.ifft2(F_dm).real
        full = np.fft.ifft2(F_full).real
        tr, te = slice(0, 8), slice(8, 12)
        print('translation of %.1f actuator pitch (and 0.6 of that in the other axis)' % shift_d)
        for step, ks in [(4, [3, 5]), (2, [3, 5, 9]), (1, [3, 9, 17])]:
            a, b = src[:, ::step, ::step], tgt[:, ::step, ::step]
            pers = np.mean((a[te] - b[te]) ** 2)
            res = ['%d x %d: %.3f' % (k, k, fit_and_score(a[tr], b[tr], a[te], b[te], k) / pers) for k in ks]
            print('   1. DM-band input, grid = d/%d   %s' % (4 // step, '   '.join(res)))
        b = tgt
        pers = np.mean((src[te] - b[te]) ** 2)
        for name, a in [('DM-band input  ', src), ('full-band input', full)]:
            print('   2. %s grid = d/4, 9 x 9: %.3f' % (name, fit_and_score(a[tr], b[tr], a[te], b[te], 9) / pers))
        print()
