"""
Temporal correlation curves of the generated 2nd stage atmospheres (same frames as atm_movie.py).

C(tau) = <phi(t) phi(t+tau)>_{x,t} / <phi^2>_{x,t}, computed with an FFT along time for every pixel.
Two versions: piston removed per frame, and piston + tip/tilt removed per frame.

OOPAO boiling is an AR(1) process on every layer (alpha = exp(-dt/t_boiling)), applied on top of
the frozen flow, so the expected curve is C_frozen(tau) * exp(-tau/tboil). C_frozen is taken from
the tboil = 0 (no boiling) atmosphere and drawn dashed over each boiling curve.
"""

import os
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from atm_movie import atm_files, comparison_sets, load_first_frames, n_frames, samplingTime, base_dir

max_lag_s = 10
D_tel     = 1.52 * 80 / 76        # param['diameter'], tel_sky_2 (90 px across)
windSpeed = np.array([0.88, 0.99, 2.79, 5.37, 4.25])
fractionalR0 = np.array([0.45, 0.1, 0.1, 0.25, 0.1])

save_dir      = f"{base_dir}/atm_comparison_plots"
save_filename = "atm_2nd_tboil_temporal_correlation.png"
save_dpi      = 200


def remove_tip_tilt(opd):
    n = opd.shape[1]
    x = np.arange(n) - (n - 1) / 2
    X, Y = np.meshgrid(x, x)
    tip  = np.einsum("tij,ij->t", opd, X) / (X**2).sum()
    tilt = np.einsum("tij,ij->t", opd, Y) / (Y**2).sum()
    return opd - tip[:, None, None] * X - tilt[:, None, None] * Y


def temporal_correlation(opd, max_lag):
    """Normalised autocorrelation along time, summed over pixels (unbiased in lag)."""
    n_t = opd.shape[0]
    flat = opd.reshape(n_t, -1).astype(np.float64)
    n_fft = 1 << int(np.ceil(np.log2(2 * n_t)))
    acov = np.zeros(n_fft)
    for k in range(0, flat.shape[1], 1000):
        F = np.fft.rfft(flat[:, k:k + 1000], n=n_fft, axis=0)
        acov += np.fft.irfft((F * F.conj()).real.sum(axis=1), n=n_fft)
    acov = acov[:max_lag + 1] / (n_t - np.arange(max_lag + 1))
    return acov / acov[0]


def correlation_job(tboil):
    opd, _ = load_first_frames(atm_files[tboil], n_frames)   # piston already removed
    max_lag = int(round(max_lag_s / samplingTime))
    return (temporal_correlation(opd, max_lag),
            temporal_correlation(remove_tip_tilt(opd), max_lag))


def decorrelation_time(lags, C, level):
    below = np.nonzero(C < level)[0]
    return lags[below[0]] if below.size else np.nan


if __name__ == "__main__":
    os.makedirs(save_dir, exist_ok=True)
    tboils = comparison_sets["tboil_0.1-5"]   # decreasing boiling, no boiling (tboil = 0) last
    with Pool(len(tboils)) as pool:
        results = dict(zip(tboils, pool.map(correlation_job, tboils)))
    lags = np.arange(int(round(max_lag_s / samplingTime)) + 1) * samplingTime

    np.savez(os.path.join(save_dir, save_filename.replace(".png", ".npz")), lags=lags, tboils=tboils,
             C_piston_removed=np.array([results[t][0] for t in tboils]),
             C_ptt_removed=np.array([results[t][1] for t in tboils]))

    # pupil crossing times for the frozen flow
    crossing = D_tel / windSpeed
    v_mean = (fractionalR0 * windSpeed).sum()
    print("pupil crossing time D/v per layer [s]:", np.round(crossing, 2),
          f"| D / (Cn2-weighted mean v = {v_mean:.2f} m/s) = {D_tel / v_mean:.2f} s")

    for k, label in enumerate(["piston removed", "piston + tip/tilt removed"]):
        print(f"\n{label}")
        print(f"{'tboil':>8} {'tau(C=1/e)':>11} {'tau(C=0.5)':>11} {'C(0.5 s)':>9} {'C(2 s)':>8}")
        for t in tboils:
            C = results[t][k]
            print(f"{t:8g} {decorrelation_time(lags, C, 1/np.e):11.3f} {decorrelation_time(lags, C, 0.5):11.3f} "
                  f"{C[int(round(0.5 / samplingTime))]:9.3f} {C[int(round(2 / samplingTime))]:8.3f}")

    # ordinal blue ramp: light = strong boiling -> dark = slow boiling, near-black = frozen flow
    colors = dict(zip(tboils, ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#2b2b2b"]))

    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, k, title in zip(axs, [0, 1], ["piston removed", "piston + tip/tilt removed"]):
        C_frozen = results[0][k]
        for t in tboils:
            label = "tboil = 0 (frozen flow)" if t == 0 else f"tboil = {t:g} s"
            ax.plot(lags[1:], results[t][k][1:], color=colors[t], lw=2, label=label)
            if t > 0:
                ax.plot(lags[1:], C_frozen[1:] * np.exp(-lags[1:] / t), color=colors[t], lw=1.2, ls="--")
        ax.plot([], [], color="0.4", lw=1.2, ls="--", label=r"$C_\mathrm{frozen}(\tau)\,e^{-\tau/t_\mathrm{boil}}$")
        ax.axvline(D_tel / v_mean, color="0.55", lw=1, ls=":")
        ax.text(D_tel / v_mean * 1.06, 0.93, f"D / v = {D_tel / v_mean:.2f} s", color="0.35", fontsize=9)
        ax.axhline(0, color="0.75", lw=0.8)
        ax.axhline(1 / np.e, color="0.75", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xlim(samplingTime, max_lag_s)
        ax.set_ylim(-0.2, 1.02)
        ax.set_title(title)
        ax.set_xlabel(r"lag $\tau$ (s)")
        ax.grid(True, which="both", alpha=0.25)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    axs[0].set_ylabel(r"$C(\tau) = \langle\phi(t)\,\phi(t+\tau)\rangle / \langle\phi^2\rangle$")
    axs[0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle(f"2nd stage atmosphere temporal correlation, r0 = 0.07 m, V0 = 3 m/s, L0 = 30 m, "
                 f"{n_frames} frames at {1/samplingTime:.0f} Hz, D = {D_tel:.2f} m", fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, save_filename), dpi=save_dpi)
    plt.close(fig)
    print("saved", os.path.join(save_dir, save_filename))
