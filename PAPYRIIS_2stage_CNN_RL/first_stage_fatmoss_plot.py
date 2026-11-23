"""
Summary + figures for first_stage_fatmoss.py: perfect-WFS gain sweeps vs the PWFS loop (gain 0.5) on the
FATMOSS atmospheres. Residuals in nm rms over the 1st-stage pupil; SR via Marechal at 640 nm (NGS) and 1.6 um
(2nd stage science band). Run from /home/evinerskas/LAM_OOPAO.
"""
import os
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from first_stage_fatmoss import GAIN_PWFS, out_dir, perfect_dir, tag

Bs = [0, 200, 400, 1000]
colors = {0: "#2a78d6", 200: "#eb6834", 400: "#1baf7a", 1000: "#eda100"}   # categorical slots 1-4
ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
skip = 200      # convergence transient excluded from the means
fig_dir = f"{out_dir}/figures"


def strehl(rms, wl):
    return np.exp(-(2 * np.pi * rms / wl) ** 2)


def pwfs_rms(B, chunk=2000):
    """Per-frame residual rms [m] of the PWFS loop, streamed from the uncompressed results npz."""
    path = f"{out_dir}/results_1st_stage_{tag(B)}_multi_layer.npz"
    cache = path.replace(".npz", "_rms.npy")
    if os.path.exists(cache):
        return np.load(cache)
    if not os.path.exists(path):
        return None
    with zipfile.ZipFile(path) as z:
        with z.open("telescope_pupil.npy") as f:
            pupil = np.load(f).astype(bool)
        with z.open("src_opds.npy") as f:
            version = np.lib.format.read_magic(f)
            hdr = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
            shape, _, dtype = hdr(f)
            frame = int(np.prod(shape[1:]))
            out = np.zeros(shape[0])
            for s in range(0, shape[0], chunk):
                n = min(chunk, shape[0] - s)
                a = np.frombuffer(f.read(n * frame * dtype.itemsize), dtype=dtype).reshape(n, *shape[1:])
                out[s:s + n] = a[:, pupil].std(axis=1)
    np.save(cache, out)
    return out


def style(ax):
    ax.grid(True, color=grid, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(ink2)
    ax.tick_params(colors=ink2, labelsize=9)


if __name__ == "__main__":
    os.makedirs(fig_dir, exist_ok=True)
    perf = {}
    for B in Bs:
        fs = [f for f in os.listdir(perfect_dir) if f.startswith(f"results_1st_stage_perfect_{tag(B)}_")]
        if fs:
            perf[B] = np.load(f"{perfect_dir}/{fs[0]}")
    pw = {B: r for B in Bs if (r := pwfs_rms(B)) is not None}

    # ---- summary table
    rows = []
    print(f"{'B [Hz]':>7} {'frames':>7} {'fit':>6} | {'perfect g=0.5':>13} {'best perfect (g)':>17} {'full-rate best':>15} | "
          f"{'PWFS g=0.5':>10} | SR640 perf/PWFS   SR1.6um perf/PWFS")
    for B in Bs:
        if B not in perf:
            continue
        p = perf[B]
        g = p["gains_navg2"]
        m2 = np.nanmean(p["rms_navg2"][:, skip:], axis=1) * 1e9
        m1 = np.nanmean(p["rms_navg1"][:, skip:], axis=1) * 1e9
        i05, ib = int(np.argmin(np.abs(g - GAIN_PWFS))), int(np.nanargmin(m2))
        fit = p["fit_rms_navg2"].mean() * 1e9
        line = f"{B:7d} {p['rms_navg2'].shape[1]:7d} {fit:6.1f} | {m2[i05]:13.1f} {m2[ib]:11.1f} ({g[ib]:.1f}) {np.nanmin(m1):11.1f} ({g[np.nanargmin(m1)]:.1f}) |"
        if B in pw:
            mp = pw[B][skip:].mean() * 1e9
            sp = [np.mean(strehl(pw[B][skip:], wl)) for wl in (640e-9, 1.6e-6)]
            ss = [np.nanmean(strehl(p["rms_navg2"][i05, skip:], wl)) for wl in (640e-9, 1.6e-6)]
            line += f" {mp:10.1f} | {ss[0]:.3f} / {sp[0]:.3f}    {ss[1]:.3f} / {sp[1]:.3f}"
        print(line)

    # ---- fig 1: residual vs gain, perfect WFS (two timings), PWFS at its gain
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, n_avg, title in zip(axes, (2, 1), ("Perfect WFS, PWFS timing (2-frame avg + 1 period delay)",
                                               "Perfect WFS, full rate (no averaging, 1 frame delay)")):
        style(ax)
        for B, p in perf.items():
            g = p[f"gains_navg{n_avg}"]
            m = np.nanmean(p[f"rms_navg{n_avg}"][:, skip:], axis=1) * 1e9
            ax.plot(g, m, "-o", color=colors[B], lw=2, ms=6, mec="white", mew=1.5, label=f"B = {B} Hz")
            ax.annotate(f"B = {B}", (g[-1], m[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color=ink2)
            if n_avg == 2 and B in pw:
                ax.plot(GAIN_PWFS, pw[B][skip:].mean() * 1e9, "X", color=colors[B], ms=10, mec="white", mew=1.5)
        fit = np.mean([p["fit_rms_navg2"].mean() for p in perf.values()]) * 1e9
        ax.axhline(fit, color=ink2, lw=1, ls="--")
        ax.annotate(f"fitting error {fit:.0f} nm", (0.2, fit), xytext=(0, 4), textcoords="offset points",
                    fontsize=8, color=ink2)
        ax.set_title(title, fontsize=10, color=ink, loc="left")
        ax.set_xlabel("loop gain", color=ink2)
        ax.set_xlim(0.15, 1.12)
    axes[0].set_ylabel("residual [nm rms]", color=ink2)
    axes[0].plot([], [], "X", color=ink2, ms=9, label=f"PWFS, gain {GAIN_PWFS}")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=ink2)
    axes[0].set_ylim(0, None)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/residual_vs_gain_perfect_vs_pwfs.png", dpi=150)

    # ---- fig 2: residual time series, PWFS vs perfect (gain 0.5, same timing), one panel per B
    shown = [B for B in Bs if B in perf]
    fig, axes = plt.subplots(len(shown), 1, figsize=(10, 2.2 * len(shown)), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    w = 200
    smooth = lambda x: np.convolve(x, np.ones(w) / w, mode="valid")
    for ax, B in zip(axes, shown):
        style(ax)
        p = perf[B]
        i05 = int(np.argmin(np.abs(p["gains_navg2"] - GAIN_PWFS)))
        ax.plot(smooth(p["rms_navg2"][i05]) * 1e9, color=ink2, lw=1.2, label="perfect WFS")
        if B in pw:
            ax.plot(smooth(pw[B]) * 1e9, color=colors[B], lw=1.2, label="PWFS")
        ax.set_title(f"B = {B} Hz", fontsize=9, color=ink, loc="left")
        ax.set_ylabel("nm rms", color=ink2, fontsize=9)
        ax.legend(frameon=False, fontsize=8, labelcolor=ink2, loc="upper right", ncol=2)
    axes[-1].set_xlabel(f"frame ({w}-frame running mean)", color=ink2)
    axes[0].set_ylim(0, None)
    fig.suptitle(f"First stage residual, gain {GAIN_PWFS}", fontsize=10, color=ink, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/residual_time_series_perfect_vs_pwfs.png", dpi=150)
    print("figures in", fig_dir)
