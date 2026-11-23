"""Convergence plot for the 3000-frame quick runs of first_stage_fatmoss.py (run from LAM_OOPAO)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

d = "PAPYRIIS_2stage_CNN_RL/2026-09-23/PAPYRIIS_arcturus_v2_15000/first_stage_quick_fatmoss"
tag = lambda B: f"r0_0.070_V0_3.002_L0_30.000_{'B' if B == 0 else 'tboil'}_{B:.3f}"
ink, ink2, grid, blue, orange = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834"
Bs = [0, 200, 400, 1000]
fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True, sharey=True)
for ax, B in zip(axes.ravel(), Bs):
    r = np.load(f"{d}/quick_{tag(B)}_nLoop_3000.npz")
    i05 = int(np.argmin(np.abs(r["perfect_gains"] - r["gain_pwfs"])))
    ax.plot(r["input_rms"] * 1e9, color=ink2, lw=1, alpha=0.6, label="open loop (input)")
    ax.plot(r["perfect_rms"][i05] * 1e9, color=blue, lw=1, label=f"perfect WFS, g={r['gain_pwfs']}")
    ax.plot(r["pwfs_rms"] * 1e9, color=orange, lw=1, label=f"PWFS, g={r['gain_pwfs']}")
    ax.axhline(r["fit_rms"].mean() * 1e9, color=ink2, lw=1, ls="--", label="fitting error")
    ax.set_title(f"B = {B} Hz   (PWFS {r['pwfs_rms'][-1000:].mean()*1e9:.0f} nm, perfect {r['perfect_rms'][i05][-1000:].mean()*1e9:.0f} nm)",
                 fontsize=9, color=ink, loc="left")
    ax.grid(True, color=grid, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.tick_params(colors=ink2, labelsize=8)
for ax in axes[1]: ax.set_xlabel("frame", color=ink2)
for ax in axes[:, 0]: ax.set_ylabel("residual [nm rms]", color=ink2)
axes[0, 0].set_ylim(0, None)
axes[0, 0].legend(frameon=False, fontsize=8, labelcolor=ink2, loc="upper right")
fig.suptitle("First stage, 3000 frames, FATMOSS atmospheres (r0 = 7 cm, V0 = 3 m/s)", fontsize=10, color=ink, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(f"{d}/first_stage_convergence_quick.png", dpi=150)
print(f"{d}/first_stage_convergence_quick.png")
