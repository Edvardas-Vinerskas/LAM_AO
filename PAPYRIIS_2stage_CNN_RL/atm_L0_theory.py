"""
Analytic tip/tilt and higher-order statistics of the 2nd stage atmosphere vs outer scale L0.

No phase screens: everything follows from the von Karman phase structure function
    D(r) = 4 pi int W(f) (1 - J0(2 pi f r)) f df,   W(f) = 0.023 r0^-5/3 (f^2 + 1/L0^2)^-11/6
which is finite for any L0 (also L0 -> infinity, pure Kolmogorov). Piston, tip/tilt and higher order
are linear projections with zero-sum weights, so only D(r) is needed: cov = -1/2 w^T D w.

Frozen flow: layer l is shifted by v_l tau, so C_l(tau) uses D(|r_i - r_j - v_l tau|), weighted by fractionalR0.
Same 90 px square of D = 1.6 m as the generated atmospheres (subsampled 2x for speed).
"""

import os

import numpy as np
from scipy.special import j0
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from atm_correlation import D_tel, windSpeed, fractionalR0

r0            = 0.07
windDirection = np.deg2rad([0, 72, 144, 216, 288])
L0_list       = [30, 100, 1e3, 1e6]          # [m]
rad2nm        = 500 / (2 * np.pi)           # phase at 500 nm -> OPD in nm
subsample     = 2
lags          = np.linspace(0, 10, 101)     # [s]

save_dir      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "atm_comparison_plots")
save_filename = "atm_2nd_L0_theory_tiptilt.png"

COLORS = dict(zip(L0_list, ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]))


def structure_function(L0, r):
    f = np.logspace(-10, 3, 6000)
    W = 0.023 * r0**(-5/3) * (f**2 + 1 / L0**2)**(-11/6)
    integrand = 4 * np.pi * W * f * (1 - j0(2 * np.pi * np.outer(r, f)))
    return np.trapezoid(integrand * f, np.log(f), axis=1)   # int g df = int g f dln(f)


def grid():
    n = 90 // subsample
    x = (np.arange(n) - (n - 1) / 2) * D_tel / n
    X, Y = np.meshgrid(x, x)
    return X.ravel(), Y.ravel()


def statistics(L0):
    X, Y = grid()
    N = X.size
    r_tab = np.r_[0, np.logspace(-4, np.log10(np.hypot(*[2 * D_tel] * 2) + windSpeed.max() * lags[-1]), 800)]
    D_tab = structure_function(L0, r_tab)

    # orthonormal piston / tip / tilt basis on the grid (orthogonal on a symmetric square)
    u_p = np.ones(N) / np.sqrt(N)
    u_x = X / np.linalg.norm(X)
    u_y = Y / np.linalg.norm(Y)
    w_tip, w_tilt = X / (X**2).sum(), Y / (Y**2).sum()     # same definition as tip_tilt() in atm_tiptilt.py

    dX = X[:, None] - X[None, :]
    dY = Y[:, None] - Y[None, :]
    out = {k: np.zeros(len(lags)) for k in ["tip", "tilt", "full", "ho"]}
    for frac, v, th in zip(fractionalR0, windSpeed, windDirection):
        for k, tau in enumerate(lags):
            C = -0.5 * frac * np.interp(np.hypot(dX - v * np.cos(th) * tau, dY - v * np.sin(th) * tau), r_tab, D_tab)
            out["tip"][k]  += w_tip @ C @ w_tip
            out["tilt"][k] += w_tilt @ C @ w_tilt
            # summed covariance of the projected field: trace(P C) with P = I - sum u u^T
            full = np.trace(C) - u_p @ C @ u_p
            out["full"][k] += full / N
            out["ho"][k]   += (full - u_x @ C @ u_x - u_y @ C @ u_y) / N
    pixel = D_tel / (90 // subsample)
    rms = {
        "TT plane rms [nm]": np.sqrt((out["tip"][0] * (X**2).mean() + out["tilt"][0] * (Y**2).mean())) * rad2nm,
        "tip rms [arcsec]":  np.sqrt(out["tip"][0]) * 500e-9 / (2 * np.pi) * 206265,
        "HO rms [nm]":       np.sqrt(out["ho"][0]) * rad2nm,
        "full rms [nm]":     np.sqrt(out["full"][0]) * rad2nm,
    }
    corr = {k: c / c[0] for k, c in out.items()}
    return rms, corr


def one_over_e(C):
    below = np.nonzero(C < 1 / np.e)[0]
    return lags[below[0]] if below.size else np.nan


if __name__ == "__main__":
    os.makedirs(save_dir, exist_ok=True)
    results = {L0: statistics(L0) for L0 in L0_list}

    print(f"{'L0':>9} {'TT rms nm':>10} {'tip arcsec':>11} {'HO rms nm':>10} | 1/e time [s]: {'tip':>5} {'tilt':>5} {'HO':>5} {'full':>5} | C_tip(10 s)")
    for L0, (rms, corr) in results.items():
        print(f"{L0:9.0f} {rms['TT plane rms [nm]']:10.0f} {rms['tip rms [arcsec]']:11.3f} {rms['HO rms [nm]']:10.0f} |"
              f"               {one_over_e(corr['tip']):5.2f} {one_over_e(corr['tilt']):5.2f} {one_over_e(corr['ho']):5.2f} "
              f"{one_over_e(corr['full']):5.2f} | {corr['tip'][-1]:.2f}")

    measured = np.load(os.path.join(save_dir, "atm_2nd_tboil_0.000_tiptilt_correlation.npz"))
    fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, key, title in zip(axs, ["tip", "ho"], ["tip", "higher order (piston + tip/tilt removed)"]):
        for L0, (_, corr) in results.items():
            label = f"L0 = {L0:g} m" if L0 < 1e3 else f"L0 = {L0 / 1e3:g} km"
            ax.plot(lags, corr[key], color=COLORS[L0], lw=2, label=label)
        ax.plot(measured["lags"], measured[key], color="#eb6834", lw=1.2, ls="--",
                label="measured, tboil = 0 file (L0 = 30 m)")
        ax.axhline(0, color="0.75", lw=0.8)
        ax.set_xlim(0, lags[-1])
        ax.set_title(f"frozen flow, theory: {title}")
        ax.set_xlabel(r"lag $\tau$ (s)")
        ax.grid(True, alpha=0.25)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    axs[0].set_ylabel(r"$C(\tau)$")
    axs[0].legend(frameon=False, fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, save_filename), dpi=200)
    plt.close(fig)
    print("saved", os.path.join(save_dir, save_filename))
