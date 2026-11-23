"""
Tip/tilt vs higher-order temporal behaviour of the generated 2nd stage atmospheres.

1. Explanatory movie of the frozen-flow atmosphere (tboil = 0): full OPD | tip/tilt | higher order
   (all on the same nm scale) + a scrolling tip/tilt trace in arcsec with a D/v scale bar.
2. Tip/tilt-only movie of all atmospheres side by side (decreasing boiling).
3. Temporal correlation of tip, tilt and higher order for frozen flow, from all 150000 frames.

Tip/tilt is the least-squares plane over the 90x90 square (no pupil mask in the files).
"""

import os
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import atm_movie
from atm_movie import (atm_files, comparison_sets, load_first_frames, n_frames, samplingTime, fps, base_dir,
                       colorize, colorbar, open_writer, make_movie, tboil_label)
from atm_correlation import temporal_correlation, D_tel, windSpeed, fractionalR0

pixel_size  = D_tel / 90
rad2arcsec  = 180 / np.pi * 3600
crossing    = D_tel / (fractionalR0 * windSpeed).sum()   # D / Cn2-weighted mean wind
trace_s     = 4                                          # length of the scrolling trace [s]
n_frames_long = 150000                                   # frames used for the correlation curves
max_lag_s   = 5

movie_dir = f"{base_dir}/atm_movies"
plot_dir  = f"{base_dir}/atm_comparison_plots"

COLORS = {"tip": "#2a78d6", "tilt": "#eb6834", "ho": "#1baf7a", "full": "#2b2b2b"}


def tip_tilt(opd):
    """Least-squares tip/tilt of piston-removed frames: coefficients [nm/px], plane, higher-order rest."""
    n = opd.shape[1]
    x = np.arange(n) - (n - 1) / 2
    X, Y = np.meshgrid(x, x)
    tip  = np.einsum("tij,ij->t", opd, X) / (X**2).sum()
    tilt = np.einsum("tij,ij->t", opd, Y) / (Y**2).sum()
    plane = (tip[:, None, None] * X + tilt[:, None, None] * Y).astype(np.float32)
    return tip, tilt, plane, opd - plane


def to_arcsec(coef_nm_per_px):
    return coef_nm_per_px * 1e-9 / pixel_size * rad2arcsec


def trace_panel(width, height, i, tip_as, tilt_as, ymax, font):
    """Scrolling tip/tilt trace over the last trace_s seconds, 'now' at the right edge."""
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    left, right, top, bottom = 70, width - 20, 34, height - 34
    n_win = int(round(trace_s / samplingTime))

    def px(k):
        return left + (k - (i - n_win + 1)) / (n_win - 1) * (right - left)

    def py(v):
        return top + (ymax - v) / (2 * ymax) * (bottom - top)

    for v in np.linspace(-ymax, ymax, 5):
        y = py(v)
        d.line([(left, y), (right, y)], fill=(225, 225, 225) if v != 0 else (170, 170, 170))
        d.text((left - 8, y - 8), f"{v:+.2f}" if v else "0", fill="black", font=font, anchor="ra")
    for s in range(trace_s + 1):
        x = right - s / trace_s * (right - left)
        d.line([(x, top), (x, bottom)], fill=(225, 225, 225))
        d.text((x, bottom + 6), "now" if s == 0 else f"-{s} s", fill="black", font=font, anchor="ma")
    d.text((8, top - 28), "arcsec", fill="black", font=font)

    k0 = max(0, i - n_win + 1)
    ks = np.arange(k0, i + 1)
    for series, key in [(tip_as, "tip"), (tilt_as, "tilt")]:
        if len(ks) > 1:
            d.line([(px(k), py(series[k])) for k in ks], fill=COLORS[key], width=2)

    # legend + D/v scale bar ending at 'now'
    for j, key in enumerate(["tip", "tilt"]):
        x0 = left + 10 + j * 90
        d.line([(x0, top - 18), (x0 + 22, top - 18)], fill=COLORS[key], width=3)
        d.text((x0 + 28, top - 27), key, fill="black", font=font)
    xb = right - crossing / trace_s * (right - left)
    d.line([(xb, top - 18), (right, top - 18)], fill="black", width=2)
    for x in (xb, right):
        d.line([(x, top - 24), (x, top - 12)], fill="black", width=2)
    d.text((xb - 8, top - 27), f"D / v = {crossing:.2f} s", fill="black", font=font, anchor="ra")
    return img


def explanatory_movie(opd, tip, tilt, plane, ho, vmax, path):
    font = ImageFont.load_default(size=16)
    side = opd.shape[1] * atm_movie.upscale
    cbar = colorbar(side, vmax, font)
    head, trace_h = 70, 280
    w = 3 * side + cbar.width
    h = head + side + trace_h
    w, h = w + w % 2, h + h % 2
    tip_as, tilt_as = to_arcsec(tip), to_arcsec(tilt)
    ymax = float(np.ceil(np.percentile(np.abs(np.r_[tip_as, tilt_as]), 99.9) * 10) / 10)
    titles = ["full OPD (piston removed)", "tip/tilt only", "higher order (OPD - tip/tilt)"]
    proc = open_writer(path, w, h)
    for i in range(n_frames):
        canvas = Image.new("RGB", (w, h), "white")
        d = ImageDraw.Draw(canvas)
        d.text((4, 4), f"tboil = 0 (frozen flow)   frame {i:5d}   t = {i * samplingTime:6.3f} s",
               fill="black", font=font)
        for k, (cube, title) in enumerate(zip([opd, plane, ho], titles)):
            d.text((k * side + 4, head - 22), title, fill="black", font=font)
            canvas.paste(Image.fromarray(colorize(cube[i], vmax)), (k * side, head))
        canvas.paste(cbar, (3 * side, head))
        canvas.paste(trace_panel(w, trace_h, i, tip_as, tilt_as, ymax, font), (0, head + side))
        proc.stdin.write(canvas.tobytes())
    proc.stdin.close()
    proc.wait()
    print("saved", path)


def correlation_plot(path):
    opd, _ = load_first_frames(atm_files[0], n_frames_long)
    tip, tilt, _, ho = tip_tilt(opd)
    max_lag = int(round(max_lag_s / samplingTime))
    curves = {
        "tip":  temporal_correlation(tip[:, None], max_lag),
        "tilt": temporal_correlation(tilt[:, None], max_lag),
        "ho":   temporal_correlation(ho, max_lag),
        "full": temporal_correlation(opd, max_lag),
    }
    del opd, ho
    lags = np.arange(max_lag + 1) * samplingTime
    labels = {"full": "full OPD (piston removed)", "tip": "tip", "tilt": "tilt",
              "ho": "higher order (piston + tip/tilt removed)"}
    np.savez(path.replace(".png", ".npz"), lags=lags, **curves)

    print(f"frozen flow, {n_frames_long} frames:")
    for key, C in curves.items():
        below = np.nonzero(C < 1 / np.e)[0]
        print(f"  {labels[key]:42s} 1/e time {lags[below[0]] if below.size else np.nan:6.3f} s | "
              f"C(D/v = {crossing:.2f} s) = {C[int(round(crossing / samplingTime))]:.2f} | C(2 s) = {C[int(round(2 / samplingTime))]:.2f}")

    tip_as, tilt_as = to_arcsec(tip), to_arcsec(tilt)
    t = np.arange(len(tip)) * samplingTime
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1, 1.25]})
    for key in ["full", "tip", "tilt", "ho"]:
        ax1.plot(lags, curves[key], color=COLORS[key], lw=2, label=labels[key])
    ax1.axvline(crossing, color="0.55", lw=1, ls=":")
    ax1.text(crossing + 0.05, 0.95, f"D / v = {crossing:.2f} s", color="0.35", fontsize=9)
    ax1.axhline(0, color="0.75", lw=0.8)
    ax1.set_xlim(0, max_lag_s)
    ax1.set_ylim(-0.25, 1.02)
    ax1.set_xlabel(r"lag $\tau$ (s)")
    ax1.set_ylabel(r"$C(\tau)$")
    ax1.set_title(f"temporal correlation, frozen flow ({n_frames_long} frames)")
    ax1.legend(frameon=False, fontsize=9)

    show = t < 15
    ax2.plot(t[show], tip_as[show], color=COLORS["tip"], lw=1.2, label="tip")
    ax2.plot(t[show], tilt_as[show], color=COLORS["tilt"], lw=1.2, label="tilt")
    y_bar = 0.9 * ax2.get_ylim()[1]
    ax2.plot([0.5, 0.5 + crossing], [y_bar, y_bar], color="black", lw=2)
    ax2.text(0.5 + crossing + 0.15, y_bar, f"D / v = {crossing:.2f} s", va="center", fontsize=9)
    ax2.axhline(0, color="0.75", lw=0.8)
    ax2.set_xlim(0, 15)
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("arcsec")
    ax2.set_title("tip/tilt time series, first 15 s")
    ax2.legend(frameon=False, fontsize=9, loc="lower right")
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.25)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("saved", path)


if __name__ == "__main__":
    os.makedirs(movie_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    tboils = comparison_sets["tboil"]   # decreasing boiling, no boiling (tboil = 0) last
    with Pool(len(tboils)) as pool:
        opds = dict(zip(tboils, [o for o, _ in pool.starmap(
            load_first_frames, [(atm_files[t], n_frames) for t in tboils])]))
    decomposed = {t: tip_tilt(opds[t]) for t in tboils}

    tip, tilt, plane, ho = decomposed[0]
    rng = np.random.default_rng(0)
    vmax = float(np.percentile(np.abs(opds[0][rng.choice(n_frames, 500, replace=False)]), 99.9))
    vmax_tt = float(np.percentile(np.abs(np.concatenate(
        [decomposed[t][2][rng.choice(n_frames, 500, replace=False)].ravel() for t in tboils])), 99.9))
    print(f"colour range: full +-{vmax:.0f} nm, tip/tilt grid +-{vmax_tt:.0f} nm")

    with Pool(2) as pool:
        jobs = [
            pool.apply_async(explanatory_movie, (opds[0], tip, tilt, plane, ho, vmax,
                                                 f"{movie_dir}/atm_2nd_tboil_0.000_tiptilt_explained_{n_frames}frames.mp4")),
            pool.apply_async(make_movie, ([decomposed[t][2] for t in tboils], [tboil_label(t) for t in tboils], vmax_tt,
                                          f"{movie_dir}/atm_2nd_tiptilt_only_comparison_{n_frames}frames.mp4", 3)),
        ]
        del opds, decomposed
        correlation_plot(f"{plot_dir}/atm_2nd_tboil_0.000_tiptilt_correlation.png")
        for job in jobs:
            job.get()
