"""
Movies of the FATMOSS 2nd stage atmospheres for several boiling factors B [Hz], same layout as atm_movie.py:
one movie per B plus a side-by-side grid, all with the same viridis colour scale (piston removed, in nm).

FATMOSS boiling: every Fourier component keeps its amplitude and its phase advances at a fixed random rate
up to ~ B * 2 dx |f|, so small scales boil fastest. The files of generated_atm_2nd_stage_150000 are named
"tboil_<B>" but hold the boiling factor B (their first frames match the "B_<B>" files of generated_atm_2nd_stage).
"""

import os
from multiprocessing import Pool

import numpy as np

from atm_movie import load_first_frames, make_movie, n_frames, vmax_pct

fatmoss_dir = "/home/evinerskas/FATMOSS"
atm_name    = "atm_OPDs_2nd_r0_0.070_V0_3.002_L0_30.000_{}_{:.3f}_multi_layer.npy"
atm_files = {
    0:    f"{fatmoss_dir}/generated_atm_2nd_stage/{atm_name.format('B', 0)}",
    200:  f"{fatmoss_dir}/generated_atm_2nd_stage_150000/{atm_name.format('tboil', 200)}",
    400:  f"{fatmoss_dir}/generated_atm_2nd_stage_150000/{atm_name.format('tboil', 400)}",
    1000: f"{fatmoss_dir}/generated_atm_2nd_stage_150000/{atm_name.format('tboil', 1000)}",
}
comparison_set = [1000, 400, 200, 0]   # decreasing boiling, frozen flow (B = 0) last

save_dir = f"{fatmoss_dir}/atm_movies"


def B_label(B):
    return f"B = {B:g} Hz"


if __name__ == "__main__":
    os.makedirs(save_dir, exist_ok=True)

    Bs = list(atm_files)
    with Pool(len(Bs)) as pool:
        opds = dict(zip(Bs, [o for o, _ in pool.starmap(load_first_frames, [(atm_files[B], n_frames) for B in Bs])]))
    for B, o in opds.items():
        print(f"B {B:g} Hz: spatial std {o.std(axis=(1, 2)).mean():.1f} nm")

    rng = np.random.default_rng(0)
    sample = np.concatenate([o[rng.choice(n_frames, 500, replace=False)].ravel() for o in opds.values()])
    vmax = float(np.percentile(np.abs(sample), vmax_pct))
    print(f"colour range: +-{vmax:.1f} nm")

    jobs = [([opds[B]], [B_label(B)], vmax, f"{save_dir}/atm_2nd_fatmoss_B_{B:.0f}_{n_frames}frames.mp4", 1)
            for B in Bs]
    jobs.append(([opds[B] for B in comparison_set], [B_label(B) for B in comparison_set], vmax,
                 f"{save_dir}/atm_2nd_fatmoss_B_comparison_{n_frames}frames.mp4", 2))
    with Pool(len(jobs)) as pool:
        pool.starmap(make_movie, jobs)
