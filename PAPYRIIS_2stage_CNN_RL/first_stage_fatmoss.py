"""
First stage (PAPYRUS) closed loops on the FATMOSS atmospheres.

Run from /home/evinerskas/LAM_OOPAO with PYTHONPATH=.:PAPYRIIS_2stage_CNN_RL

  perfect <B> <nLoop>   perfect-WFS loops (gain sweep) on the first nLoop frames, summary npz only
  pwfs    <B> [nLoop]   env.run_first_stage_loop (PWFS), full results npz as in PAPYRIIS_main.run_1st_stage
  project 0             project the B = 0 FATMOSS 2nd-stage atmosphere (15000 frames) to the 1st stage

Perfect WFS: the "measurement" is the residual OPD in the pupil, reconstructed onto the same 195 KL modes
the PWFS loop controls (least squares on the poked DM response). Timing is identical to run_first_stage_loop:
2-frame averaging of the measurement, then one averaged period of delay, integrator with leak.
So perfect vs PWFS isolates the wavefront-sensing error (noise, non-linearity, aliasing); the perfect loop
still has fitting + temporal error. "fit" = fitting error alone (atm OPD minus its projection, no delay).
"""

import os
import sys
import time
import zipfile

import numpy as np

from PAPYRIIS_2stage_CNN_RL.OOPAO_PAPYRIIS_env import OOPAO_environment_PAPYRIIS

proj_dir    = "PAPYRIIS_2stage_CNN_RL/projected_atm_1st_stage_150000_fatmoss"
proj_dir_B0 = "PAPYRIIS_2stage_CNN_RL/projected_atm_1st_stage_fatmoss"
fatmoss_B0  = "/home/evinerskas/FATMOSS/generated_atm_2nd_stage/atm_OPDs_2nd_r0_0.070_V0_3.002_L0_30.000_B_0.000_multi_layer.npy"
out_dir     = "PAPYRIIS_2stage_CNN_RL/2026-09-23/PAPYRIIS_arcturus_v2_15000/first_stage_gain_05_fatmoss"
perfect_dir = "PAPYRIIS_2stage_CNN_RL/2026-09-23/PAPYRIIS_arcturus_v2_15000/first_stage_perfect_wfs_fatmoss"
quick_dir   = "PAPYRIIS_2stage_CNN_RL/2026-09-23/PAPYRIIS_arcturus_v2_15000/first_stage_quick_fatmoss"

GAIN_PWFS = 0.5          # PAPYRIIS_main: best first-stage gain
LEAK      = 0.995
GAINS_PERFECT = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]


def tag(B):
    # the 150000-frame FATMOSS files are named "tboil_<B>" but hold the boiling factor B
    return f"r0_0.070_V0_3.002_L0_30.000_{'B' if B == 0 else 'tboil'}_{B:.3f}"


def proj_path(B):
    return f"{proj_dir_B0 if B == 0 else proj_dir}/atm_OPDs_1st_{tag(B)}_multi_layer.npz"


def load_projected(B, n=None):
    """First n frames of atm_OPDs_1st from an uncompressed npz (streamed: only n frames are read)."""
    with zipfile.ZipFile(proj_path(B)) as z, z.open("atm_OPDs_1st.npy") as f:
        version = np.lib.format.read_magic(f)
        read_header = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = read_header(f)
        n = shape[0] if n is None else min(n, shape[0])
        frame = int(np.prod(shape[1:]))
        a = np.frombuffer(f.read(n * frame * dtype.itemsize), dtype=dtype).reshape(n, *shape[1:])
    return a.copy()


def perfect_reconstructor(env, stroke=1e-8):
    """Least-squares reconstructor residual OPD (pupil pixels) -> DM coefs, on the controlled KL modes."""
    tel, ngs, dm = env.tel_1st, env.ngs_1st, env.dm_1st
    pupil = tel.pupil.astype(bool)
    M2C = env.M2C_1st_CL
    # DM response read from dm.OPD (what DeformableMirror.relay adds to the source OPD): repeated
    # "ngs*tel*dm" propagations accumulate the DM OPD in place, so they cannot be used for poking
    D = np.zeros((pupil.sum(), M2C.shape[1]))
    for k in range(M2C.shape[1]):
        dm.coefs = M2C[:, k] * stroke
        D[:, k] = np.asarray(dm.OPD)[pupil] / stroke
    dm.coefs = 0
    R = M2C @ np.linalg.pinv(D)
    return R, pupil


def run_perfect(env, atm, gains, n_avg=2, frame_delay=2, leak=LEAK):
    tel, ngs, dm, atm_obj, stt = env.tel_1st, env.ngs_1st, env.dm_1st, env.atm_1st, env.slow_tt
    env.Papytwin.set_pupil(calibration=True)
    atm_obj.initializeAtmosphere(telescope=tel)
    atm_obj.generateNewPhaseScreen(seed=env.seed)
    R, pupil = perfect_reconstructor(env)
    k = 2 * np.pi / ngs.wavelength
    nLoop = atm.shape[0]

    # fitting error alone: remove the best DM fit of the incoming OPD, no delay
    fit_rms = np.zeros(nLoop)
    for i in range(nLoop):
        dm.coefs = -R @ atm[i][pupil]
        atm_obj.update(atm[i])
        ngs ** atm_obj * tel * dm * stt
        r = ngs.OPD[pupil]
        fit_rms[i] = np.std(r)
    out = {"fit_rms": fit_rms, "gains": np.array(gains), "n_avg": n_avg, "frame_delay": frame_delay,
           "wavelength": ngs.wavelength}

    rms = np.zeros((len(gains), nLoop))
    for g, gain in enumerate(gains):
        dm.coefs = 0
        atm_obj.update(atm[0])
        ngs ** atm_obj * tel * dm * stt
        meas = ngs.OPD[pupil].copy()
        buf = [np.zeros(pupil.sum()) for _ in range(frame_delay - 1)]
        acc, cnt = 0, 0
        t0 = time.perf_counter()
        for i in range(nLoop):
            atm_obj.update(atm[i])
            acc = acc + meas
            cnt += 1
            if (i + 1) % n_avg == 0:
                buf.append(acc / cnt)
                dm.coefs = leak * dm.coefs - gain * (R @ buf.pop(0))
                acc, cnt = 0, 0
            ngs ** atm_obj * tel * dm * stt
            meas = ngs.OPD[pupil].copy()
            rms[g, i] = np.std(meas)
            if not np.isfinite(rms[g, i]) or rms[g, i] > 1e-5:   # diverged (> 10 um rms)
                rms[g, i:] = np.nan
                break
        sr = np.exp(-(k * rms[g]) ** 2)
        print(f"  perfect n_avg={n_avg} gain {gain:.2f}: residual {np.nanmean(rms[g, 100:]) * 1e9:6.1f} nm rms, "
              f"SR {np.nanmean(sr[100:]):.3f}  ({time.perf_counter() - t0:.0f} s)", flush=True)
    out["rms"] = rms
    return out


def perfect_main(B, nLoop):
    env = OOPAO_environment_PAPYRIIS()
    atm = load_projected(B, nLoop)
    print(f"B = {B} Hz: {atm.shape[0]} frames, input {np.mean([a[env.tel_1st.pupil.astype(bool)].std() for a in atm[::100]]) * 1e9:.1f} nm rms", flush=True)
    os.makedirs(perfect_dir, exist_ok=True)
    res = {}
    for n_avg in (2, 1):   # matched PWFS timing, then full-rate reference
        r = run_perfect(env, atm, GAINS_PERFECT, n_avg=n_avg)
        res.update({f"{k}_navg{n_avg}": v for k, v in r.items()})
    print(f"  fitting error alone: {np.mean(r['fit_rms']) * 1e9:.1f} nm rms", flush=True)
    np.savez(f"{perfect_dir}/results_1st_stage_perfect_{tag(B)}_nLoop_{atm.shape[0]}.npz", **res)


def pwfs_main(B, nLoop=None):
    env = OOPAO_environment_PAPYRIIS()
    t0 = time.perf_counter()
    atm = load_projected(B, nLoop)
    print(f"B = {B} Hz: loaded {atm.shape} in {time.perf_counter() - t0:.0f} s", flush=True)
    nLoop = atm.shape[0]
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.perf_counter()
    res = env.run_first_stage_loop(nLoop, atm, gainCL=GAIN_PWFS, leak=LEAK)
    print(f"loop done in {(time.perf_counter() - t0) / 3600:.2f} h", flush=True)
    cfg = res.pop("config")
    suffix = "" if nLoop in (15000, 150000) else f"_nLoop_{nLoop}"
    np.savez(f"{out_dir}/results_1st_stage_{tag(B)}_multi_layer{suffix}.npz", **res,
             nLoop=cfg.nLoop, gainCL=cfg.gainCL, leak=cfg.leak, frame_delay=cfg.frame_delay,
             photon_noise=cfg.photon_noise)
    print("saved", flush=True)


def quick_main(B, nLoop=3000):
    """Short convergence check: perfect WFS (gains 0.3/0.5/0.7, PWFS timing) + PWFS run_first_stage_loop."""
    env = OOPAO_environment_PAPYRIIS()
    atm = load_projected(B, nLoop)
    pupil = env.tel_1st.pupil.astype(bool)
    in_rms = atm[:, pupil].std(axis=1)
    perf = run_perfect(env, atm, [0.3, 0.5, 0.7])
    res = env.run_first_stage_loop(nLoop, atm, gainCL=GAIN_PWFS, leak=LEAK)
    pupil = env.tel_1st.pupil.astype(bool)
    pw_rms = res["src_opds"][:, pupil].std(axis=1)
    os.makedirs(quick_dir, exist_ok=True)
    np.savez(f"{quick_dir}/quick_{tag(B)}_nLoop_{nLoop}.npz", input_rms=in_rms, pwfs_rms=pw_rms,
             perfect_rms=perf["rms"], perfect_gains=perf["gains"], fit_rms=perf["fit_rms"], gain_pwfs=GAIN_PWFS)
    print(f"B = {B}: input {in_rms.mean() * 1e9:.0f} nm, fit {perf['fit_rms'].mean() * 1e9:.0f} nm, "
          f"PWFS last 1000 frames {pw_rms[-1000:].mean() * 1e9:.0f} nm", flush=True)


def project_B0():
    env = OOPAO_environment_PAPYRIIS()
    atm_2nd = np.load(fatmoss_B0)
    atm_1st = env.project_atmosphere_to_first_stage(atm_2nd)
    os.makedirs(proj_dir_B0, exist_ok=True)
    a = env.atm_2nd
    np.savez(proj_path(0), atm_OPDs_1st=atm_1st, r0=a.r0, L0=a.L0, windSpeed=a.windSpeed,
             fractionalR0=a.fractionalR0, windDirection=a.windDirection, altitude=a.altitude)


if __name__ == "__main__":
    mode = sys.argv[1]
    B = int(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else None
    {"perfect": lambda: perfect_main(B, n), "pwfs": lambda: pwfs_main(B, n), "project": project_B0,
     "quick": lambda: quick_main(B, n or 3000)}[mode]()
