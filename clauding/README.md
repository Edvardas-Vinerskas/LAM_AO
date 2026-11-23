# PO4AO variants

Four modifications of PO4AO (`po4ao_PAPYRUS/po4ao_PAPYRUS_upd.py`), each in its own folder, on a shared
re-implementation of the algorithm. The reasoning, the literature and the results are in the review
document; this file says what is where and how to run it.

| Folder | Idea | Entry point |
| --- | --- | --- |
| `po4ao_common/` | Shared code: agent, replay buffer, baseline networks, toy simulator, DAO bench adapter | |
| `po4ao_pix/` | Networks read the WFS image (76 px) instead of its reconstruction on the DM grid | `po4ao_pix.py` |
| `po4ao_pol/` | Networks also see the total DM command (pseudo-open-loop information) | `po4ao_pol.py` |
| `po4ao_pinn/` | Dynamics built on frozen flow of the pseudo-open-loop wavefront, plus a small CNN correction | `po4ao_pinn.py` |
| `po4ao_reward/` | Alternative planning losses (log, modal log, Strehl, pessimism, smoothness...) | `po4ao_reward.py` |

Each folder has the three files of the original layout: `po4ao_config_*.py`, `po4ao_models_*.py` and the run
script. Python 3.8+, PyTorch; nothing else.

## Running on the toy simulator

    python run_baseline.py --integrator        # integrator gain scan (reference)
    python run_baseline.py --n_history 64      # unchanged PO4AO
    python po4ao_pol/po4ao_pol.py
    python po4ao_pinn/po4ao_pinn.py
    python po4ao_pix/po4ao_pix.py
    python po4ao_reward/po4ao_reward.py --kind log
    python collect_results.py                  # table of all the runs found in */runs/
    python dynamics_benchmark.py [--pix]       # off-line test of the dynamics models (prediction, transfer, response to actions)
    python po4ao_pix/sampling_study.py         # numpy study: does a finer grid help to predict frozen flow?

Every run script lists its options at the top of the file. All runs with the same `--seed` see the same turbulence.

The simulator (`po4ao_common/toy_env.py`) is a fast stand-in, not OOPAO: frozen-flow von Karman screens, a 17 x 17
DM with 241 actuators, a gradient WFS sampled 76 pixels across with a saturation that gives an optical gain of
about 0.7, a two-frame delay, and the bench's own leaky integration, clamp and linear reconstruction. It has the
same `flatten()` / `step(action)` interface as the bench adapter and returns three things per frame: the
reconstructed residual (`obs.dm`), the WFS image (`obs.pix`) and the total command (`obs.cmd`).

## Running on PAPYRUS

    python po4ao_pol/po4ao_pol.py --bench

`po4ao_common/bench_dao.py` is a transcription of the `step()` / `flatten_dm()` of the original script (same shared
memories) and `runner.run_bench` reproduces its structure: control loop in the main process, training in a spawned
process on shared buffers. **None of this has been run on the bench.** The asynchronous loop was exercised on the
simulator on CPU only. Known differences with the original, all deliberate: the command sent to the DM is the
clamped one, leak and clamp come from the config, the replay buffer only returns windows that are contiguous in
time, the action is zero outside the valid actuators.

Pixel space needs four things that are not in this repository (frame reader, valid-pixel mask, pixel-to-command
reconstructor, interaction matrix on the same pixels); they are passed through `cfg['bench']`, see the top of
`bench_dao.py`. The toy simulator gives a 2-channel slopes map; for a pyramid, `PyramidFrames`
(`po4ao_pix/po4ao_models_pix.py`) turns a raw camera frame into the four pupil images divided by the mean flux and
minus the flat-wavefront reference (OOPAO's `slopesMaps` normalisation). It was tested on synthetic frames only.

## Using the simulator with your own environment

Anything with `flatten()`, `step(action)`, `sample_noise(sigma)` returning `Obs(dm, pix, cmd)` and exposing `mask`,
`KL_projection`, `n_act`, `device` works with `runner.run_sim`; `ramatwin_env.rama_env` and the OOPAO PAPYRIIS
environment are close to that already.
