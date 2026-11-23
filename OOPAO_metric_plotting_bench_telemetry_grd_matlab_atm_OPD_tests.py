"""
Atmosphere tests: compares on-sky data with simulated atmospheric OPDs (KL mode variance + temporal PSDs only)

#   on sky atm reconstruction from 1st stage DM shape (dmCmdCube)
#   on sky 1st stage residual reconstruction from 2nd stage CNN reconstruction in DM space (CL1OL2), optional
#   simulated atm: 2nd stage OPD screens (90x90, original size, no resizing) projected onto the 1st stage KL basis,
#                  with the 1st stage DM built at the OPD resolution (mode_calculator_fromOPD)
#                  (pure atmosphere, no DM / loop filtering)

Plotting style follows OOPAO_metric_plotting_bench_telemetry_grd_matlab_1st_stage_tests.py:
    on sky     -> atm solid black, 1st stage dashed ('--')
    simulation -> dotted (':'), group 1 (OOPAO) shades of blue, group 2 (FATMOSS) shades of red, dark -> light

Figures are saved at save_dpi into save_dir (nothing is shown)
Run from the LAM_OOPAO folder
"""

import os
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend: figures are saved to disk instead of shown
import matplotlib.pyplot as plt
from functions import *

save_dir = 'PAPYRIIS_2stage_CNN_RL/atm_comparison_plots/atm_OPD_tests_low_boil'   # folder where all figures are saved
save_dpi = 200
os.makedirs(save_dir, exist_ok=True)

plot_onsky    = True   # plot on-sky results (solid/dashed lines)
plot_sim      = True   # plot simulated atmospheric OPDs (dotted lines)
plot_residual = True   # on-sky 1st stage residual from CL1OL2 (no sim equivalent here)

frequency_1st = 400    # sampling frequency of the on-sky 1st stage residual (CL1OL2, measured by the 2nd stage)
frequency_atm = 200    # sampling frequency of the on-sky 1st stage DM commands
frequency_opd = 400    # sampling frequency of the simulated atmospheric OPDs (one OPD per 1st stage loop iteration)
KL_frames     = [-10000, -1]              # [start, end] indices for the timeseries window
psd_modes     = [0, 10, 20, 30, 40]  # KL mode indices to plot PSDs for
label_1st     = "2nd stage RL"   # legend text for the on-sky 1st stage residual curves




# on sky data
loaddir = 'bench_sky_04_15/arcturus_v7'
RL_telemetry_1st = np.load(f'{loaddir}/2026-04-16T01_19_12_telemetry_data_RLiter50.npy', allow_pickle = True)
CL1st_OL_2nd     = np.load(f'{loaddir}/2026-04-16T01_27_23_telemetry_2nd_data_CL1OL2_noCRED2.npy', allow_pickle = True)
#2026-04-16T01_27_23_telemetry_2nd_data_CL1OL2_noCRED2
#2026-04-16T01_23_42_telemetry_2nd_data_pythint_noCRED2
#2026-04-16T01_19_37_telemetry_2nd_data_RLiter50_noCRED2



# atmospheric OPDs to compare: label, folder, file (comment out the ones you don't want)
# the OPDs are used at their original size (90x90), the 1st stage KL basis is built at that resolution
# group 1 -> shades of blue, group 2 -> shades of red (dark -> light in list order)
atmdir_oopao   = "PAPYRIIS_2stage_CNN_RL/generated_atm_2nd_stage_150000"   # OOPAO, 150000 frames, compressed npz (only KL_frames are kept)
atmdir_fatmoss = "/home/evinerskas/FATMOSS/generated_atm_2nd_stage"        # FATMOSS, 15000 frames, npy saved as (90, 90, time)
group_1 = [
    # {"label": "OOPAO, tboil = 0.0", "folder": atmdir_oopao, "file": "atm_OPDs_2nd_r0_0.070_V0_3.002_L0_30.000_tboil_0.000_multi_layer.npz"},
    {"label": "OOPAO, tboil = 5.0", "folder": atmdir_oopao, "file": "atm_OPDs_2nd_r0_0.070_V0_3.002_L0_30.000_tboil_5.000_multi_layer.npz"},
    {"label": "OOPAO, tboil = 2.0", "folder": atmdir_oopao, "file": "atm_OPDs_2nd_r0_0.070_V0_3.002_L0_30.000_tboil_2.000_multi_layer.npz"},
    {"label": "OOPAO, tboil = 1.0", "folder": atmdir_oopao, "file": "atm_OPDs_2nd_r0_0.070_V0_3.002_L0_30.000_tboil_1.000_multi_layer.npz"},
    # {"label": "OOPAO, tboil = 0.5", "folder": atmdir_oopao, "file": "atm_OPDs_2nd_r0_0.070_V0_3.002_L0_30.000_tboil_0.500_multi_layer.npz"},
    # {"label": "OOPAO, tboil = 0.1", "folder": atmdir_oopao, "file": "atm_OPDs_2nd_r0_0.070_V0_3.002_L0_30.000_tboil_0.100_multi_layer.npz"},
]
group_2 = [
    # {"label": "FATMOSS, B = 0",   "folder": atmdir_fatmoss, "file": "atm_OPDs_2nd_r0_0.070_V0_3.002_L0_30.000_B_0.000_multi_layer.npy"},
    {"label": "FATMOSS, B = 100", "folder": atmdir_fatmoss, "file": "atm_OPDs_2nd_r0_0.070_V0_3.002_L0_30.000_B_100.000_multi_layer.npy"},
    {"label": "FATMOSS, B = 200", "folder": atmdir_fatmoss, "file": "atm_OPDs_2nd_r0_0.070_V0_3.002_L0_30.000_B_200.000_multi_layer.npy"},
    {"label": "FATMOSS, B = 400", "folder": atmdir_fatmoss, "file": "atm_OPDs_2nd_r0_0.070_V0_3.002_L0_30.000_B_400.000_multi_layer.npy"},
    {"label": "FATMOSS, B = 1000", "folder": atmdir_fatmoss, "file": "atm_OPDs_2nd_r0_0.070_V0_3.002_L0_30.000_B_1000.000_multi_layer.npy"},
]

# colours as in the original file (on sky), sims get a dark -> light gradient of their group colour
color_atm           = "black"
color_1st_variance  = "cornflowerblue"
color_1st_psd       = "indianred"
def sim_shades(cmap_name, n):
    return [plt.get_cmap(cmap_name)(x) for x in np.linspace(0.9, 0.45, n)]
for group, cmap_name in [(group_1, "Blues"), (group_2, "Reds")]:
    for sim, color in zip(group, sim_shades(cmap_name, len(group))):
        sim["color"] = color
simulations = group_1 + group_2


#TODO these don't have the representative masks (masking done as per PAPYRIIS_main.py or just look inside the 2 below functions)
dm_1st_inf, pupil_mask_1 = first_stage_dm_builder()
dm_2nd_inf_sky, pupil_mask_2 = second_stage_dm_builder(sign = 1.6e-5)   # on sky data: real DM amplitude

dm_1st_inf = dm_1st_inf.reshape(100, 100, 241)
dm_1st_inf_masked = dm_1st_inf[pupil_mask_1, :]

dm_2nd_inf_sky_masked = dm_2nd_inf_sky.reshape(100, 100, 97)[pupil_mask_2, :]

M2C_1st = - np.load('PAPYRIIS_2stage_CNN_RL/M2C_1rst.npy')
M2C_2nd = np.load('PAPYRIIS_2stage_CNN_RL/M2C_KL.npy')

C2M_2nd = np.linalg.pinv(M2C_2nd)
M2P_1st = dm_1st_inf_masked @ M2C_1st
projector_kl_2nd_to_1st_sky = np.linalg.pinv(M2P_1st) @ (dm_2nd_inf_sky_masked @ M2C_2nd)

# M2P_1st scaled to nm before computing covariance
GG = mode_covariance(M2P_1st * 1e9)   # full covariance: KL_variance_calculator (from functions.py) uses the cross terms for the total
GG_diag = np.diag(GG)                 # diagonal only: tPSD_calculator

def read_npy_header(fh):
    version = np.lib.format.read_magic(fh)
    read_header = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
    return read_header(fh)


def load_opd_frames(path, frame_bounds, key='atm_OPDs_2nd', chunk_frames=2000):
    """
    Returns the OPD frames inside frame_bounds as (time, N, N) in meters, without loading the full file:
        .npy                          -> memmap (FATMOSS saves (N, N, time), transposed here to (time, N, N))
        .npz uncompressed (savez)     -> memmap of the zip member
        .npz compressed (savez_compr) -> streamed, frames before the window are decompressed and discarded
    """
    if path.endswith('.npy'):
        opds = np.load(path, mmap_mode='r')
        if opds.shape[0] == opds.shape[1] != opds.shape[2]:   # square frames first -> time is the last axis
            opds = opds.transpose(2, 0, 1)
        return np.asarray(opds[frame_bounds[0]:frame_bounds[1]], dtype=np.float64)

    with zipfile.ZipFile(path) as z:
        info = z.getinfo(f"{key}.npy")
        if info.compress_type != zipfile.ZIP_STORED:
            with z.open(info) as fh:
                shape, fortran_order, dtype = read_npy_header(fh)
                start, stop, _ = slice(*frame_bounds).indices(shape[0])
                frame_bytes = int(np.prod(shape[1:])) * dtype.itemsize
                for skipped in range(0, start, chunk_frames):
                    fh.read(min(chunk_frames, start - skipped) * frame_bytes)
                opds = np.frombuffer(fh.read((stop - start) * frame_bytes), dtype=dtype)
            return opds.reshape((stop - start, *shape[1:])).astype(np.float64)

    with open(path, 'rb') as fh:
        fh.seek(info.header_offset + 26)                                  # local file header: name/extra lengths at byte 26
        name_len, extra_len = np.frombuffer(fh.read(4), dtype='<u2')
        fh.seek(info.header_offset + 30 + name_len + extra_len)
        shape, fortran_order, dtype = read_npy_header(fh)
        offset = fh.tell()
    opds = np.memmap(path, dtype=dtype, mode='r', offset=offset, shape=shape, order='F' if fortran_order else 'C')
    return np.asarray(opds[frame_bounds[0]:frame_bounds[1]], dtype=np.float64)


# 1st stage DM influence functions + pupil at the OPD resolution (no resizing of the OPDs), built once per resolution
dm_1st_at_resolution = {}
def dm_1st_masked(resolution):
    if resolution not in dm_1st_at_resolution:
        dm_inf, pupil = first_stage_dm_builder(resolution = resolution)
        pupil = pupil.astype(bool)
        dm_1st_at_resolution[resolution] = (dm_inf.reshape(resolution, resolution, -1)[pupil, :], pupil)
    return dm_1st_at_resolution[resolution]


# ---------------------------------------------------Load data---------------------------------------------------#
if plot_onsky:
    if plot_residual:
        next_states_1st = CL1st_OL_2nd.item()['slavedreconsCube'].squeeze() #1st stage residual as measured by 2nd stage
        modes_1st_stage = next_states_1st @ C2M_2nd.T @ projector_kl_2nd_to_1st_sky.T
        coefs_var_1st_stage, modes_1st_stage_masked, coefs_var_1st_stage_total = KL_variance_calculator(modes_1st_stage, GG, KL_frames)

    modes_atm = mode_calculator_fromDM(RL_telemetry_1st.item()['dmCmdCube'].squeeze(), M2C_1st)
    coefs_var_atm, modes_atm_masked, coefs_var_atm_total = KL_variance_calculator(modes_atm, GG, KL_frames)

if plot_sim:
    for sim in simulations:
        path = f"{sim['folder']}/{sim['file']}"
        print(f"loading {path}")
        opds = load_opd_frames(path, KL_frames)   # (time, N, N), original size
        dm_inf_masked, pupil = dm_1st_masked(opds.shape[-1])
        modes = mode_calculator_fromOPD(opds[:, pupil], M2C_1st, dm_inf_masked)   # piston removed inside the pupil
        del opds
        sim["var_atm"], sim["modes_atm_masked"], sim["var_atm_total"] = KL_variance_calculator(modes, GG, [0, len(modes)])


# ---------------------------------------------------KL mode variance---------------------------------------------------#
plt.figure(figsize=(19, 8))  # wider than the original (12, 8) to fit the legend on the right
if plot_onsky:
    plt.plot(coefs_var_atm, color=color_atm, label=f"Atm, σ² = {coefs_var_atm_total:,.0f} nm²")
if plot_sim:
    for sim in simulations:
        plt.plot(sim["var_atm"], ':', color=sim["color"], lw=2.5, label=f"Atm (sim, {sim['label']}), σ² = {sim['var_atm_total']:,.0f} nm²")
if plot_onsky and plot_residual:
    plt.plot(coefs_var_1st_stage, '--', color=color_1st_variance, lw=2.5, label=f"{label_1st}, σ² = {coefs_var_1st_stage_total:,.0f} nm²")

plt.title(f"KL mode variance", fontsize=30)
plt.yscale("log")
plt.xscale("log")
plt.xlabel("KL mode index", fontsize=24)
plt.ylabel("Temporal residual variance (nm²)", fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True, which='both', alpha=0.5)
plt.minorticks_on()
plt.legend(fontsize=16, loc='upper left', bbox_to_anchor=(1.01, 1))  # outside the axes so it does not cover the curves
plt.tight_layout()
plt.savefig(f"{save_dir}/KL_mode_variance.png", dpi=save_dpi, bbox_inches='tight')
plt.close()


# ---------------------------------------------------Temporal PSD---------------------------------------------------#
for mode in psd_modes:
    plt.figure(figsize=(19, 8))  # wider than the original (12, 8) to fit the legend on the right
    if plot_onsky and plot_residual:
        f_1st, psd_1st, _ = tPSD_calculator(modes_1st_stage_masked, mode, GG_diag, frequency_1st)
        plt.plot(f_1st, psd_1st, '--', color=color_1st_psd, lw=2.5, label=label_1st)
    if plot_onsky:
        f_atm, psd_atm, _ = tPSD_calculator(modes_atm_masked, mode, GG_diag, frequency_atm)
        plt.plot(f_atm, psd_atm, lw=2.5, color=color_atm, label="atm")
    if plot_sim:
        for sim in simulations:
            f_atm, psd_atm, _ = tPSD_calculator(sim["modes_atm_masked"], mode, GG_diag, frequency_opd)
            plt.plot(f_atm, psd_atm, ':', color=sim["color"], lw=2.5, label=f"atm (sim, {sim['label']})")

    plt.title("Tip PSD" if mode == 0 else f"PSD KL mode {mode}", fontsize=30)
    plt.xlabel("frequency (Hz)", fontsize=24)
    plt.ylabel("PSD (nm² Hz⁻¹)", fontsize=24)
    plt.yscale("log")
    plt.xscale("log")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, which='both', alpha=0.5)
    plt.minorticks_on()
    plt.legend(fontsize=16, loc='upper left', bbox_to_anchor=(1.01, 1))  # outside the axes so it does not cover the curves
    plt.tight_layout()
    plt.savefig(f"{save_dir}/tPSD_mode_{mode}.png", dpi=save_dpi, bbox_inches='tight')
    plt.close()

if plot_sim:
    for sim in simulations:
        print(f"total atm variance ({sim['label']}): {sim['var_atm_total']:,.0f} nm²")
print(f"figures saved in {save_dir}")
