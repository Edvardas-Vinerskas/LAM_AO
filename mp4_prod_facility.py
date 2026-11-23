import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.animation as animation
from matplotlib.colors import SymLogNorm
from matplotlib.patches import Circle
from matplotlib import cm
import matplotlib
from numpy.fft import fft2, fftshift
from functions import *
import time

"""import OOPAO
from OOPAO.Source import Source
from OOPAO.Telescope import Telescope
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.Atmosphere import Atmosphere
from OOPAO.Detector import Detector
from OOPAO.Pyramid import Pyramid
from OOPAO.ShackHartmann import ShackHartmann
from OOPAO.Zernike import Zernike
from OOPAO.calibration.InteractionMatrix import InteractionMatrix
from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis
from OOPAO.tools.displayTools import displayMap, makeSquareAxes
from OOPAO.calibration.CalibrationVault import CalibrationVault
from OOPAO.tools.displayTools import cl_plot, displayMap
"""
import torch

"""from po4ao_edw.OOPAO_environment_ZWFS import OOPAO_environment_ZWFS
from po4ao_edw.OOPAO_environment_PWFS import OOPAO_environment_PWFS
from po4ao_edw.OOPAO_environment_ZWFS_1_stage import OOPAO_environment_ZWFS_1_stage
from OOPAO.ZWFS2 import ZWFS2
from PIL import Image"""


import imageio
#from astropy.io import fits


def array_to_video(images, output_video, fps=30, cmap='viridis'):
    colormap = matplotlib.colormaps[cmap]

    num_frames, height, width = images.shape
    images_rgb = np.zeros((num_frames, height, width, 3), dtype=np.uint8)

    global_min = images.min()
    global_max = images.max()
    print(f"Global min: {global_min:.3f}, Global max: {global_max:.3f}")

    for i, frame in enumerate(images):
        if global_max - global_min == 0:
            frame_norm = np.ones_like(frame) * 0.5
        else:
            frame_norm = (frame - global_min) / (global_max - global_min)

        frame_colored = colormap(frame_norm)
        images_rgb[i] = (frame_colored[:, :, :3] * 255).astype(np.uint8)

    imageio.mimsave(output_video, images_rgb, fps=fps, bitrate='5000k', macro_block_size=2)


# Usage
# Load data

phase   = np.load('bench_sky_04_15/onsky_arcturus_1st200_2nd400_v10_20260416-020659/2026-04-16T02_17_55_telemetry_2nd_data_CL1OL2_phase.npy')

phase_mean_subtracted = phase - phase.mean(axis = 0)


array_to_video(phase_mean_subtracted, 'phase_arctan_mean_subtracted.mp4', fps=150)

