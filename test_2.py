import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import sys
repo = '/home/evinerskas/LAM_OOPAO'
if repo not in sys.path:
    sys.path.insert(0, repo)
from OOPAO.Source import Source
from OOPAO.Telescope import Telescope
from OOPAO.Atmosphere import Atmosphere
from OOPAO.ZWFS2_v2 import ZWFS2
from OOPAO.ZWFS_v2 import ZWFS
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis
from OOPAO.calibration.InteractionMatrix import InteractionMatrix
