"""
ZWFS v2 tutorial: OOPAO/ZWFS_v2.py (one ZWFS) and OOPAO/ZWFS2_v2.py (two ZWFS, 'atan' reconstruction).


    1. create the ZWFS: its camera is noise-free by default (as for the Pyramid)
    2. calibrate (InteractionMatrix) with that noise-free camera -> exact interaction matrix
       (InteractionMatrix(noise='off') switches off photon/readout noise but not the dark current)
    3. set the real camera with set_camera(...) just before the AO loop (it recalibrates the reference by itself)
    4. run the loop: the mean dark level is subtracted from every frame automatically (wfs.subtract_dark = True)

Pipeline of one measurement (src**atm*tel*dm*zwfs):
    wfs.raw_data               noise-free intensity [photons/px/frame], telescope grid
    wfs.cam.frame              raw camera output [camera units: x QE x gain (+ ADC)] + noise, camera grid (nSubap + 2*n_pix_edge)
    wfs.frame_dark_subtracted  cam.frame in float - wfs.dark_level [camera units] (the noise stays)
    wfs.img_ZWFS               frame_dark_subtracted in [photons/camera px/frame] (input of the asin/atan reconstruction)
    wfs.signal                 (img_ZWFS/norma - img_ref/norma_ref) on the valid pixels
"""
#
import matplotlib.pyplot as plt
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


if sys.platform == 'win32' and 'ipykernel' not in sys.modules:
    plt.switch_backend('TkAgg')

# ------------------------------------------------------------- telescope, atmosphere, DM
src = Source(optBand='H', magnitude=0)

tel = Telescope(resolution=90,
                diameter=8,
                samplingTime=1/400,
                centralObstruction=0)
src * tel


atm = Atmosphere(telescope=tel,
                 r0=1.2,
                 L0=30,
                 windSpeed=[0.88, 0.99, 2.79, 5.37, 4.25],
                 fractionalR0=[0.45, 0.1, 0.1, 0.25, 0.1],
                 windDirection=[0, 72, 144, 216, 288],
                 altitude=[0, 1000, 5000, 10000, 12000])
atm.initializeAtmosphere(telescope=tel)   

dm = DeformableMirror(telescope=tel,
                      nSubap=16,
                      mechCoupling=0.35)

M2C_KL = compute_KL_basis(tel=tel, atm=atm, dm=dm)

# ------------------------------------------------------------- 1. the ZWFS (noise-free camera)
zwfs = ZWFS(tel=tel,
             diameter=1.96,                                  
             phase_shift=0.5*np.pi,          
             zpf=50,                                         
             n_pix_edge=3,                                   
             seed=0)                                         # reproducible camera noise (zwfs1: seed, zwfs2: seed+1)


print('camera frames   :', zwfs.cam.frame.shape)                
print('valid pixels    :', zwfs.nSignal)                     
print('camera noise    :', zwfs.cam.photonNoise, zwfs.cam.readoutNoise, zwfs.cam.darkCurrent)   # False 0 0

# ------------------------------------------------------------- 2. interaction matrix (noise-free)
calib = InteractionMatrix(ngs=src,
                          tel=tel,
                          dm=dm,
                          wfs=zwfs,
                          M2C=M2C_KL,
                          stroke=10e-9,
                          nMeasurements=1,
                          noise='off',
                          single_pass = False,
                          display=True)
reconstructor = M2C_KL @ calib.M                            # signal -> DM commands

# ------------------------------------------------------------- 3. real camera, before the loop

cam_param = {
    'quantization' : None,     
    'FWC'          : None,  
    'sensor'       : 'CCD',
    'QE'           : 0.8,
    'darkCurrent'  : 334,      
    'readoutNoise' : 25,     
    'photonNoise'  : True,
    'gain'         : 1
    }

# the same properties are applied to the two cameras, then the reference is recalibrated (noise-free)
zwfs.set_camera(sensor          =cam_param['sensor'],
                bits            =cam_param['quantization'],              
                FWC             =cam_param['FWC'],                        
                QE              =cam_param['QE'],
                darkCurrent     =cam_param['darkCurrent'],
                readoutNoise    =cam_param['readoutNoise'],
                photonNoise     =cam_param['photonNoise'],
                gain            =cam_param['gain'],
                integrationTime =tel.samplingTime)

print('dark level      :', zwfs.dark_level, 'camera units/frame') 
# zwfs.subtract_dark = False   # keeps the dark in zwfs.signal and zwfs.img_ZWFS (as for the Pyramid)
print('camera noise    :', zwfs.cam.photonNoise, zwfs.cam.readoutNoise, zwfs.cam.darkCurrent)


# ------------------------------------------------------------- 4. closed loop
n_loop = 100
loop_gain = 0.4
residual = np.zeros(n_loop)
dm.coefs = 0


for k in range(n_loop):
    atm.update()
    src**atm*tel*dm*zwfs                                     
    residual[k] = np.std(src.OPD[tel.pupil == 1])*1e9        
    dm.coefs = dm.coefs - loop_gain*reconstructor@zwfs.signal
    strehl = np.exp(-np.var(tel.src.phase[np.where(tel.pupil == 1)]))
    print(strehl)

print(f'residual: {residual[0]:.0f} nm (first frame) -> {residual[-10:].mean():.0f} nm (last 10 frames)')
print('raw frame type  :', zwfs.cam.frame.dtype, '| max', zwfs.cam.frame.max(),
      '' if cam_param['quantization'] is None else f"ADU of {2**cam_param['quantization']-1}")

# -------------------------------------------------------------
raw_frames   = zwfs.cam.frame                      # raw camera output (integers if cam.bits is 8/16/32/64)
frames       = zwfs.frame_dark_subtracted      # float, dark subtracted
img_photons  = zwfs.img_ZWFS                   # photons per camera pixel per frame
phase_rec    = zwfs.reconstructor('asin', iteration=20)   # phase from the ZWFS image [rad], camera grid

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].imshow(frames)
ax[0].set_title('raw camera frame')
ax[1].imshow(phase_rec)
ax[1].set_title('asin phase [rad]')
ax[2].plot(residual)
ax[2].set_title('residual [nm]'); ax[2].set_xlabel('loop iteration')
plt.show()

#------------------------------------------------------------- notes
# NOTE apparently dark is not subtracted in the Pyramid either (its reference is noise-free too)

