# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 11:33:27 2024

@author: mmotte

v2: the detector follows the Pyramid / ShackHartmann convention (noise-free wfs.cam by default,
WFS signal computed from the camera frame). The previous version is kept in ZWFS_mine.py for comparison.
GPU: same backend convention as the Pyramid (see OOPAO.runtime and the GPU section of the ZWFS docstring).
"""
import warnings
import matplotlib
import os
import sys
if 'ipykernel' in sys.modules:
    pass
elif os.environ.get('DISPLAY'):
    matplotlib.use('TkAgg')
else:
    matplotlib.use('Agg')  # headless (no X display)

from contextlib import contextmanager

import numpy as np

from .Detector import Detector
from .tools.tools import set_binning, warning, OopaoError
from .runtime import array_backend, gpu_resident, precision_bits
from .runtime import backend_of as _backend_of, stack_squeeze as _stack_squeeze, to_backend as _to_backend, to_numpy as _to_numpy

xp, global_gpu_flag = array_backend()
if global_gpu_flag:
    fft2 = xp.fft.fft2
    ifft2 = xp.fft.ifft2
    fftshift = xp.fft.fftshift
else:
    from numpy.fft import fft2, ifft2, fftshift

import logging  # For logging messages


logging.basicConfig(level=logging.INFO)  # Set up logging
logger = logging.getLogger(__name__)  # Create logger

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


#%%

class ZWFS:
    def __init__(
        self,
        tel,
        diameter: float = 1.06,
        phase_shift: float = np.pi/2,
        transmittance: float = 1.0,
        zpf: int = 4,
        phase_shift_unit: str = "radian",
        propagation_method: str = "MFT",
        nSubap: int = None,
        n_pix_edge: int = 0,
        binning: int = 1,
        user_valid_pixels = None,
        seed: int = None,
        ):
        """ ZWFS
        The ZWFS consist in a 2D mask at the focal plane of the telescope to perform the Fourier Filtering of the EM-Field.
        By default the ZWFS detector is considered to be noise-free (for calibration purposes). These properties can be switched on and off on the fly (see wfs.cam).

        Parameters
        ----------
        tel: TYPE
            The telescope object to which the ZWFS is associated. This object carries the phase, flux and pupil information.
        diameter: float
            The diameter of the ZWFS mask in lambda/D: default value 1.06
        phase_shift: float
            The phase shift induce by the ZWFS in radian, lambda or pi depending on chosen phase shift unit: default value pi/2
        transmittance: float
            The portion of flux going into the mask
        zpf: int
            In FFT propagation method, it is the Zero padding factor. In MFT propagation method it corresponds to the diameter of the mask in fourier plane (ie the resolution of the fourier transform inside the mask): default value 4.
        phase_shift_unit: string
            If one want to express the depth in lambda, pi, or radian. default : radian
        propagation_method: string
            Select the fourier transform propagation method: default MFT, possibility with FFT but MFT recommended.
        nSubap: int
            Number of pixels across the pupil on the ZWFS detector (as for the Pyramid). It has to divide tel.resolution.
            default : None (tel.resolution, one detector pixel per telescope pixel)
        n_pix_edge: int
            Number of detector pixels around the pupil (as n_pix_edge for the Pyramid): the detector has nSubap + 2*n_pix_edge pixels.
            These pixels receive no light (the light outside the pupil is removed) but get the camera noise. default : 0
        binning: int
            Additional binning factor of the ZWFS detector signals (as for the Pyramid). default : 1
        user_valid_pixels: bool array
            User-defined valid pixels on the camera grid. default : None (pupil binned on the camera grid)
        seed: int
            Seed of the camera noise. default : None (independent random streams for each ZWFS)

        ************************** PROPAGATING THE LIGHT TO THE ZWFS OBJECT **************************
        The light can be propagated from a telescope object tel through the ZWFS object wfs using the * operator:
        _ tel*wfs
        This operation will trigger:
            _ propagation of the tel.src light through the ZWFS mask (phase and flux) -> wfs.raw_data
            _ integration of wfs.raw_data by the ZWFS camera wfs.cam (noise, QE, binning...) -> wfs.cam.frame
            _ computation of the ZWFS signals from the camera frame -> wfs.signal and wfs.signal_2D

        If the tel.src object is an asterism, each source is propagated to the ZWFS and measured separately
        (as in the Pyramid relay): wfs.signals, wfs.signals_2D and wfs.frames are stacked along the first axis.

        ************************** UNITS **************************
        Everything is derived from the source flux:
        _ the incoming field has the amplitude sqrt(src.intensity) [sqrt(photons/px/frame)], scintillation included
        _ the model of the reference wave (Ib, beta) and of the pupil intensity Ip uses the amplitude sqrt(src.fluxMap),
          i.e. the known source flux without scintillation (or the actual field if known_EM=True)
        _ the camera frame is converted back to photons with the camera response measured at the calibration

        ************************** MAIN PROPERTIES **************************
        _ wfs.raw_data                  : perfect intensity in the detector plane [photons/pixel/frame], telescope grid
        _ wfs.cam.frame                 : camera frame [QE and gain applied + camera noise]
        _ wfs.img_ZWFS                  : camera frame in [photons/camera pixel/frame] (with camera noise if enabled), same units as wfs.Ip and wfs.Ib
                                        (input of phase_sin and of the reconstructor).
                                        Equal to wfs.raw_data for a noise-free camera with nSubap = tel.resolution
        _ wfs.frame_dark_subtracted     : wfs.cam.frame converted to float - dark_level
        _ wfs.norma                     : incident flux on the valid pixels [photons/frame]
        _ wfs.signal_2D                 : img_ZWFS/norma - img_ref/norma_ref on the valid pixels
        _ wfs.signal                    : wfs.signal_2D[wfs.valid_pixels]

        ************************** DETECTOR **************************
        The detector noise can be set as for the Pyramid (see Detector class for more details):
            _ wfs.cam.photonNoise, wfs.cam.readoutNoise, wfs.cam.darkCurrent, wfs.cam.QE ...
            _ or wfs.set_camera(QE = 0.8, darkCurrent = 334, ...), which also calls wfs.wfs_calibration()
            _ wfs.subtract_dark = False switches dark subtraction off (the Pyramid does not subtract the dark).

        ************************** GPU **************************
        Same convention as the Pyramid (see OOPAO.runtime):
        _ the light propagation (MFT or FFT) runs on the GPU when CuPy and a CUDA device are available (OOPAO_BACKEND=auto/cuda),
          at the OOPAO precision (OOPAO_PRECISION)
        _ by default the results are brought back to the CPU: raw_data, the camera frames, Ib, beta, Ip, img_ZWFS, the signals,
          sin_phase and the reconstructed phase are NumPy arrays, as on a CPU-only machine
        _ with OOPAO_GPU_RESIDENT=1 they stay on the GPU (CuPy arrays): camera noise, signal processing and reconstruction run
          on the GPU without read-backs. wfs.valid_pixels, wfs.Z_bool, wfs.Zmask and the calibration (img_ref aside) stay NumPy.
        _ a cube of phase screens (e.g. InteractionMatrix with nMeasurements > 1) returns NumPy signals, as the Pyramid.
        """
        self.gpu_available = global_gpu_flag
        # results kept on the GPU only with GPU residency (OOPAO_GPU_RESIDENT=1), as the Pyramid
        self.gpu_resident = gpu_resident()
        # backend of the results (raw_data, camera frames, model, signals)
        self._backend = xp if self.gpu_resident else np
        precision = precision_bits()
        self.precision = np.float64 if precision == 64 else np.float32
        self.precision_complex = np.complex128 if precision == 64 else np.complex64

        if propagation_method != 'FFT' and propagation_method != 'MFT':
            warnings.warn('Unknown propagation method, using FFT')
            self.propagation_method = 'FFT'
        else:
            self.propagation_method = propagation_method

        self.telescope = tel
        tel.print_properties()
        self.tag = 'ZWFS'
        self.pupil_mask = (_to_numpy(self.telescope.pupil) != 0)
        if propagation_method == 'FFT':
            if zpf <=2:
                warnings.warn('Zero-padding factor should be superior to 2 for an FFT propagation method')
            else:
                self.zpf = zpf
                self.resolution = np.int64(self.zpf*self.telescope.resolution)
        elif propagation_method == 'MFT':
            if zpf <=4:
                warnings.warn('Number of pixel inside the mask should be superior to 4 for a MFT propagation method')
            else:
                self.resolution = zpf
                self.zpf = zpf


        self.transmittance = transmittance
        self.diameter = np.float64(diameter)
        if diameter <=0:
            warnings.warn(f'{diameter} is not a possible diameter, using d=1.06 instead')
            self.diameter = 1.06


        self.radius = self.diameter/2
        if phase_shift_unit == 'lambda':
            self.phase_shift = phase_shift * 2 * np.pi
        elif phase_shift_unit == 'radian':
            self.phase_shift = float(phase_shift)
        elif phase_shift_unit == 'pi':
            self.phase_shift = float(phase_shift) * np.pi
        else:
            raise ValueError("phase_shift_unit must be 'lambda', 'radian', or 'pi'")

        # factor applied to the field filtered by the dot: EM_det = EM - epsilon*EM_b
        self.epsilon = (1-np.exp(1j*self.phase_shift))

        self.pixel_size = self.telescope.D/self.resolution
        self.Z_bool = self.ZWFS_bool()
        self.Zmask = self.ZWFS_mask()


        self.amplitude_mask = self.Z_bool*np.sqrt(self.transmittance)
        if self.propagation_method == 'MFT':
            self._precompute_mft()

        # number of pixels across the pupil on the detector (as for the Pyramid)
        if nSubap is None:
            nSubap = self.telescope.resolution
        if self.telescope.resolution % nSubap != 0:
            raise OopaoError(f'nSubap ({nSubap}) must divide the telescope resolution ({self.telescope.resolution})')
        self.nSubap = int(nSubap)
        self.n_pix_edge = int(n_pix_edge)
        self._pad_tel = self.n_pix_edge*(self.telescope.resolution//self.nSubap)
        self.binning = binning
        self.user_valid_pixels = user_valid_pixels
        # telescope pixels per camera pixel (set by wfs_calibration from the camera frame)
        self.cam_binning_factor = 1
        self.isCalibrated = False
        self.subtract_dark = True

        # WFS detector object (see Detector class), noise-free by default
        self.cam = Detector(self.nSubap + 2*self.n_pix_edge)
        self.seed_camera(seed)
        # reference acquisition
        self.wfs_calibration()


    def ZWFS_bool(self):
        if self.propagation_method == 'MFT':
            Zradius_pixels = self.resolution/2
            center = (self.resolution/2-0.5,self.resolution/2-0.5)
        else:
            Zradius_pixels = self.zpf*self.radius
            center = (self.resolution/2,self.resolution/2)

        Y, X = np.ogrid[:self.resolution, :self.resolution]
        Z_bool = (X - center[0])**2 + (Y - center[1])**2 <= Zradius_pixels**2
        return Z_bool
    def ZWFS_mask(self, phase = None):
        if phase is None:
            phase = self.phase_shift
        Zmask= (1-(1-np.exp(1j*phase))*self.Z_bool)*np.sqrt(self.transmittance)

        return Zmask
    def propagation(self, mask = None, phase = None, epsilon = None, computing_Ib = False, amplitude = None) :
        # amplitude: pupil amplitude of the field [sqrt(photons/px/frame)], by default the source flux (no scintillation)
        # computed on the propagation backend (CuPy when a GPU is available): EM_det and IM_detect are on that backend
        if phase is None:
            phase = self.telescope.src.phase
        if amplitude is None:
            amplitude = self.flux_amplitude
        EM = xp.asarray(amplitude, dtype=self.precision)*xp.exp(1j*xp.asarray(phase, dtype=self.precision))

        if self.propagation_method == 'MFT':
            _ , EM_b = self._MFT_fast(EM, mask = self._xp_copy('Z_bool', self.Z_bool))
            if computing_Ib:
                EM_det = EM_b*np.sqrt(self.transmittance)
            else:
                EM_det = (EM-(1-np.exp(1j*self.phase_shift))*EM_b)*np.sqrt(self.transmittance)
        else:
            if mask is None:
                mask = self._xp_copy('Zmask', self.Zmask, self.precision_complex)
            elif mask is self.amplitude_mask:
                mask = self._xp_copy('amplitude_mask', self.amplitude_mask, self.precision)
            else:
                mask = xp.asarray(mask)
            N = int(self.resolution)
            n = self.telescope.resolution
            center = slice(N//2-n//2, N//2+n//2)
            EM_pad = xp.zeros((N, N), dtype=self.precision_complex)
            EM_pad[center, center] = EM
            Psi_turb = fftshift(fft2(fftshift(EM_pad))) # 1er prop
            Psi_FP_turb = mask*Psi_turb # PF multiplication par masque
            EM_det = fftshift(ifft2(fftshift((Psi_FP_turb))))[center, center] # PP detecteur
        # light outside the pupil is removed (ZWFS_mine convention, as in the CNN training frames); ZWFS.py keeps it
        EM_det = EM_det*self._xp_copy('pupil_mask', self.pupil_mask)
        IM_detect = xp.abs(EM_det)**2
        return EM_det,IM_detect

    def _xp_copy(self, name, arr, dtype = None):
        # copy of a NumPy mask on the propagation backend, made again only when wfs.<name> is replaced
        cache = self.__dict__.setdefault('_xp_cache', {})
        entry = cache.get(name)
        if entry is None or entry[0] is not arr:
            entry = (arr, xp.asarray(arr, dtype = dtype))
            cache[name] = entry
        return entry[1]

    def _set_detector_plane(self, EM_det, IM_detect):
        # field kept on the propagation backend (read back by the EM_detect property), intensity on the backend of the results
        self._EM_detect = EM_det
        self.raw_data = _to_backend(IM_detect, self._backend)

    @property
    def EM_detect(self):
        # field in the detector plane, telescope grid
        return _to_backend(self._EM_detect, self._backend)

    @property
    def flux_amplitude(self):
        # pupil amplitude known from the source flux [sqrt(photons/px/frame)]:
        # src.fluxMap = pupilReflectivity*nPhoton*samplingTime*(D/resolution)**2, without scintillation
        b = self._backend
        return b.sqrt(_to_backend(self.telescope.src.fluxMap, b).astype(float))

    def _field_amplitude(self):
        # amplitude of the incoming field [sqrt(photons/px/frame)], scintillation included
        # (src.intensity = fluxMap*scintillation, as for the Pyramid and the ShackHartmann)
        b = self._backend
        return b.sqrt(_to_backend(self.telescope.src.intensity, b).astype(float))

    def _to_cam(self, arr, mode = 'sum'):
        # telescope grid -> camera grid: edge pixels added (zeros), then sum over the binned pixels as the detector does
        # (on the backend of arr)
        b = _backend_of(arr)
        arr = b.asarray(arr)
        if self._pad_tel > 0:
            arr = b.pad(arr, self._pad_tel)
        if self.cam_binning_factor == 1:
            return arr
        return set_binning(arr, self.cam_binning_factor, mode = mode)

    def _to_tel(self, arr):
        # camera grid -> telescope grid (edge pixels removed)
        if self.cam_binning_factor != 1:
            arr = arr.repeat(self.cam_binning_factor, axis=0).repeat(self.cam_binning_factor, axis=1)
        p = self._pad_tel
        return arr[p:arr.shape[0]-p, p:arr.shape[1]-p]

    @property
    def valid_pixels(self):
        # valid pixels on the camera grid (NumPy bool array)
        return self._valid_pixels
    @valid_pixels.setter
    def valid_pixels(self, val):
        val = _to_numpy(val).astype(bool)
        self._valid_pixels = val
        # copies on the backend of the results; integer indices avoid the GPU synchronisation of a[mask]
        self._valid_pixels_b = _to_backend(val, self._backend)
        self._valid_index_b = _to_backend(np.flatnonzero(val), self._backend)
        self.nSignal = int(val.sum())

    def _valid_values(self, arr):
        # arr[valid_pixels] without boolean indexing
        return arr.ravel()[self._valid_index_b]

    def _from_valid_values(self, val):
        # 2D map on the camera grid, val on the valid pixels and 0 elsewhere
        b = self._backend
        out = b.zeros(self.valid_pixels.size)
        out[self._valid_index_b] = _to_backend(val, b)
        return out.reshape(self.valid_pixels.shape)

    def seed_camera(self, seed = None):
        # Detector seeds its random states with int(time.time()): two ZWFS created in the same second
        seeds = np.random.SeedSequence(seed).generate_state(4)
        for name, s in zip(['photon', 'readout', 'background', 'dark_shot'], seeds):
            setattr(self.cam, 'random_state_' + name + '_noise', np.random.RandomState(int(s)))

    @contextmanager
    def _noise_free_camera(self):
        # switch off the camera noise (and the dark/background terms) for the calibration
        # QE, gain and quantisation stay active so that the camera response is measured
        properties = ['photonNoise', 'readoutNoise', 'backgroundNoise', 'darkCurrent', 'backgroundMap', 'sensor', 'integrationTime']
        saved = {name: getattr(self.cam, name) for name in properties}
        try:
            self.cam.photonNoise = False
            self.cam.readoutNoise = 0
            self.cam.backgroundNoise = False
            self.cam.darkCurrent = 0
            self.cam.backgroundMap = None
            # no EMCCD amplification noise, same mean gain
            if self.cam.sensor == 'EMCCD':
                self.cam.sensor = 'CCD'
            # read out the reference frame immediately
            self.cam.integrationTime = None
            yield
        finally:
            for name, value in saved.items():
                setattr(self.cam, name, value)

    def set_camera(self, **properties):
        # set the detector properties (see Detector class), then recalibrate the reference and the camera response
        for name, value in properties.items():
            if not hasattr(self.cam, name):
                raise OopaoError(f'The Detector has no property {name}')
            setattr(self.cam, name, value)
        if 'bits' in properties and 'output_precision' not in properties:
            # the Detector fixes its output type (uint for 8/16/32/64 bits, else float) at the first readout only
            self.cam.output_precision = None
        self.wfs_calibration()

    @property
    def dark_level(self):
        """ Mean dark level of one camera frame [camera units], computed from the current camera properties
        (no measurement, so it follows any change of the camera). It follows the Detector readout chain:
            dark [e-] = darkCurrent*integrationTime -> EMCCD gain -> hardware binning (sum of binning**2 pixels)
            -> CCD/CMOS gain -> ADC (x (2**bits-1)/FWC if bits is set)
        Readout noise and dark shot noise have zero mean and stay in the frames.
        The clipping at 0 ADU and the truncation of an integer camera output are not included: the level is exact for
        the illuminated pixels (never clipped), the pixels without light below 0 ADU are clipped by the camera.
        """
        cam = self.cam
        if cam.darkCurrent == 0:
            return 0.0
        if cam.integrationTime is None:
            raise OopaoError('The dark level needs the camera integrationTime (the Detector dark current is defined per second)')
        level = cam.darkCurrent*cam.integrationTime*cam.binning**2*cam.gain
        if cam.bits is not None:
            if cam.FWC is None:
                raise OopaoError('bits without FWC: the Detector rescales every frame by its own maximum, the dark level is unknown. Set the FWC.')
            level *= (2**cam.bits-1)/cam.FWC
        return level

    def _subtract_dark(self, frame):
        # raw camera frame -> float frame without the mean dark level (if wfs.subtract_dark), on the backend of the results
        frame = _to_backend(frame, self._backend).astype(float)
        if self.subtract_dark:
            frame -= self.dark_level
        return frame

    @property
    def frame_dark_subtracted(self):
        # last camera frame in float, mean dark level subtracted (wfs.cam.frame is the raw camera output)
        return self._subtract_dark(self.cam.frame)

    def wfs_calibration(self):
        """
        Acquires the reference with the noise switched off
        It has to be called again when the camera properties change the frame units or size are modified (QE, gain, bits, binning...).
        """
        flat = xp.zeros(self.pupil_mask.shape, dtype=self.precision)
        self.Amplitude = self.flux_amplitude
        # reference frame through the noise-free camera
        with self._noise_free_camera():
            self._set_detector_plane(*self.propagation(phase = flat, amplitude = self.Amplitude))
            self*self.cam
            ref_frame = _to_numpy(self.cam.frame).astype(float)

        # telescope pixels per camera pixel (nSubap, binning of the WFS and of the detector)
        n_cam = ref_frame.shape[0]
        n_tel = self.telescope.resolution + 2*self._pad_tel
        if n_tel % n_cam != 0:
            raise OopaoError(f'The camera frame ({n_cam} px) must bin the padded pupil ({n_tel} px) by an integer factor')
        self.cam_binning_factor = n_tel//n_cam

        # valid pixels: pupil binned on the camera grid, or user-defined mask
        if self.user_valid_pixels is None:
            self.valid_pixels = self._to_cam(self.pupil_mask.astype(float), mode = 'mean') > 0.5
        else:
            self.valid_pixels = self.user_valid_pixels
            if self.valid_pixels.shape != ref_frame.shape:
                raise OopaoError(f'user_valid_pixels {self.valid_pixels.shape} does not match the camera frame {ref_frame.shape}')

        # camera response [camera units per photon]: QE, gain, quantisation
        photons = self._to_cam(_to_numpy(self.raw_data))
        self.camera_response = ref_frame[self.valid_pixels].sum()/photons[self.valid_pixels].sum()


        # model of the flat wavefront [photons/camera pixel/frame]
        self.Ip = self._to_cam(self.transmittance*self.Amplitude**2)
        self.reset_Ib_beta()

        # reference image [photons/camera pixel/frame] and its incident flux (img_ref setter -> ref_signal)
        self.norma = self._valid_values(self.Ip).sum()
        self.norma_ref = self.norma
        self.img_ref = ref_frame/self.camera_response
        # current image = reference (img_ZWFS setter -> zero signal), as ZWFS.py after its wfs_measure at init
        self.img_ZWFS = self.img_ref.copy()
        self.sin_phase = self.phase_sin()
        if self.isCalibrated is False:
            print('ZWFS calibrated!')
        self.isCalibrated = True

    #similar to Pyramid.py
    def signalProcessing(self, cameraFrame = None):
        """
        converts every camera frame to float and subtracts the mean dark level wfs.dark_level, computed
        from the camera properties (darkCurrent*integrationTime*binning**2*gain, ADC scaling).
        """

        if cameraFrame is None:
            cameraFrame = self.cam.frame
            # frames summed by the camera at the last readout (integrationTime > samplingTime)
            n_frames = max(len(getattr(self.cam, 'buffer', [])), 1)
        else:
            n_frames = 1
        # raw camera frame -> float, mean dark level subtracted (see dark_level)
        frame = self._subtract_dark(cameraFrame)
        if frame.shape != self.valid_pixels.shape:
            raise OopaoError(f'The camera frame {frame.shape} does not match the calibration {self.valid_pixels.shape}: call wfs_calibration() again')
        # normalisation by the incident flux on the valid pixels [photons/frame]
        self.norma = self._valid_values(self.Ip).sum()
        # camera frame -> photons per camera pixel per frame (same units as Ip and Ib); the img_ZWFS setter computes the signals
        self.img_ZWFS = frame/(self.camera_response*n_frames)
        return self.signal_2D, self.signal

    def wfs_measure(self, phase_in = None, reconstruction_method = None, known_EM = False, reconstructor_iteration:int = 1, estimated_phase = 0, integrate = True):
        if reconstructor_iteration<1:
            warnings.warn('Number of iteration ofr the reconstructor should be superior or equal to 1, Taking 1')
            reconstructor_iteration = 1
        if phase_in is not None:
            self.telescope.src.phase = phase_in
        phase = self.telescope.src.phase
        if np.ndim(phase) not in (2, 3):
            raise OopaoError('Wrong dimension for the input phase. Aborting')
        # incoming field amplitude, scintillation included
        amplitude = self._field_amplitude()

        if np.ndim(phase) == 2:
            self._measure_phase(phase, amplitude, known_EM, estimated_phase, integrate)
            if integrate and reconstruction_method is not None:
                self.retrieved_phase = self.reconstructor(reconstruction_method, iteration= reconstructor_iteration)
        else:
            # cube of phase screens (e.g. InteractionMatrix with nMeasurements > 1), measured one by one as for the Pyramid:
            # wfs.signal_2D [camera grid x n] and wfs.signal [nSignal x n] (NumPy, as the Pyramid), the other properties are those of the last screen
            n_phase = phase.shape[2]
            signal_2D = np.zeros(self.valid_pixels.shape + (n_phase,))
            signal = np.zeros((self.nSignal, n_phase))
            for k in range(n_phase):
                self._measure_phase(phase[:, :, k], amplitude[:, :, k] if np.ndim(amplitude) == 3 else amplitude, known_EM, estimated_phase, integrate)
                if integrate:
                    signal_2D[:, :, k], signal[:, k] = _to_numpy(self.signal_2D), _to_numpy(self.signal)
            if integrate:
                self._signal_2D, self._signal = signal_2D, signal

    def _measure_phase(self, phase, amplitude, known_EM, estimated_phase, integrate):
        # measurement of one phase screen
        self.Amplitude = amplitude
        # perfect intensity in the detector plane [photons/pixel/frame], telescope grid
        self._set_detector_plane(*self.propagation(phase = phase, amplitude = self.Amplitude))
        if known_EM:
            # known field: actual phase and intensity
            self.reset_Ib_beta(phase = phase, amplitude = self.Amplitude)
            self.Ip = self._to_cam(self.transmittance*self.Amplitude**2)
        else:
            # estimated phase, intensity known from the source flux
            self.reset_Ib_beta(phase = estimated_phase)
            self.Ip = self._to_cam(self.transmittance*self.flux_amplitude**2)

        if integrate:
            # camera frame -> img_ZWFS, signal_2D, signal
            self.wfs_integrate()
            self.sin_phase = self.phase_sin()

    def phase_sin(self, Ib = None, Ip = None, img_ZWFS = None, phase_shift = None, clipping = True):
        # on the backend of the results
        b = self._backend
        if Ib is None:
            Ib = self.Ib
        if Ip is None:
            Ip = self.Ip
        if img_ZWFS is None:
            img_ZWFS = self.img_ZWFS
        if phase_shift is None:
            phase_shift = self.phase_shift
        Ib, Ip, img_ZWFS = _to_backend(Ib, b), _to_backend(Ip, b), _to_backend(img_ZWFS, b)
        # sin_phase = (img_ZWFS-Ip-4*Ib*(np.sin(phase_shift/2))**2)/(4*np.sqrt(Ib*Ip)*np.sin(phase_shift/2))
        eps = 1e-12
        denominator = 4*b.sqrt(b.maximum(Ib, eps)*b.maximum(Ip, eps))*np.sin(phase_shift/2)
        denominator = b.where(b.abs(denominator) < eps, b.sign(denominator)*eps + (denominator==0)*eps, denominator)
        sin_phase = (img_ZWFS - Ip - 4*Ib*(np.sin(phase_shift/2))**2) / denominator
        if clipping:
            sin_phase = b.clip(sin_phase, -1,1)
        sin_phase = b.where(self._valid_pixels_b, sin_phase, 0.0)
        return sin_phase
    def reinitialise_class(self):
        self.reset_Ib_beta()
        self.wfs_measure(phase_in=np.zeros(self.telescope.pupil.shape))
    def reset_Ib_beta(self, phase = 0, amplitude = None):
        # reference wave b: field through the dot of the mask for the given phase (source flux by default), camera grid
        # (binned on the propagation backend, then moved to the backend of the results)
        Psi_b, Ib = self.propagation(mask = self.amplitude_mask, phase = phase, amplitude = amplitude, computing_Ib = True)
        self.Ib = _to_backend(self._to_cam(Ib), self._backend)
        self.beta = _to_backend(xp.angle(self._to_cam(Psi_b)), self._backend)
    def reconstructor(self, reconstr = 'linear', iteration = 1, clipping = True):
        # the phase is reconstructed on the camera grid, on the backend of the results
        b = self._backend
        valid = self._valid_pixels_b
        self.reset_Ib_beta()

        self.sin_phase = self.phase_sin()
        try:

            for i in range(iteration):
                if reconstr == 'linear':
                    phase = 1/b.cos(self.phase_shift/2+self.beta)*(b.sin(self.phase_shift/2+self.beta)+self.sin_phase)
                    phase = b.where(valid, phase, 0)
                if reconstr == 'asin':
                    phase = self.phase_shift/2+self.beta+b.arcsin(self.sin_phase)
                    phase = phase - self._valid_values(phase).mean()
                    phase = b.where(valid, phase, 0)
                if iteration>1:
                    # |phase| > 2 pi -> sign(phase)
                    phase = b.where(b.abs(phase)>2*np.pi, b.sign(phase), phase)
                    self.reset_Ib_beta(phase = self._to_tel(phase))
                    self.sin_phase = self.phase_sin(clipping = clipping)
            return phase

        except Exception as e:
            logger.warning("Reconstruction failed: %s", e, exc_info=True)
            return None


    def _MFT(self, Na:int, Nb:int, m, EMF_pupil, mask = 1):
        # reference (NumPy) implementation, the propagation uses _MFT_fast
        Nb = np.int64(Nb)
        Na = np.int64(Na)
        x = np.zeros((Na, 1))
        u = np.zeros((Nb, 1))
        x[:,0] = (np.arange(0, Na)-(Na)/2+0.5)*1/Na
        u[:,0] = (np.arange(0, Nb)-(Nb)/2+0.5)*m/Nb
        Ft = (1/(Na**2)*np.exp(-2*1j*np.pi*(u@x.T))@EMF_pupil@np.exp(-2*1j*np.pi*(x@u.T)))
        iFt = (m/Nb)**2*np.exp(2*1j*np.pi*(x@u.T))@(Ft*mask)@np.exp(2*1j*np.pi*(u@x.T))
        return Ft, iFt

    def _precompute_mft(self):
        # MFT matrices on the propagation backend, at the OOPAO precision
        Na = self.telescope.resolution
        Nb = self.resolution
        m = self.diameter

        x = (np.arange(Na) - Na/2 + 0.5) / Na
        u = (np.arange(Nb) - Nb/2 + 0.5) * m / Nb

        self._mft_prefac = m / (Na * Nb)
        self._mft_A = xp.asarray(np.exp(-2j * np.pi * np.outer(u, x)), dtype=self.precision_complex)   # Nb x Na
        self._mft_B = xp.asarray(np.exp(-2j * np.pi * np.outer(x, u)), dtype=self.precision_complex)   # Na x Nb
        self._imft_A = xp.asarray(np.exp( 2j * np.pi * np.outer(x, u)), dtype=self.precision_complex)  # Na x Nb
        self._imft_B = xp.asarray(np.exp( 2j * np.pi * np.outer(u, x)), dtype=self.precision_complex)  # Nb x Na

    def _MFT_fast(self, EMF_pupil, mask=1):
        Ft = self._mft_prefac * (self._mft_A @ EMF_pupil @ self._mft_B)
        iFt = self._mft_prefac * (self._imft_A @ (Ft * mask) @ self._imft_B)
        return Ft, iFt

    # properties and setters of ZWFS.py (same names), with the normalisation: signal = img_ZWFS/norma - img_ref/norma_ref
    # the images and signals are on the backend of the results
    @property
    def img_ZWFS(self):
        # image [photons/camera pixel/frame]
        return self._img_ZWFS
    @img_ZWFS.setter
    def img_ZWFS(self, val):
        val = _to_backend(val, self._backend)
        self._img_ZWFS = val
        self._signal_2D = self._backend.where(self._valid_pixels_b, val/self.norma - self._ref_signal_2D, 0)
        self._signal = self._valid_values(self._signal_2D)
    @property
    def signal(self):
        return self._signal
    @signal.setter
    def signal(self,val):
        val = _to_backend(val, self._backend)
        self._signal_2D = self._from_valid_values(val)
        self._img_ZWFS = (self._signal_2D + self._ref_signal_2D)*self.norma*self._valid_pixels_b
        self._signal = val
    @property
    def signal_2D(self):
        return self._signal_2D
    @property
    def img_ref(self):
        # reference image [photons/camera pixel/frame]
        return self._img_ref
    @img_ref.setter
    def img_ref(self, val):
        val = _to_backend(val, self._backend)
        self._img_ref = val
        self._ref_signal_2D = val/self.norma_ref*self._valid_pixels_b
        self._ref_signal = self._valid_values(self._ref_signal_2D)
    @property
    def ref_signal(self):
        return self._ref_signal
    @ref_signal.setter
    def ref_signal(self,val):
        val = _to_backend(val, self._backend)
        self._ref_signal_2D = self._from_valid_values(val)
        self._img_ref = self._ref_signal_2D*self.norma_ref
        self._ref_signal = val

    def relay(self, src):
        if src.tag == 'source':
            src_list = [src]
        elif src.tag == 'asterism':
            src_list = src.src
        signal_2D_list = []
        signal_list = []
        frames_list = []

        for src in src_list:
            src.optical_path.append([self.tag, self])
            self.src = src
            self.wfs_measure(phase_in=self.src.phase)
            signal_2D_list.append(self.signal_2D)
            signal_list.append(self.signal)
            frames_list.append(self.cam.frame)

        # stacked over the sources (wfs.signal / wfs.signal_2D keep the last source, see the img_ZWFS setter), GPU arrays stay on the GPU
        self.signals_2D = _stack_squeeze(signal_2D_list)
        self.signals = _stack_squeeze(signal_list)
        self.frames = _stack_squeeze(frames_list)

        return

    def wfs_integrate(self):
        # propagate to the detector to apply the noise
        self*self.cam
        if self.isCalibrated:
            return self.signalProcessing()
        else:
            return None, None

    def __mul__(self,obj):
        if obj.tag == 'detector':
            obj._integrated_time += self.telescope.samplingTime
            # edge pixels around the pupil (n_pix_edge), then binning to the detector resolution (on the backend of raw_data)
            intensity = _backend_of(self.raw_data).pad(self.raw_data, self._pad_tel) if self._pad_tel > 0 else self.raw_data
            frame = (obj.set_binning(intensity, intensity.shape[0] / obj.resolution))
            if self.binning != 1:
                try:
                    frame = (obj.rebin(frame, (obj.resolution // self.binning, obj.resolution//self.binning)))
                except:
                    warning('The shape of the detector (' + str(obj.frame.shape) + ')' + 'is not valid with the binning value requested:' + str(self.binning) + '! -- Ignoring the binning.')
            obj.integrate(frame)
        else:
            raise OopaoError('Error light propagated to the wrong type of object')

#%%
