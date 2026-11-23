# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:42:34 2024

@author: mmotte

v2: ZWFS2.py but with detectors.
"""

import sys
import warnings

import numpy as np
from vZWFS.ZWFS_v2 import ZWFS
from OOPAO.tools.tools import OopaoError


class _CameraPair:
    # wfs.cam of the ZWFS2: setting a property sets it on the two cameras (e.g. InteractionMatrix(noise='off')),
    # reading a property returns the one of zwfs1.cam (the two camera frames side by side are in wfs.frame)
    def __init__(self, cams):
        object.__setattr__(self, '_cams', cams)
    def __getattr__(self, name):
        return getattr(self._cams[0], name)
    def __setattr__(self, name, value):
        for cam in self._cams:
            setattr(cam, name, value)


class ZWFS2:
    def __init__(self, tel = None, diameter:float = None, phase_shift = [-np.pi/2,np.pi/2], flux_ratio:float = 1/2, transmittance:complex = 1, zpf = 4, phase_shift_unit = 'radian', propagation_method = 'MFT', ZWFS1 = None, ZWFS2=None,
                 nSubap:int = None, n_pix_edge:int = 0, binning:int = 1, user_valid_pixels = None, seed:int = None):
        """ Two ZWFS sharing the flux of the telescope (flux_ratio in zwfs1, 1-flux_ratio in zwfs2), each with its own camera.
        nSubap, n_pix_edge, binning, user_valid_pixels: camera of each ZWFS (see ZWFS_v2). seed: noise of zwfs1, seed+1 for zwfs2.
        With n_pix_edge, the two pupils of wfs.frame are separated by 2*n_pix_edge pixels (as the Pyramid pupils with n_pix_separation = 2*n_pix_edge).

        ************************** MAIN PROPERTIES **************************
        The properties of the two ZWFS are concatenated (camera grid, zwfs1 on the left):
        _ wfs.raw_data    : the two noise-free intensities [photons/px/frame], telescope grid (2 x tel.resolution wide)
        _ wfs.frame       : the two camera frames side by side (as wfs.cam.frame for one WFS), raw camera output
        _ wfs.frame_dark_subtracted : wfs.frame in float with the mean dark level subtracted (wfs.dark_level, wfs.subtract_dark)
        _ wfs.img_ZWFS    : the two images in [photons/camera pixel/frame]
        _ wfs.signal_2D   : the two 2D signals, wfs.signal the valid pixels of zwfs1 then of zwfs2
        _ wfs.valid_pixels (alias wfs.pupil_mask), wfs.img_ref, wfs.ref_signal
        """

        if tel is not None:
            self.telescope = tel
        if tel is None and (ZWFS1 is None or ZWFS2 is None):
            raise ValueError('If at least one ZWFS is not entered manually, a telescope object needs to be input in order to create the ZWFS class')
        if isinstance(phase_shift, float):
            phase_shift = [phase_shift-np.pi,phase_shift]
        self.phase_shift = np.array(phase_shift)
        self.phase_shift_unit = phase_shift_unit
        # independent camera noise for the two ZWFS
        seeds = [None, None] if seed is None else [seed, seed + 1]
        camera_kwargs = dict(nSubap = nSubap, n_pix_edge = n_pix_edge, binning = binning, user_valid_pixels = user_valid_pixels)
        if ZWFS1 is None:

            self.zwfs1 = ZWFS( tel=self.telescope, diameter=diameter, phase_shift = phase_shift[0], transmittance=flux_ratio*transmittance, zpf = zpf, phase_shift_unit=phase_shift_unit, propagation_method=propagation_method, seed = seeds[0], **camera_kwargs)
        else:
            self.zwfs1 = ZWFS1
        if ZWFS2 is None:
            self.zwfs2 = ZWFS( tel=self.telescope, diameter=diameter, phase_shift = phase_shift[1], transmittance=(1-flux_ratio)*transmittance, zpf = zpf, phase_shift_unit=phase_shift_unit, propagation_method=propagation_method, seed = seeds[1], **camera_kwargs)
        else:
            self.zwfs2 = ZWFS2


        self.zpf = zpf
        self.cam = _CameraPair([self.zwfs1.cam, self.zwfs2.cam])
        # flat wavefront (also resets tel.src.phase, which can still hold a calibration cube)
        self.wfs_measure(phase_in=np.zeros(self.zwfs1.telescope.pupil.shape))
        self._ref_signal = np.append(self.zwfs1.ref_signal, self.zwfs2.ref_signal)
        self._signal = np.append(self.zwfs1.signal, self.zwfs2.signal)
        # valid pixels of the two ZWFS (camera grid), pupil_mask as in ZWFS2.py
        self.valid_pixels = np.concatenate((self.zwfs1.valid_pixels ,self.zwfs2.valid_pixels), axis = 1)
        self.pupil_mask = self.valid_pixels
        self._img_ref = np.concatenate((self.zwfs1.img_ref ,self.zwfs2.img_ref), axis = 1)
        self._img_ZWFS = np.concatenate((self.zwfs1.img_ZWFS,self.zwfs2.img_ZWFS), axis = 1)
        self.tag = 'ZWFS'
    def wfs_measure(self, phase_in = None, reconstructor = None, reconstructor_iteration = 1, known_EM = False):
        if reconstructor_iteration<1:
            warnings.warn('Number of iteration ofr the reconstructor should be superior or equal to 1, Taking 1')
            reconstructor_iteration = 1
        self.zwfs1.wfs_measure(phase_in, known_EM=known_EM)
        self.zwfs2.wfs_measure(phase_in, known_EM=known_EM)
        if reconstructor is not None:
            self.retrieved_phase = self.reconstructor(reconstructor, iteration= reconstructor_iteration)

        self.frame = np.concatenate((self.zwfs1.cam.frame, self.zwfs2.cam.frame), axis = 1)
        self.signal_2D = np.concatenate((self.zwfs1.signal_2D, self.zwfs2.signal_2D), axis = 1)

        self.nSignal = self.zwfs1.nSignal + self.zwfs2.nSignal

    def wfs_calibration(self):
        # reference of the two ZWFS (noise-free camera, see ZWFS_v2)
        self.zwfs1.wfs_calibration()
        self.zwfs2.wfs_calibration()
        self.valid_pixels = np.concatenate((self.zwfs1.valid_pixels ,self.zwfs2.valid_pixels), axis = 1)
        self.pupil_mask = self.valid_pixels

    def set_camera(self, **properties):
        # apply the same detector properties to the two ZWFS cameras (see Detector class), then recalibrate
        for zwfs in [self.zwfs1, self.zwfs2]:
            for name, value in properties.items():
                if not hasattr(zwfs.cam, name):
                    raise OopaoError(f'The Detector has no property {name}')
                setattr(zwfs.cam, name, value)
            if 'bits' in properties and 'output_precision' not in properties:
                # the Detector fixes its output type (uint for 8/16/32/64 bits, else float) at the first readout only
                zwfs.cam.output_precision = None
        self.wfs_calibration()

    @property
    def subtract_dark(self):
        return self.zwfs1.subtract_dark and self.zwfs2.subtract_dark
    @subtract_dark.setter
    def subtract_dark(self, val):
        self.zwfs1.subtract_dark = self.zwfs2.subtract_dark = bool(val)

    @property
    def dark_level(self):
        return np.array([self.zwfs1.dark_level, self.zwfs2.dark_level])

    @property
    def frame_dark_subtracted(self):
        return np.concatenate((self.zwfs1.frame_dark_subtracted, self.zwfs2.frame_dark_subtracted), axis = 1)

    def reinitialise_class(self):
        self.zwfs1.reset_Ib_beta()
        self.zwfs2.reset_Ib_beta()
        self.wfs_measure(phase_in=np.zeros(self.telescope.pupil.shape))

    def _filter_modes(self, projector, phase):
        projected_phase=projector@phase.ravel()
        return projected_phase

    def _compute_proj(self, modal_basis):
        pupil = self.zwfs1.valid_pixels & self.zwfs2.valid_pixels
        modal_basis = modal_basis.reshape((-1, modal_basis.shape[-1]))/modal_basis[pupil, :].std(axis=0)  # Flatten for projection
        cov_modes = modal_basis.T @ modal_basis  # Compute mode covariance
        return np.diag(1 / np.diag(cov_modes)) @ modal_basis.T  # Pseudo-inverse projection matrix
    def reconstructor(self, reconstructor = 'atan', iteration = 1, damping_iteration = 1, filter_modes = False, modal_basis = None, print_error = False):
        # the phase is reconstructed on the camera grid (valid pixels of each ZWFS instead of tel.pupil)
        valid1 = self.zwfs1.valid_pixels
        valid2 = self.zwfs2.valid_pixels
        clip = False
        self.zwfs1.reset_Ib_beta()
        self.zwfs2.reset_Ib_beta()
        self.zwfs1.sin_phase = self.zwfs1.phase_sin(clipping=clip)
        self.zwfs2.sin_phase = self.zwfs2.phase_sin(clipping=clip)
        phase1 = np.zeros(np.append(iteration+1,valid1.shape))
        phase2 = np.zeros(np.append(iteration+1,valid2.shape))
        tol = 1e-5
        if filter_modes:
            if modal_basis is None:
                raise AttributeError('Modal basis must not be empty when filter modes is None')
            if modal_basis[...,0].shape !=valid1.shape:
                raise ValueError('Axes (0,1) of modal basis should be the shape of the camera grid and axis 3 should be the modes to project the phase on')
            self.modal_basis = modal_basis/modal_basis[valid1&valid2, :].std(axis = 0)
            projector = self._compute_proj(self.modal_basis)
        convergence = []
        for i in range(iteration):
            self.sin_phase = np.concatenate((self.zwfs1.sin_phase,self.zwfs2.sin_phase), axis = 0)
            M11 = -np.sin(self.zwfs1.beta+self.zwfs1.phase_shift/2)
            M11[np.isnan(M11)]=0
            M12 =  np.cos(self.zwfs1.beta+self.zwfs1.phase_shift/2)
            M12[np.isnan(M12)]=0
            M21 = -np.sin(self.zwfs2.beta+self.zwfs2.phase_shift/2)
            M21[np.isnan(M21)]=0
            M22 = np.cos(self.zwfs2.beta+self.zwfs2.phase_shift/2)
            M22[np.isnan(M22)]=0
            self.det_M = M11*M22-M12*M21
            eps = 1e-5
            bad = np.abs(self.det_M) < eps

            if (bad.any() & (self.det_M[valid2&valid1]==0)).sum()>0:
                warnings.warn('Some pixels has a null determinant')
            det = self.det_M.copy()


            self.cos_phi = (M22*self.zwfs1.sin_phase-self.zwfs2.sin_phase*M12)/det
            self.sin_phi = (-M21*self.zwfs1.sin_phase+self.zwfs2.sin_phase*M11)/det
            if reconstructor == 'linear':
                retrieved_phase = self.sin_phi/self.cos_phi

            if reconstructor == 'atan':
                retrieved_phase = np.arctan2(self.sin_phi,self.cos_phi)
                if filter_modes:
                    projected_phase = self._filter_modes(projector, retrieved_phase)
                    retrieved_phase = (self.modal_basis*projected_phase[None,None, :]).sum(axis=-1)
            if iteration>1:

                phase1[i+1, valid1] =(1-damping_iteration)*phase1[i,valid1] +damping_iteration*retrieved_phase[valid1]
                phase1[i+1, valid1]-=phase1[i+1, valid1].mean()
                phase1[i+1, ~valid1] = 0
                phase2[i+1, valid2] = (1-damping_iteration)*phase2[i,valid2] +damping_iteration*retrieved_phase[valid2]
                phase2[i+1, valid2]-=phase2[i+1, valid2].mean()
                phase2[i+1, ~valid2] = 0
                err1 = np.linalg.norm(phase1[i+1, valid1] - phase1[i,valid1]) / np.sqrt(np.sum(valid1))
                err2 = np.linalg.norm(phase2[i+1, valid2] - phase2[i, valid2]) / np.sqrt(np.sum(valid2))

                err = max(err1, err2)
                convergence.append(err)
                if print_error:
                    if err < tol:
                        print(
                            f"\rConvergence atteinte à l'itération {i}, erreur = {err:.3e}".ljust(100),
                            file=sys.stderr,
                            flush=True,
                        )
                        break
                    else:
                        print(
                            f"\rErreur interne : {err:.3e}".ljust(100),
                            end="",
                            file=sys.stderr,
                            flush=True,
                        )
                # reference wave from the estimated phase (camera grid -> telescope grid for the propagation)
                self.zwfs1.reset_Ib_beta(phase = self.zwfs1._to_tel(phase1[i+1]))
                self.zwfs1.sin_phase = self.zwfs1.phase_sin(clipping=clip)
                self.zwfs2.reset_Ib_beta(phase = self.zwfs2._to_tel(phase2[i+1]))
                self.zwfs2.sin_phase = self.zwfs2.phase_sin(clipping=clip)
        self.convergence = convergence
        retrieved_phase[valid2&valid1]-=retrieved_phase[valid2&valid1].mean()
        retrieved_phase[~valid2|~valid1]=0
        return retrieved_phase
    def _unconcatenate(self,vzimg):
        z1 = vzimg[...,:,:vzimg.shape[-1]//2]
        z2 = vzimg[...,:,vzimg.shape[-1]//2:]
        return z1, z2
    @property
    def raw_data(self):
        return np.concatenate((self.zwfs1.raw_data, self.zwfs2.raw_data), axis = 1)
    @property
    def img_ZWFS(self):
        self._img_ZWFS = np.concatenate((self.zwfs1.img_ZWFS,self.zwfs2.img_ZWFS), axis = 1)
        return self._img_ZWFS
    @img_ZWFS.setter
    def img_ZWFS(self,val):
        self.zwfs1.img_ZWFS, self.zwfs2.img_ZWFS = self._unconcatenate(val)
        self._img_ZWFS = val
    @property
    def ref_signal(self):
        return np.append(self.zwfs1.ref_signal, self.zwfs2.ref_signal)
    @ref_signal.setter
    def ref_signal(self,val):
        self.zwfs1.ref_signal = val[:self.zwfs1.nSignal]
        self.zwfs2.ref_signal = val[self.zwfs1.nSignal:]
        self._img_ref = np.concatenate((self.zwfs1.img_ref ,self.zwfs2.img_ref), axis = 1)
    @property
    def signal(self):
        self._signal = np.concatenate((self.zwfs1.signal, self.zwfs2.signal), axis = 0)
        return self._signal
    @property
    def img_ref(self):
        self._img_ref = np.concatenate((self.zwfs1.img_ref ,self.zwfs2.img_ref), axis = 1)
        return self._img_ref
    @img_ref.setter
    def img_ref(self, val):
        img1, img2 = self._unconcatenate(val)
        self.zwfs1.img_ref = img1
        self.zwfs2.img_ref = img2
        self._img_ref = val
        self._ref_signal = np.append(self.zwfs1.ref_signal, self.zwfs2.ref_signal)

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
            frames_list.append(self.frame)

        self.signals_2D = np.squeeze(np.array(signal_2D_list))
        self.signals = np.squeeze(np.array(signal_list))
        self.frames = np.squeeze(np.array(frames_list))

        return
