from ozitelemetry.OZITelemetry import OZITele
from pathlib import Path
import numpy as np
import gc


loaddir = 'bench_sky_04_15'
target = 'arcturus'
overwrite = False  # skip files whose phase/OPDs outputs already exist




telemetry = np.load('bench_sky_04_15/onsky_arcturus_1st200_2nd400_v10_20260416-020659/2026-04-16T02_17_55_telemetry_2nd_data_CL1OL2.npy', allow_pickle = True)
telemetry_path = 'bench_sky_04_15/onsky_arcturus_1st200_2nd400_v10_20260416-020659/2026-04-16T02_17_55_telemetry_2nd_data_CL1OL2.npy'

tele = OZITele(tele_path=str(telemetry_path), is_onsky=True)


reference = telemetry.item()['ref'].astype(np.float32).copy()
reference[0, :] = 0
# reference /= reference.sum()

tele.img = reference[None]                        # 1-frame cube through the same pipeline
tele.extract_Zimages()
phase_ref = tele.reconstruct_phase(tele.img_ZWFS1[0], tele.img_ZWFS2[0], 'atan', 0.5, 10)

np.save('bench_sky_04_15/onsky_arcturus_1st200_2nd400_v10_20260416-020659/2026-04-16T02_17_55_telemetry_2nd_data_CL1OL2_reference_phase.npy', phase_ref)






errrr
def find_2nd_stage_files(loaddir, target):
    # full telemetry only: the *_noCRED2 copies lack CRED2Cube, which OZITele needs
    files = sorted(Path(loaddir).glob(f'onsky_{target}_*/*_telemetry_2nd_data_*.npy'))
    return [f for f in files if not f.stem.endswith('_noCRED2')]


if __name__ == '__main__':
    files = find_2nd_stage_files(loaddir, target)
    print(f'Found {len(files)} {target} 2nd stage telemetry files in {loaddir}')

    for i, file_path in enumerate(files, 1):
        savedir = file_path.parent
        file_save = file_path.stem
        phase_path = savedir / f'{file_save}_phase.npy'
        opd_path = savedir / f'{file_save}_OPDs.npy'

        if not overwrite and phase_path.exists() and opd_path.exists():
            print(f'[{i}/{len(files)}] skipping {savedir.name}/{file_path.name} (already done)')
            continue

        print(f'[{i}/{len(files)}] reconstructing {savedir.name}/{file_path.name}')
        try:
            tele = OZITele(tele_path=str(file_path), is_onsky=True)
            tele.reconstruct_all_phase(method='atan', iteration=10, damping=0.5)

            np.save(phase_path, tele.phase)
            np.save(opd_path, tele.OPDs)
        except Exception as e:
            print(f'[{i}/{len(files)}] FAILED {file_path.name}: {e!r}')
        finally:
            tele = None
            gc.collect()
