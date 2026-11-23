"""
Make movies of the generated 2nd stage atmospheres (atm_OPDs_2nd) for several tboil values.

Only the first n_frames frames are read: the npz is decompressed as a stream, so the
150000-frame files are never fully loaded. One movie per tboil plus a side-by-side grid,
all with the same colour scale (piston removed per frame, in nm).

ffmpeg: uses imageio_ffmpeg if installed, otherwise the FFMPEG_EXE environment variable.
"""

import os
import zipfile
from multiprocessing import Pool

import numpy as np
import matplotlib
from PIL import Image, ImageDraw, ImageFont
import subprocess

base_dir = os.path.dirname(os.path.abspath(__file__))
atm_name = "atm_OPDs_2nd_r0_0.070_V0_3.002_L0_30.000_tboil_{:.3f}_multi_layer.npz"
atm_files = {
    0.002: f"{base_dir}/generated_atm_2nd_stage/{atm_name.format(0.002)}",
    0.02:  f"{base_dir}/generated_atm_2nd_stage/{atm_name.format(0.02)}",
    0.05:  f"{base_dir}/generated_atm_2nd_stage/{atm_name.format(0.05)}",
    0:     f"{base_dir}/generated_atm_2nd_stage_150000/{atm_name.format(0)}",
    0.1:   f"{base_dir}/generated_atm_2nd_stage_150000/{atm_name.format(0.1)}",
    0.5:   f"{base_dir}/generated_atm_2nd_stage_150000/{atm_name.format(0.5)}",
    1:     f"{base_dir}/generated_atm_2nd_stage_150000/{atm_name.format(1)}",
    2:     f"{base_dir}/generated_atm_2nd_stage_150000/{atm_name.format(2)}",
    5:     f"{base_dir}/generated_atm_2nd_stage_150000/{atm_name.format(5)}",
}
# side-by-side movies: file tag -> tboils in order of decreasing boiling (no boiling, tboil = 0, last)
comparison_sets = {
    "tboil":       [0.02, 0.05, 0.1, 1, 5, 0],
    "tboil_0.1-5": [0.1, 0.5, 1, 2, 5, 0],
}

n_frames     = 15000
samplingTime = 1/400   # param['samplingTime_2nd']
fps          = 100
upscale      = 4
cmap         = matplotlib.colormaps["viridis"]
vmax_pct     = 99.9    # colour range = +- this percentile of |OPD| over all movies

save_dir = f"{base_dir}/atm_movies"


def get_ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return os.environ["FFMPEG_EXE"]


def load_first_frames(path, n):
    """First n frames of the OPD cube [m], from an OOPAO npz (streamed) or a plain npy (FATMOSS, memory-mapped)."""
    if path.endswith(".npy"):
        cube = np.load(path, mmap_mode="r")
        assert cube.shape[0] >= n, (path, cube.shape)
        opd, meta = np.array(cube[:n], dtype=np.float32), {}
    else:
        with zipfile.ZipFile(path) as z, z.open("atm_OPDs_2nd.npy") as f:
            version = np.lib.format.read_magic(f)
            read_header = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
            shape, fortran, dtype = read_header(f)
            assert not fortran and shape[0] >= n, (path, shape)
            frame_size = int(np.prod(shape[1:]))
            buf = f.read(n * frame_size * dtype.itemsize)
            meta = {k: np.load(z.open(f"{k}.npy")) for k in ["r0", "L0", "windSpeed"]}
        opd = np.frombuffer(buf, dtype=dtype).reshape((n,) + tuple(shape[1:])).astype(np.float32)
    opd = (opd - opd.mean(axis=(1, 2), keepdims=True)) * 1e9   # remove piston, m -> nm
    return opd, meta


def colorize(frame, vmax):
    rgb = (cmap((frame + vmax) / (2 * vmax))[..., :3] * 255).astype(np.uint8)
    return np.repeat(np.repeat(rgb, upscale, axis=0), upscale, axis=1)


def colorbar(height, vmax, font):
    width, bar_w = 90, 16
    grad = np.linspace(1, 0, height)[:, None].repeat(bar_w, axis=1)
    img = Image.new("RGB", (width, height), "white")
    img.paste(Image.fromarray((cmap(grad)[..., :3] * 255).astype(np.uint8)), (6, 0))
    d = ImageDraw.Draw(img)
    for v in np.linspace(-vmax, vmax, 5):
        y = int((vmax - v) / (2 * vmax) * (height - 1))
        d.line([(6 + bar_w, y), (6 + bar_w + 4, y)], fill="black")
        d.text((6 + bar_w + 7, min(max(y - 7, 0), height - 15)), f"{v:.0f}", fill="black", font=font)
    d.text((6 + bar_w + 7, height // 2 + 20), "nm", fill="black", font=font)
    return img


def tboil_label(tboil):
    return f"tboil = {tboil:g} s"


def panel(frame, i, label, vmax, font):
    img = Image.fromarray(colorize(frame, vmax))
    out = Image.new("RGB", (img.width, img.height + 44), "white")
    out.paste(img, (0, 44))
    d = ImageDraw.Draw(out)
    d.text((4, 2), label, fill="black", font=font)
    d.text((4, 22), f"frame {i:5d}   t = {i * samplingTime:6.3f} s", fill="black", font=font)
    return out


def open_writer(path, w, h):
    return subprocess.Popen(
        [get_ffmpeg(), "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", path],
        stdin=subprocess.PIPE)


def make_movie(frames_list, labels, vmax, path, ncols):
    font = ImageFont.load_default(size=16)
    side = frames_list[0].shape[1] * upscale
    cbar = colorbar(side, vmax, font)
    pw, ph = side, side + 44
    nrows = int(np.ceil(len(labels) / ncols))
    w, h = ncols * pw + cbar.width, nrows * ph
    w, h = w + w % 2, h + h % 2   # yuv420p needs even dimensions
    proc = open_writer(path, w, h)
    for i in range(n_frames):
        canvas = Image.new("RGB", (w, h), "white")
        for k, (frames, label) in enumerate(zip(frames_list, labels)):
            canvas.paste(panel(frames[i], i, label, vmax, font), ((k % ncols) * pw, (k // ncols) * ph))
        canvas.paste(cbar, (ncols * pw, h - side))
        proc.stdin.write(canvas.tobytes())
    proc.stdin.close()
    proc.wait()
    print("saved", path)


def _movie_job(args):
    make_movie(*args)


if __name__ == "__main__":
    os.makedirs(save_dir, exist_ok=True)

    tboils = list(atm_files)
    with Pool(len(tboils)) as pool:
        loaded = pool.starmap(load_first_frames, [(atm_files[t], n_frames) for t in tboils])
    opds = [o for o, _ in loaded]
    for t, (o, m) in zip(tboils, loaded):
        print(f"tboil {t:g}: r0 {m['r0']}, L0 {m['L0']}, windSpeed {m['windSpeed']}, "
              f"spatial std {o.std(axis=(1, 2)).mean():.1f} nm")

    rng = np.random.default_rng(0)
    sample = np.concatenate([o[rng.choice(n_frames, 500, replace=False)].ravel() for o in opds])
    vmax = float(np.percentile(np.abs(sample), vmax_pct))
    print(f"colour range: +-{vmax:.1f} nm")

    jobs = [([o], [tboil_label(t)], vmax, f"{save_dir}/atm_2nd_tboil_{t:.3f}_{n_frames}frames.mp4", 1)
            for o, t in zip(opds, tboils)]
    opd_of = dict(zip(tboils, opds))
    for tag, tboil_set in comparison_sets.items():
        jobs.append(([opd_of[t] for t in tboil_set], [tboil_label(t) for t in tboil_set], vmax,
                     f"{save_dir}/atm_2nd_{tag}_comparison_{n_frames}frames.mp4", 3))
    with Pool(len(jobs)) as pool:
        pool.map(_movie_job, jobs)
