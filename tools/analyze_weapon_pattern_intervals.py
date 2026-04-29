import argparse
import csv
import glob
import os
import subprocess
from dataclasses import dataclass

import imageio_ffmpeg as imff
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


@dataclass
class DetectResult:
    times: np.ndarray
    env: np.ndarray
    sr: int
    peaks: np.ndarray


def decode_audio_mono_f32(path: str, sr: int = 22050) -> tuple[np.ndarray, int]:
    ffmpeg = imff.get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        path,
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-",
    ]
    raw = subprocess.check_output(cmd)
    x = np.frombuffer(raw, dtype=np.float32)
    return x, sr


def detect_peaks_by_expected_hits(env: np.ndarray, sr: int, expected_hits: int) -> np.ndarray:
    best = None
    for dist_ms in (220, 240, 260, 280, 300, 320, 340, 360):
        distance = int(sr * dist_ms / 1000)
        for prom_p in np.linspace(70, 99.5, 180):
            prom = np.percentile(env, prom_p)
            peaks, _ = find_peaks(env, distance=distance, prominence=prom)
            err = abs(len(peaks) - expected_hits)
            cand = (err, abs(dist_ms - 280), abs(prom_p - 90), peaks)
            if best is None or cand < best:
                best = cand
    if best is None:
        raise RuntimeError("peak detection failed")
    return best[-1]


def extract_peak_windows(env: np.ndarray, peaks: np.ndarray, half_window: int) -> np.ndarray:
    windows = []
    for p in peaks:
        lo = max(0, p - half_window)
        hi = min(len(env), p + half_window + 1)
        w = env[lo:hi]
        if len(w) < 2 * half_window + 1:
            pad_left = max(0, half_window - p)
            pad_right = max(0, p + half_window + 1 - len(env))
            w = np.pad(w, (pad_left, pad_right), mode="edge")
        w = w.astype(np.float64)
        w = (w - w.mean()) / (w.std() + 1e-9)
        windows.append(w)
    return np.vstack(windows)


def pca_reduce(x: np.ndarray, out_dim: int = 4) -> np.ndarray:
    xc = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    basis = vt[:out_dim].T
    return xc @ basis


def kmeans2(x: np.ndarray, seed: int = 7, iters: int = 60) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(x)
    if n < 2:
        return np.zeros(n, dtype=int)
    c_idx = rng.choice(n, size=2, replace=False)
    c = x[c_idx].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        d0 = np.sum((x - c[0]) ** 2, axis=1)
        d1 = np.sum((x - c[1]) ** 2, axis=1)
        new_labels = (d1 < d0).astype(int)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for k in (0, 1):
            sel = x[labels == k]
            if len(sel) > 0:
                c[k] = sel.mean(axis=0)
    return labels


def optimize_delta(times: np.ndarray, labels: np.ndarray, lo: float = -0.10, hi: float = 0.10) -> float:
    grid = np.linspace(lo, hi, 4001)
    best = None
    for delta in grid:
        offsets = np.where(labels == 1, delta, 0.0)
        adj = times + offsets
        d = np.diff(adj)
        val = d.std()
        cand = (val, abs(delta), delta)
        if best is None or cand < best:
            best = cand
    return float(best[-1])


def quantiles(x: np.ndarray, qs=(1, 5, 10, 25, 50, 75, 90, 95, 99)) -> list[tuple[int, float]]:
    vals = np.percentile(x, list(qs))
    return list(zip(qs, vals))


def analyze(path: str, expected_hits: int, sr: int = 22050) -> tuple[DetectResult, np.ndarray, float, np.ndarray]:
    x, sr = decode_audio_mono_f32(path, sr=sr)
    env = gaussian_filter1d(np.abs(x), sigma=max(1, int(sr * 0.004)))
    peaks = detect_peaks_by_expected_hits(env, sr, expected_hits)
    times = peaks / sr

    half_window = int(sr * 0.08)
    win = extract_peak_windows(env, peaks, half_window=half_window)
    feat = pca_reduce(win, out_dim=4)
    labels = kmeans2(feat)

    delta = optimize_delta(times, labels, lo=-0.10, hi=0.10)
    adj_times = times + np.where(labels == 1, delta, 0.0)
    return DetectResult(times=times, env=env, sr=sr, peaks=peaks), labels, delta, adj_times


def main() -> None:
    ap = argparse.ArgumentParser(description="Detect A/B hit patterns and optimize pattern-specific peak offsets.")
    ap.add_argument("--weapon", required=True, help="Weapon zh name, e.g. 匕首")
    ap.add_argument("--expected-hits", type=int, required=True, help="Expected number of hits in 1 min sample")
    ap.add_argument("--audio-dir", default=".audio/1 min", help="Directory containing weapon mp3 files")
    ap.add_argument("--out-dir", default=".temp", help="Output directory")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.audio_dir, "*.mp3")))
    target = None
    for f in files:
        if os.path.basename(f).startswith(args.weapon):
            target = f
            break
    if target is None:
        raise FileNotFoundError(f"cannot find mp3 for weapon={args.weapon}")

    det, labels, delta, adj_times = analyze(target, expected_hits=args.expected_hits)

    raw_d = np.diff(det.times)
    adj_d = np.diff(adj_times)
    raw_q = quantiles(raw_d)
    adj_q = quantiles(adj_d)

    os.makedirs(args.out_dir, exist_ok=True)
    stem = args.weapon
    out_csv = os.path.join(args.out_dir, f"{stem}_pattern_hits.csv")
    out_report = os.path.join(args.out_dir, f"{stem}_pattern_report.txt")
    out_q = os.path.join(args.out_dir, f"{stem}_pattern_interval_quantiles.csv")

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["idx", "time_s_raw", "pattern", "offset_s", "time_s_adjusted"])
        for i, t in enumerate(det.times):
            off = delta if labels[i] == 1 else 0.0
            w.writerow([i, f"{t:.6f}", "B" if labels[i] == 1 else "A", f"{off:.6f}", f"{(t + off):.6f}"])

    with open(out_q, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["mode", "percentile", "interval_s", "ratio_to_p50"])
        raw_p50 = np.percentile(raw_d, 50)
        adj_p50 = np.percentile(adj_d, 50)
        for p, v in raw_q:
            w.writerow(["raw", p, f"{v:.6f}", f"{(v / raw_p50):.4f}"])
        for p, v in adj_q:
            w.writerow(["adjusted", p, f"{v:.6f}", f"{(v / adj_p50):.4f}"])

    with open(out_report, "w", encoding="utf-8") as f:
        f.write(f"weapon={args.weapon}\n")
        f.write(f"audio={target}\n")
        f.write(f"hits={len(det.times)} expected={args.expected_hits}\n")
        f.write(f"pattern_A={int(np.sum(labels == 0))} pattern_B={int(np.sum(labels == 1))}\n")
        f.write(f"delta_B_minus_A_s={delta:.6f}\n")
        f.write(f"raw_interval_mean={raw_d.mean():.6f}\n")
        f.write(f"raw_interval_std={raw_d.std():.6f}\n")
        f.write(f"adjusted_interval_mean={adj_d.mean():.6f}\n")
        f.write(f"adjusted_interval_std={adj_d.std():.6f}\n")
        f.write(f"std_improvement={(1 - adj_d.std() / (raw_d.std() + 1e-12)) * 100:.2f}%\n")

    print(f"[ok] report: {out_report}")
    print(f"[ok] hit labels: {out_csv}")
    print(f"[ok] quantiles: {out_q}")


if __name__ == "__main__":
    main()
