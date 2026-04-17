import csv
import glob
import os
import subprocess
from collections import defaultdict

import imageio_ffmpeg as imff
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


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
    return np.frombuffer(raw, dtype=np.float32), sr


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
    return best[-1]


def extract_windows(env: np.ndarray, peaks: np.ndarray, half_window: int) -> np.ndarray:
    out = []
    for p in peaks:
        lo = max(0, p - half_window)
        hi = min(len(env), p + half_window + 1)
        w = env[lo:hi]
        if len(w) < 2 * half_window + 1:
            pad_left = max(0, half_window - p)
            pad_right = max(0, p + half_window + 1 - len(env))
            w = np.pad(w, (pad_left, pad_right), mode="edge")
        w = w.astype(np.float64)
        out.append((w - w.mean()) / (w.std() + 1e-9))
    return np.vstack(out)


def pca_reduce(x: np.ndarray, out_dim: int = 6) -> np.ndarray:
    xc = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    return xc @ vt[:out_dim].T


def kmeans2(x: np.ndarray, seed: int = 7, iters: int = 80) -> np.ndarray:
    rng = np.random.default_rng(seed)
    c = x[rng.choice(len(x), size=2, replace=False)].copy()
    labels = np.zeros(len(x), dtype=int)
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


def load_expected_hits(path: str) -> list[tuple[str, int]]:
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append((row["weapon"], int(row["hits"])))
    return rows


def match_audio_file(audio_dir: str, weapon_name: str) -> str:
    cands = sorted(glob.glob(os.path.join(audio_dir, "*.mp3")))
    alias = {
        "戰爭軍刀": "刺客彎刀",
    }
    lookup_names = [weapon_name]
    if weapon_name in alias:
        lookup_names.append(alias[weapon_name])
    for p in cands:
        base = os.path.basename(p)
        for name in lookup_names:
            if base.startswith(name):
                return p
    raise FileNotFoundError(f"audio not found for weapon={weapon_name}")


def main() -> None:
    summary_csv = ".temp/weapon_waveforms_summary.csv"
    audio_dir = ".audio/1 min"
    out_root = ".temp/boxplots_ab_global"
    out_plots = os.path.join(out_root, "plots")
    os.makedirs(out_plots, exist_ok=True)

    weapons = load_expected_hits(summary_csv)

    per_weapon = {}
    all_windows = []
    all_meta = []

    missing_weapons = []
    for weapon, expected_hits in weapons:
        try:
            audio = match_audio_file(audio_dir, weapon)
        except FileNotFoundError:
            missing_weapons.append(weapon)
            continue
        x, sr = decode_audio_mono_f32(audio, sr=22050)
        env = gaussian_filter1d(np.abs(x), sigma=max(1, int(sr * 0.004)))
        peaks = detect_peaks_by_expected_hits(env, sr, expected_hits)
        times = peaks / sr
        windows = extract_windows(env, peaks, half_window=int(sr * 0.08))
        per_weapon[weapon] = {"audio": audio, "times": times, "windows": windows, "expected_hits": expected_hits}

        for i in range(len(times)):
            all_windows.append(windows[i])
            all_meta.append((weapon, i))

    if len(all_windows) == 0:
        raise RuntimeError("no audio matched; cannot build global A/B")
    all_windows_np = np.vstack(all_windows)
    feat = pca_reduce(all_windows_np, out_dim=6)
    labels_raw = kmeans2(feat)

    # Semantic alignment for global A/B:
    # A = cluster whose peak position is earlier in the local window, B = later.
    peak_pos = np.argmax(all_windows_np, axis=1)
    m0 = float(np.mean(peak_pos[labels_raw == 0]))
    m1 = float(np.mean(peak_pos[labels_raw == 1]))
    if m0 <= m1:
        to_ab = {0: "A", 1: "B"}
    else:
        to_ab = {0: "B", 1: "A"}

    weapon_labels = defaultdict(dict)
    for (weapon, i), lab in zip(all_meta, labels_raw):
        weapon_labels[weapon][i] = to_ab[int(lab)]

    summary_rows = []
    global_labels_csv = os.path.join(out_root, "global_hit_labels.csv")
    with open(global_labels_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["weapon", "idx", "time_s_raw", "pattern_ab"])
        for weapon, _ in weapons:
            if weapon not in per_weapon:
                continue
            times = per_weapon[weapon]["times"]
            labels = weapon_labels[weapon]
            for i, t in enumerate(times):
                w.writerow([weapon, i, f"{t:.6f}", labels[i]])

    interval_quantiles_csv = os.path.join(out_root, "interval_quantiles_by_weapon_pair.csv")
    with open(interval_quantiles_csv, "w", encoding="utf-8-sig", newline="") as fq:
        wq = csv.writer(fq)
        wq.writerow(["weapon", "pair_type", "n", "mean_s", "std_s", "percentile", "interval_s", "ratio_to_p50"])
        q_levels = [1, 5, 10, 25, 50, 75, 90, 95, 99]

        for weapon, _ in weapons:
            if weapon not in per_weapon:
                continue
            times = per_weapon[weapon]["times"]
            labels = [weapon_labels[weapon][i] for i in range(len(times))]

            by = defaultdict(list)
            for i in range(len(times) - 1):
                pair = labels[i] + labels[i + 1]
                by[pair].append(times[i + 1] - times[i])

            plot_data = []
            plot_order = []
            for pair in ("AA", "AB", "BA", "BB"):
                arr = np.array(by.get(pair, []), dtype=float)
                if len(arr) == 0:
                    continue
                plot_order.append(pair)
                plot_data.append(arr)
                p50 = np.percentile(arr, 50)
                qs = np.percentile(arr, q_levels)
                for lv, qv in zip(q_levels, qs):
                    wq.writerow([weapon, pair, len(arr), f"{arr.mean():.6f}", f"{arr.std():.6f}", lv, f"{qv:.6f}", f"{(qv/p50):.4f}"])
                summary_rows.append([weapon, pair, len(arr), float(arr.mean()), float(arr.std()), float(p50)])

            if len(plot_data) > 0:
                plt.figure(figsize=(7.5, 5), dpi=170)
                bp = plt.boxplot(plot_data, tick_labels=plot_order, patch_artist=True, showmeans=True, widths=0.6)
                palette = {"AA": "#4e79a7", "AB": "#e15759", "BA": "#59a14f", "BB": "#b07aa1"}
                for patch, pair in zip(bp["boxes"], plot_order):
                    patch.set_facecolor(palette[pair])
                    patch.set_alpha(0.45)
                for med in bp["medians"]:
                    med.set_color("#111111")
                    med.set_linewidth(2)
                for mean in bp["means"]:
                    mean.set_marker("o")
                    mean.set_markerfacecolor("#111111")
                    mean.set_markeredgecolor("#111111")
                    mean.set_markersize(4)
                plt.title(f"{weapon} Interval by Pair Type (Global A/B, Raw)")
                plt.xlabel("Pair Type")
                plt.ylabel("Interval (s)")
                plt.grid(axis="y", alpha=0.25)
                plt.tight_layout()
                plt.savefig(os.path.join(out_plots, f"{weapon}.png"))
                plt.close()

    summary_csv_out = os.path.join(out_root, "pair_stats_summary.csv")
    with open(summary_csv_out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["weapon", "pair_type", "n", "mean_s", "std_s", "p50_s"])
        for row in summary_rows:
            weapon, pair, n, mean_s, std_s, p50 = row
            w.writerow([weapon, pair, n, f"{mean_s:.6f}", f"{std_s:.6f}", f"{p50:.6f}"])

    report = os.path.join(out_root, "global_ab_report.txt")
    with open(report, "w", encoding="utf-8") as f:
        f.write("global A/B definition:\n")
        f.write("- A = earlier local peak position cluster\n")
        f.write("- B = later local peak position cluster\n")
        f.write(f"cluster0_peakpos_mean={m0:.3f}\n")
        f.write(f"cluster1_peakpos_mean={m1:.3f}\n")
        f.write(f"weapons_total={len(weapons)}\n")
        f.write(f"weapons_processed={len(per_weapon)}\n")
        f.write(f"weapons_missing_audio={len(missing_weapons)}\n")
        if missing_weapons:
            f.write("missing_list=" + ",".join(missing_weapons) + "\n")
        f.write(f"plots_dir={out_plots}\n")

    print(f"[ok] plots: {out_plots}")
    print(f"[ok] labels: {global_labels_csv}")
    print(f"[ok] quantiles: {interval_quantiles_csv}")
    print(f"[ok] summary: {summary_csv_out}")
    print(f"[ok] report: {report}")


if __name__ == "__main__":
    main()
