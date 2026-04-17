import argparse
import csv
import glob
import json
import os
import subprocess
from dataclasses import dataclass

import imageio_ffmpeg as imff
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


@dataclass
class DetectParams:
    sigma_ms: float
    distance_ms: float
    prominence_pct: float


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


def build_env(x: np.ndarray, sr: int, sigma_ms: float) -> np.ndarray:
    sigma = max(1, int(sr * sigma_ms / 1000.0))
    return gaussian_filter1d(np.abs(x), sigma=sigma)


def detect_hits(env: np.ndarray, sr: int, distance_ms: float, prominence_pct: float) -> tuple[np.ndarray, np.ndarray]:
    distance = max(1, int(sr * distance_ms / 1000.0))
    prom = float(np.percentile(env, prominence_pct))
    peaks, props = find_peaks(env, distance=distance, prominence=prom)
    prominences = props.get("prominences", np.zeros(len(peaks), dtype=np.float64))
    return peaks, prominences


def prune_double_hits(peaks: np.ndarray, prominences: np.ndarray, sr: int) -> np.ndarray:
    if len(peaks) < 3:
        return peaks
    p = peaks.copy()
    pr = prominences.copy()
    changed = True
    while changed and len(p) >= 3:
        changed = False
        d = np.diff(p) / sr
        med = float(np.median(d))
        if med <= 0:
            break
        # If two peaks are too close, drop lower-prominence one.
        too_close = np.where(d < (0.45 * med))[0]
        if len(too_close) == 0:
            break
        i = int(too_close[0])
        if pr[i] <= pr[i + 1]:
            drop = i
        else:
            drop = i + 1
        p = np.delete(p, drop)
        pr = np.delete(pr, drop)
        changed = True
    return p


def intervals_stats(times: np.ndarray) -> tuple[float, float]:
    if len(times) < 2:
        return float("nan"), float("nan")
    d = np.diff(times)
    return float(d.mean()), float(d.std(ddof=0))


def aps_first_last(times: np.ndarray) -> float:
    if len(times) < 2:
        return float("nan")
    span = float(times[-1] - times[0])
    if span <= 0:
        return float("nan")
    return float((len(times) - 1) / span)


def calibrate_params(beat_files: list[str], sr: int) -> DetectParams:
    sigma_grid = [2.0, 3.0, 4.0, 5.0]
    dist_grid = [220, 240, 260, 280, 300, 320, 340]
    prom_grid = [85.0, 87.5, 90.0, 92.5, 95.0, 96.5, 98.0]
    target_hits = 60

    best = None
    decoded = []
    for f in beat_files:
        x, _ = decode_audio_mono_f32(f, sr=sr)
        decoded.append(x)

    for sigma_ms in sigma_grid:
        envs = [build_env(x, sr, sigma_ms) for x in decoded]
        for distance_ms in dist_grid:
            for prom_pct in prom_grid:
                total_abs_err = 0.0
                total_cv = 0.0
                valid = 0
                for env in envs:
                    peaks, _ = detect_hits(env, sr, distance_ms, prom_pct)
                    n = len(peaks)
                    total_abs_err += abs(n - target_hits)
                    if n >= 3:
                        t = peaks / sr
                        d = np.diff(t)
                        m = float(d.mean())
                        s = float(d.std(ddof=0))
                        if m > 0:
                            total_cv += s / m
                            valid += 1
                avg_cv = (total_cv / valid) if valid > 0 else 999.0
                # Prioritize exact hit count on beat, then interval regularity.
                score = total_abs_err * 10.0 + avg_cv
                cand = (score, total_abs_err, avg_cv, sigma_ms, distance_ms, prom_pct)
                if best is None or cand < best:
                    best = cand

    assert best is not None
    return DetectParams(sigma_ms=best[3], distance_ms=best[4], prominence_pct=best[5])


def estimate_period_by_autocorr(env: np.ndarray, sr: int, lo_s: float = 0.45, hi_s: float = 2.4) -> float:
    # Downsample for fast autocorr search (period scale is large enough).
    ds = 32
    x = env[::ds].astype(np.float64)
    sr_ds = sr / ds
    x = x - x.mean()
    if np.allclose(x.std(), 0.0):
        return 1.0
    x = x / (x.std() + 1e-12)
    n = len(x)
    nfft = 1 << int(np.ceil(np.log2(max(2, 2 * n - 1))))
    fx = np.fft.rfft(x, n=nfft)
    ac = np.fft.irfft(fx * np.conj(fx), n=nfft)[:n]
    lo = max(1, int(sr_ds * lo_s))
    hi = min(len(ac) - 1, int(sr_ds * hi_s))
    if hi <= lo:
        return 1.0
    lag = lo + int(np.argmax(ac[lo : hi + 1]))
    return float(lag / sr_ds)


def detect_weapon_hits_optimized(
    x: np.ndarray, sr: int, base_sigma_ms: float, expected_aps: float | None = None
) -> tuple[np.ndarray, np.ndarray, DetectParams]:
    # Base envelope for period estimation.
    base_env = build_env(x, sr, base_sigma_ms)
    est_period = estimate_period_by_autocorr(base_env, sr, lo_s=0.45, hi_s=2.4)
    est_hits = max(20.0, min(140.0, 60.0 / max(est_period, 1e-6)))

    sigma_grid = [max(1.5, base_sigma_ms - 0.8), base_sigma_ms, base_sigma_ms + 0.8]
    dist_grid = np.linspace(est_period * 0.5, est_period * 0.82, 6)
    prom_grid = [84.0, 88.0, 91.0, 94.0, 96.0, 98.0]

    # If we know expected APS, include an expected-period search band.
    if expected_aps is not None and expected_aps > 0:
        exp_period = 1.0 / expected_aps
        dist_grid = np.unique(
            np.concatenate(
                [
                    dist_grid,
                    np.linspace(exp_period * 0.45, exp_period * 0.82, 7),
                ]
            )
        )
        # High-APS weapons often have weaker alternating peaks; allow lower prominence.
        if expected_aps >= 1.45:
            sigma_grid = np.unique(np.concatenate([sigma_grid, [1.0, 1.2, 1.6]])).tolist()
            prom_grid = [45.0, 50.0, 55.0, 60.0, 65.0, 72.0, 80.0, 88.0, 94.0]

    best = None
    best_tuple = None
    for sigma_ms in sigma_grid:
        env = build_env(x, sr, sigma_ms)
        for dist_s in dist_grid:
            distance_ms = float(dist_s * 1000.0)
            for prom_pct in prom_grid:
                peaks, prominences = detect_hits(env, sr, distance_ms, prom_pct)
                peaks = prune_double_hits(peaks, prominences, sr)
                if len(peaks) < 5:
                    continue
                t = peaks / sr
                d = np.diff(t)
                m = float(d.mean())
                s = float(d.std(ddof=0))
                cv = (s / m) if m > 0 else 999.0
                span = float(t[-1] - t[0])
                hits = float(len(peaks))
                # Objective:
                # 1) Low interval CV (core quality)
                # 2) Span should cover ~1 minute window
                # 3) Hits should be close to rough estimate from autocorr
                aps = (hits - 1.0) / span if span > 0 else 0.0
                aps_penalty = 0.0
                if expected_aps is not None and expected_aps > 0:
                    aps_penalty = abs(aps - expected_aps) / expected_aps
                score = cv * 3.5 + abs(span - 59.0) * 0.08 + abs(hits - est_hits) * 0.015 + aps_penalty * 4.0
                cand = (score, cv, abs(span - 59.0), abs(hits - est_hits), sigma_ms, distance_ms, prom_pct)
                if best is None or cand < best:
                    best = cand
                    best_tuple = (env, peaks, DetectParams(sigma_ms=sigma_ms, distance_ms=distance_ms, prominence_pct=prom_pct))

    if best_tuple is None:
        # Fallback to a safe default.
        env = build_env(x, sr, base_sigma_ms)
        peaks, prominences = detect_hits(env, sr, distance_ms=300.0, prominence_pct=95.0)
        peaks = prune_double_hits(peaks, prominences, sr)
        return env, peaks, DetectParams(sigma_ms=base_sigma_ms, distance_ms=300.0, prominence_pct=95.0)

    return best_tuple


def plot_wave(
    out_png: str,
    weapon: str,
    x: np.ndarray,
    sr: int,
    env: np.ndarray,
    peaks: np.ndarray,
    aps: float,
    n_hits: int,
) -> None:
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    t = np.arange(len(x), dtype=np.float64) / sr
    peak_t = peaks / sr
    y_env = env / (np.max(env) + 1e-12)
    y_raw = x / (np.max(np.abs(x)) + 1e-12)

    fig = plt.figure(figsize=(14, 4.8), dpi=120)
    ax = fig.add_subplot(111)
    ax.plot(t, y_raw, linewidth=0.4, alpha=0.35, label="raw")
    ax.plot(t, y_env, linewidth=1.0, alpha=0.95, label="envelope")
    if len(peak_t) > 0:
        ax.vlines(peak_t, ymin=-1.05, ymax=1.05, linewidth=0.5, alpha=0.6, color="tab:red", label="hits")
    ax.set_xlim(0, t[-1] if len(t) else 60.0)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("normalized amp")
    ax.set_title(f"{weapon} | hits={n_hits} | aps(first-last)={aps:.6f}")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute weapon APS from 1-min audio by first-last rule and export wave plots.")
    ap.add_argument("--audio-dir", default=".audio/1 min", help="Directory with weapon mp3 files")
    ap.add_argument("--beat-dir", default=".audio/1 min beat", help="Directory with beat reference mp3 files")
    ap.add_argument("--out-dir", default=".temp/aps_from_1min", help="Output directory")
    ap.add_argument("--sr", type=int, default=11025, help="Decode sample rate")
    args = ap.parse_args()

    weapon_files = sorted(glob.glob(os.path.join(args.audio_dir, "*.mp3")))
    beat_files = sorted(glob.glob(os.path.join(args.beat_dir, "*.mp3")))
    if not weapon_files:
        raise FileNotFoundError(f"no weapon mp3 under {args.audio_dir}")
    if not beat_files:
        raise FileNotFoundError(f"no beat mp3 under {args.beat_dir}")

    os.makedirs(args.out_dir, exist_ok=True)
    plots_dir = os.path.join(args.out_dir, "wave_plots")
    os.makedirs(plots_dir, exist_ok=True)

    params = calibrate_params(beat_files, sr=args.sr)

    speed_by_name: dict[str, float] = {}
    weapon_json = os.path.join("docs", "dataset", "raw", "weapon.json")
    if os.path.exists(weapon_json):
        try:
            obj = json.loads(open(weapon_json, "r", encoding="utf-8").read())
            for row in obj.get("datas", []):
                name = str(row.get("Name", "")).strip()
                if not name:
                    continue
                sp = row.get("Speed", None)
                try:
                    speed_by_name[name] = float(sp)
                except Exception:
                    continue
        except Exception:
            speed_by_name = {}

    summary_path = os.path.join(args.out_dir, "aps_summary.csv")
    hits_path = os.path.join(args.out_dir, "hit_times.csv")
    report_path = os.path.join(args.out_dir, "aps_report.txt")

    summary_rows = []
    hit_rows = []

    per_weapon_params = []
    for wf in weapon_files:
        weapon = os.path.splitext(os.path.basename(wf))[0]
        x, sr = decode_audio_mono_f32(wf, sr=args.sr)
        expected_aps = speed_by_name.get(weapon)
        env, peaks, used_params = detect_weapon_hits_optimized(
            x, sr, base_sigma_ms=params.sigma_ms, expected_aps=expected_aps
        )
        times = peaks / sr
        aps = aps_first_last(times)
        mean_itv, std_itv = intervals_stats(times)
        cv = (std_itv / mean_itv) if mean_itv and not np.isnan(mean_itv) and mean_itv > 0 else float("nan")
        per_weapon_params.append((weapon, used_params))

        plot_path = os.path.join(plots_dir, f"{weapon}.png")
        plot_wave(plot_path, weapon, x, sr, env, peaks, aps, len(times))

        first_s = float(times[0]) if len(times) else float("nan")
        last_s = float(times[-1]) if len(times) else float("nan")
        summary_rows.append(
            {
                "weapon": weapon,
                "audio": wf,
                "hits": len(times),
                "first_s": first_s,
                "last_s": last_s,
                "aps_first_last": aps,
                "interval_mean_s": mean_itv,
                "interval_std_s": std_itv,
                "interval_cv": cv,
                "wave_plot": plot_path,
            }
        )

        for i, t in enumerate(times):
            hit_rows.append({"weapon": weapon, "idx": i, "time_s": float(t)})

    summary_rows.sort(key=lambda r: r["weapon"])
    hit_rows.sort(key=lambda r: (r["weapon"], r["idx"]))

    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "weapon",
                "hits",
                "first_s",
                "last_s",
                "aps_first_last",
                "interval_mean_s",
                "interval_std_s",
                "interval_cv",
                "audio",
                "wave_plot",
            ]
        )
        for r in summary_rows:
            w.writerow(
                [
                    r["weapon"],
                    r["hits"],
                    f"{r['first_s']:.6f}" if not np.isnan(r["first_s"]) else "",
                    f"{r['last_s']:.6f}" if not np.isnan(r["last_s"]) else "",
                    f"{r['aps_first_last']:.6f}" if not np.isnan(r["aps_first_last"]) else "",
                    f"{r['interval_mean_s']:.6f}" if not np.isnan(r["interval_mean_s"]) else "",
                    f"{r['interval_std_s']:.6f}" if not np.isnan(r["interval_std_s"]) else "",
                    f"{r['interval_cv']:.6f}" if not np.isnan(r["interval_cv"]) else "",
                    r["audio"],
                    r["wave_plot"],
                ]
            )

    with open(hits_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["weapon", "idx", "time_s"])
        for r in hit_rows:
            w.writerow([r["weapon"], r["idx"], f"{r['time_s']:.6f}"])

    # Anomaly hints: same weapon APS should be stable; high CV likely hit-definition issue.
    bad = [r for r in summary_rows if not np.isnan(r["interval_cv"]) and r["interval_cv"] > 0.20]
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("APS from 1-min audio\n")
        f.write(f"weapons={len(summary_rows)}\n")
        f.write(
            f"calibrated_params: sigma_ms={params.sigma_ms}, distance_ms={params.distance_ms}, prominence_pct={params.prominence_pct}\n"
        )
        f.write(f"high_cv_count(>0.20)={len(bad)}\n")
        if bad:
            f.write("high_cv_weapons=" + ",".join(r["weapon"] for r in bad) + "\n")
        f.write("per_weapon_params:\n")
        for weapon, p in per_weapon_params:
            f.write(
                f"- {weapon}: sigma_ms={p.sigma_ms:.2f}, distance_ms={p.distance_ms:.2f}, prominence_pct={p.prominence_pct:.2f}\n"
            )
        f.write(f"summary_csv={summary_path}\n")
        f.write(f"hit_times_csv={hits_path}\n")
        f.write(f"wave_plots_dir={plots_dir}\n")

    print(f"[ok] summary: {summary_path}")
    print(f"[ok] hits: {hits_path}")
    print(f"[ok] report: {report_path}")
    print(f"[ok] wave plots: {plots_dir}")


if __name__ == "__main__":
    main()
    # Optional APS prior from weapon dataset.
    speed_by_name: dict[str, float] = {}
    weapon_json = os.path.join("docs", "dataset", "raw", "weapon.json")
    if os.path.exists(weapon_json):
        try:
            obj = json.loads(open(weapon_json, "r", encoding="utf-8").read())
            for row in obj.get("datas", []):
                name = str(row.get("Name", "")).strip()
                if not name:
                    continue
                sp = row.get("Speed", None)
                try:
                    speed_by_name[name] = float(sp)
                except Exception:
                    continue
        except Exception:
            pass
