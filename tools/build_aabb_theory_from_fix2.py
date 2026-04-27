import argparse
import csv
import glob
import os
import subprocess
from collections import defaultdict

import imageio_ffmpeg as imff
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


def decode_audio_mono_f32(path: str, sr: int = 11025) -> tuple[np.ndarray, int]:
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


def find_audio(audio_dir: str, weapon: str) -> str:
    for p in sorted(glob.glob(os.path.join(audio_dir, "*.mp3"))):
        if os.path.splitext(os.path.basename(p))[0] == weapon:
            return p
    raise FileNotFoundError(f"audio not found for weapon={weapon}")


def load_fix2_hits(path: str) -> dict[str, dict[int, float]]:
    by_weapon = defaultdict(dict)
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            by_weapon[row["weapon"]][int(row["idx"])] = float(row["time_s"])
    return by_weapon


def load_ab_labels(path: str) -> dict[str, dict[int, tuple[float, str]]]:
    by_weapon = defaultdict(dict)
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            by_weapon[row["weapon"]][int(row["idx"])] = (float(row["time_s_raw"]), row["pattern_ab"])
    return by_weapon


def main() -> None:
    ap = argparse.ArgumentParser(description="Join fix2 hits with AB labels and build AA/BB theoretical wave plots.")
    ap.add_argument("--fix2-hits", default=".temp/aps_from_1min_fix2/hit_times.csv")
    ap.add_argument("--ab-labels", default=".temp/boxplots_ab_global/global_hit_labels.csv")
    ap.add_argument("--audio-dir", default=".audio/1 min")
    ap.add_argument("--out-root", default=".temp/aps_with_AB")
    ap.add_argument("--sr", type=int, default=11025)
    args = ap.parse_args()

    out_root = args.out_root
    os.makedirs(out_root, exist_ok=True)
    out_plots = os.path.join(out_root, "theoretical_plots")
    os.makedirs(out_plots, exist_ok=True)

    fix2 = load_fix2_hits(args.fix2_hits)
    ab = load_ab_labels(args.ab_labels)

    hits_with_ab_csv = os.path.join(out_root, "hits_with_ab.csv")
    aabb_summary_csv = os.path.join(out_root, "aabb_aps_summary.csv")
    report_txt = os.path.join(out_root, "aabb_theory_report.txt")

    joined_rows = []
    summary_rows = []
    deltas = []

    weapons = sorted(set(fix2.keys()) & set(ab.keys()))
    for weapon in weapons:
        idxs = sorted(set(fix2[weapon].keys()) & set(ab[weapon].keys()))
        if len(idxs) < 3:
            continue

        seq = []
        for idx in idxs:
            t_fix2 = fix2[weapon][idx]
            t_ab, pat = ab[weapon][idx]
            delta = t_ab - t_fix2
            deltas.append(delta)
            seq.append((idx, t_fix2, pat, t_ab, delta))
            joined_rows.append((weapon, idx, t_fix2, pat, t_ab, delta))

        seq.sort(key=lambda x: x[0])
        pair_vals = defaultdict(list)
        for i in range(len(seq) - 1):
            p = seq[i][2] + seq[i + 1][2]
            dt = seq[i + 1][1] - seq[i][1]  # use fix2 time as ground truth
            pair_vals[p].append(dt)

        aa = np.array(pair_vals.get("AA", []), dtype=float)
        bb = np.array(pair_vals.get("BB", []), dtype=float)
        if len(aa) == 0 and len(bb) == 0:
            continue

        mean_aa = float(np.mean(aa)) if len(aa) else float("nan")
        mean_bb = float(np.mean(bb)) if len(bb) else float("nan")
        both = np.concatenate([aa, bb]) if (len(aa) + len(bb)) else np.array([], dtype=float)
        mean_aabb = float(np.mean(both)) if len(both) else float("nan")
        aps_aabb = (1.0 / mean_aabb) if mean_aabb > 0 else float("nan")

        n_hits = len(seq)
        first_t = seq[0][1]
        t_theory = np.array([first_t + i * mean_aabb for i in range(n_hits)], dtype=float) if mean_aabb > 0 else np.array([])

        summary_rows.append(
            (
                weapon,
                n_hits,
                len(aa),
                len(bb),
                mean_aa,
                mean_bb,
                mean_aabb,
                aps_aabb,
            )
        )

        # Plot theoretical wave overlay
        try:
            audio = find_audio(args.audio_dir, weapon)
            x, sr = decode_audio_mono_f32(audio, sr=args.sr)
            t = np.arange(len(x), dtype=np.float64) / sr
            env = gaussian_filter1d(np.abs(x), sigma=max(1, int(sr * 0.0035)))
            env = env / (np.max(env) + 1e-12)
            raw = x / (np.max(np.abs(x)) + 1e-12)

            real_t = np.array([x[1] for x in seq], dtype=float)
            real_ab = [x[2] for x in seq]

            fig = plt.figure(figsize=(16, 5), dpi=130)
            ax = fig.add_subplot(111)
            ax.plot(t, raw, linewidth=0.35, alpha=0.25, color="#334155", label="raw")
            ax.plot(t, env, linewidth=1.0, alpha=0.95, color="#2563eb", label="envelope")

            ax.vlines(real_t, ymin=-1.05, ymax=1.05, linewidth=0.35, alpha=0.25, color="#64748b", label="real hits(fix2)")
            for i, (ht, abv) in enumerate(zip(real_t, real_ab)):
                y = 1.02 if i % 2 == 0 else 0.94
                c = "#0f766e" if abv == "A" else "#7c2d12"
                ax.text(ht, y, abv, fontsize=6.5, color=c, ha="center", va="bottom", rotation=90)

            if len(t_theory):
                ax.vlines(
                    t_theory,
                    ymin=-0.9,
                    ymax=0.9,
                    linewidth=0.7,
                    alpha=0.6,
                    color="#16a34a",
                    label="theory hits (first+n*mean(AA,BB))",
                )

            ax.set_xlim(0, t[-1] if len(t) else 60.0)
            ax.set_ylim(-1.1, 1.12)
            ax.set_xlabel("time (s)")
            ax.set_ylabel("normalized amp")
            ax.set_title(
                f"{weapon} | n={n_hits} | meanAA={mean_aa:.6f} meanBB={mean_bb:.6f} "
                f"meanAABB={mean_aabb:.6f} aps={aps_aabb:.6f}"
            )
            ax.grid(axis="y", alpha=0.2)
            ax.legend(loc="upper right")
            fig.tight_layout()
            fig.savefig(os.path.join(out_plots, f"{weapon}.png"))
            plt.close(fig)
        except Exception:
            # Keep pipeline robust; report still generated.
            pass

    with open(hits_with_ab_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["weapon", "idx", "time_s_fix2", "pattern_ab", "time_s_ab_raw", "delta_ab_minus_fix2_s"])
        for row in joined_rows:
            w.writerow([row[0], row[1], f"{row[2]:.6f}", row[3], f"{row[4]:.6f}", f"{row[5]:+.6f}"])

    with open(aabb_summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["weapon", "n_hits", "n_AA", "n_BB", "mean_AA_s", "mean_BB_s", "mean_AABB_s", "aps_from_AABB"])
        for row in summary_rows:
            w.writerow(
                [
                    row[0],
                    row[1],
                    row[2],
                    row[3],
                    f"{row[4]:.6f}" if row[4] == row[4] else "",
                    f"{row[5]:.6f}" if row[5] == row[5] else "",
                    f"{row[6]:.6f}" if row[6] == row[6] else "",
                    f"{row[7]:.6f}" if row[7] == row[7] else "",
                ]
            )

    with open(report_txt, "w", encoding="utf-8") as f:
        f.write(f"weapons_joined={len(summary_rows)}\n")
        if deltas:
            arr = np.array(deltas, dtype=float)
            f.write(f"ab_minus_fix2_mean_s={arr.mean():+.6f}\n")
            f.write(f"ab_minus_fix2_std_s={arr.std(ddof=0):.6f}\n")
            f.write(f"ab_minus_fix2_p50_s={np.percentile(arr,50):+.6f}\n")
        f.write(f"plots_dir={out_plots}\n")

    print(f"[ok] hits+ab: {hits_with_ab_csv}")
    print(f"[ok] aabb summary: {aabb_summary_csv}")
    print(f"[ok] report: {report_txt}")
    print(f"[ok] plots: {out_plots}")


if __name__ == "__main__":
    main()

