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


def load_summary(summary_csv: str) -> list[dict]:
    with open(summary_csv, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_ab_labels(labels_csv: str) -> dict[str, list[tuple[int, float, str]]]:
    by_weapon = defaultdict(list)
    with open(labels_csv, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            weapon = row["weapon"].strip()
            idx = int(row["idx"])
            t = float(row["time_s_raw"])
            ab = row["pattern_ab"].strip()
            by_weapon[weapon].append((idx, t, ab))
    for w in by_weapon:
        by_weapon[w].sort(key=lambda x: x[0])
    return by_weapon


def find_audio(audio_dir: str, weapon: str) -> str:
    for p in sorted(glob.glob(os.path.join(audio_dir, "*.mp3"))):
        if os.path.splitext(os.path.basename(p))[0] == weapon:
            return p
    raise FileNotFoundError(f"audio not found for weapon={weapon}")


def plot_weapon_with_ab(out_png: str, weapon: str, x: np.ndarray, sr: int, hits: list[tuple[int, float, str]], aps: float) -> None:
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    t = np.arange(len(x), dtype=np.float64) / sr
    env = gaussian_filter1d(np.abs(x), sigma=max(1, int(sr * 0.0035)))
    env = env / (np.max(env) + 1e-12)
    raw = x / (np.max(np.abs(x)) + 1e-12)

    hit_t = np.array([h[1] for h in hits], dtype=np.float64)
    hit_ab = [h[2] for h in hits]

    fig = plt.figure(figsize=(16, 5), dpi=130)
    ax = fig.add_subplot(111)
    ax.plot(t, raw, linewidth=0.35, alpha=0.28, color="#1f2937", label="raw")
    ax.plot(t, env, linewidth=1.0, alpha=0.95, color="#2563eb", label="envelope")

    if len(hit_t):
        ax.vlines(hit_t, ymin=-1.05, ymax=1.05, linewidth=0.45, alpha=0.5, color="#ef4444", label="hits")
        for i, (ht, ab) in enumerate(zip(hit_t, hit_ab)):
            y = 1.02 if i % 2 == 0 else 0.94
            color = "#0f766e" if ab == "A" else "#7c2d12"
            ax.text(ht, y, ab, fontsize=6.5, color=color, ha="center", va="bottom", rotation=90)

    ax.set_xlim(0, t[-1] if len(t) else 60.0)
    ax.set_ylim(-1.1, 1.12)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("normalized amp")
    ax.set_title(f"{weapon} | hits={len(hits)} | aps(first-last)={aps:.6f} | labels=AB")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build APS wave plots with A/B label on each hit node.")
    ap.add_argument("--summary-csv", default=".temp/aps_from_1min_fix2/aps_summary.csv")
    ap.add_argument("--labels-csv", default=".temp/boxplots_ab_global/global_hit_labels.csv")
    ap.add_argument("--audio-dir", default=".audio/1 min")
    ap.add_argument("--out-root", default=".temp/aps_with_AB")
    ap.add_argument("--sr", type=int, default=11025)
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    out_plots = os.path.join(args.out_root, "plots")
    os.makedirs(out_plots, exist_ok=True)

    summary = load_summary(args.summary_csv)
    labels = load_ab_labels(args.labels_csv)

    out_csv = os.path.join(args.out_root, "aps_with_ab_summary.csv")
    rows = []
    for r in summary:
        weapon = r["weapon"].strip()
        if weapon not in labels:
            continue
        audio = find_audio(args.audio_dir, weapon)
        x, sr = decode_audio_mono_f32(audio, sr=args.sr)
        aps = float(r["aps_first_last"]) if r["aps_first_last"] else float("nan")
        hits = labels[weapon]
        a_count = sum(1 for _, _, ab in hits if ab == "A")
        b_count = sum(1 for _, _, ab in hits if ab == "B")
        out_png = os.path.join(out_plots, f"{weapon}.png")
        plot_weapon_with_ab(out_png, weapon, x, sr, hits, aps)
        rows.append(
            {
                "weapon": weapon,
                "hits": len(hits),
                "A_count": a_count,
                "B_count": b_count,
                "aps_first_last": aps,
                "plot": out_png,
            }
        )

    rows.sort(key=lambda x: x["weapon"])
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["weapon", "hits", "A_count", "B_count", "aps_first_last", "plot"])
        for r in rows:
            w.writerow([r["weapon"], r["hits"], r["A_count"], r["B_count"], f"{r['aps_first_last']:.6f}", r["plot"]])

    print(f"[ok] plots: {out_plots}")
    print(f"[ok] summary: {out_csv}")


if __name__ == "__main__":
    main()

