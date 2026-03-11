#!/usr/bin/env python3
"""
DATA 37100 Final Project — Analysis Script
Generates all figures and prints the summary table.
Run from repo root:
  python src/run_analysis.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

GAN_OUT  = Path("./untrack/outputs/final/gan")
DIFF_OUT = Path("./untrack/outputs/final/diffusion")
FIG_OUT  = Path("./untrack/outputs/final/figures")
FIG_OUT.mkdir(parents=True, exist_ok=True)

LRS = [0.0001, 0.0002, 0.0004]
DSTEPS = [1, 2]

np.random.seed(42)


# ── helpers ──────────────────────────────────────────────────────────────────

def load_log(run_dir: Path):
    rows = []
    with open(run_dir / "train_log.csv", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def smooth(vals, w=20):
    out = []
    for i in range(len(vals)):
        start = max(0, i - w + 1)
        out.append(sum(vals[start : i + 1]) / (i - start + 1))
    return out


def run_dir_for(lr, d):
    return GAN_OUT / f"ds-fashionmnist_ep-1_bs-128_lr-{lr}_dsteps-{d}_z-128_ch-64"


def tile_diversity(grid_img_path, ncols=8):
    img = mpimg.imread(str(grid_img_path))
    if img.ndim == 3:
        img = img.mean(axis=2)
    H, W = img.shape
    th, tw = H // ncols, W // ncols
    tiles = []
    for r in range(ncols):
        for c in range(ncols):
            tiles.append(img[r * th : (r + 1) * th, c * tw : (c + 1) * tw].flatten())
    tiles = np.array(tiles)
    idx = np.random.choice(len(tiles), size=min(32, len(tiles)), replace=False)
    sub = tiles[idx]
    diffs = [np.mean(np.abs(sub[i] - sub[j])) for i in range(len(sub)) for j in range(i + 1, len(sub))]
    return float(np.mean(diffs)) if diffs else 0.0


# ── 1. GAN baseline loss curves ───────────────────────────────────────────────

baseline_dir = GAN_OUT / "ds-fashionmnist_ep-1_bs-128_lr-0.0002_dsteps-1_z-128_ch-64"
log = load_log(baseline_dir)
steps = [r["step"] for r in log]
lossD = [r["lossD"] for r in log]
lossG = [r["lossG"] for r in log]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(steps, lossD, alpha=0.25, color="steelblue")
axes[0].plot(steps, smooth(lossD), color="steelblue", lw=2, label="smoothed")
axes[0].set_title("GAN Baseline — Discriminator Loss")
axes[0].set_xlabel("step"); axes[0].set_ylabel("lossD"); axes[0].legend()

axes[1].plot(steps, lossG, alpha=0.25, color="coral")
axes[1].plot(steps, smooth(lossG), color="coral", lw=2, label="smoothed")
axes[1].set_title("GAN Baseline — Generator Loss")
axes[1].set_xlabel("step"); axes[1].set_ylabel("lossG"); axes[1].legend()

plt.tight_layout()
plt.savefig(FIG_OUT / "01_baseline_loss_curves.png", dpi=120)
plt.close()
print(f"[fig] 01_baseline_loss_curves.png  (finalD={lossD[-1]:.4f}, finalG={lossG[-1]:.4f})")


# ── 2. GAN baseline sample progression ───────────────────────────────────────

sample_files = sorted((baseline_dir / "samples").glob("grid_step*.png"))
fig, axes = plt.subplots(1, len(sample_files), figsize=(4 * len(sample_files), 4))
if len(sample_files) == 1:
    axes = [axes]
for ax, fpath in zip(axes, sample_files):
    ax.imshow(mpimg.imread(str(fpath)), cmap="gray")
    ax.set_title(fpath.stem, fontsize=8)
    ax.axis("off")
plt.suptitle("GAN Baseline — Sample Grids Over Training", y=1.02)
plt.tight_layout()
plt.savefig(FIG_OUT / "02_baseline_sample_progression.png", dpi=120, bbox_inches="tight")
plt.close()
print("[fig] 02_baseline_sample_progression.png")


# ── 3. Diffusion baseline samples ────────────────────────────────────────────

diff_run = next(iter(DIFF_OUT.iterdir()))
with open(diff_run / "summary.json") as f:
    summary = json.load(f)

diff_samples = sorted((diff_run / "samples").glob("samples_step*.png"))
diff_denoise = sorted((diff_run / "samples").glob("denoise_steps_step*.png"))

fig, axes = plt.subplots(1, len(diff_samples), figsize=(4 * len(diff_samples), 4))
if len(diff_samples) == 1:
    axes = [axes]
for ax, fpath in zip(axes, diff_samples):
    ax.imshow(mpimg.imread(str(fpath)), cmap="gray")
    ax.set_title(fpath.stem, fontsize=8)
    ax.axis("off")
plt.suptitle("Diffusion Baseline (T=200, eps) — Samples", y=1.02)
plt.tight_layout()
plt.savefig(FIG_OUT / "03_diffusion_samples.png", dpi=120, bbox_inches="tight")
plt.close()
print("[fig] 03_diffusion_samples.png")

if diff_denoise:
    fig, axes = plt.subplots(1, len(diff_denoise), figsize=(4 * len(diff_denoise), 4))
    if len(diff_denoise) == 1:
        axes = [axes]
    for ax, fpath in zip(axes, diff_denoise):
        ax.imshow(mpimg.imread(str(fpath)), cmap="gray")
        ax.set_title(fpath.stem, fontsize=7)
        ax.axis("off")
    plt.suptitle("Diffusion Baseline — Denoising Trajectory", y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_OUT / "03b_diffusion_denoise.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("[fig] 03b_diffusion_denoise.png")


# ── 4. Load all 6 grid run logs ───────────────────────────────────────────────

all_logs = {}
for lr in LRS:
    for d in DSTEPS:
        all_logs[(lr, d)] = load_log(run_dir_for(lr, d))


# ── 5. Grid loss curves ───────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
for col, lr in enumerate(LRS):
    for row, d in enumerate(DSTEPS):
        ax = axes[row][col]
        log2 = all_logs[(lr, d)]
        st = [r["step"] for r in log2]
        ld = [r["lossD"] for r in log2]
        lg = [r["lossG"] for r in log2]
        ax.plot(st, ld, alpha=0.2, color="steelblue")
        ax.plot(st, smooth(ld), color="steelblue", lw=1.8, label="lossD")
        ax.plot(st, lg, alpha=0.2, color="coral")
        ax.plot(st, smooth(lg), color="coral", lw=1.8, label="lossG")
        ax.set_title(f"lr={lr}, d_steps={d}", fontsize=10)
        ax.set_ylim(-0.1, 8)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        if row == 0 and col == 0:
            ax.legend(fontsize=8)
fig.suptitle("GAN Grid: Loss Curves  (rows = d_steps, cols = lr)", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(FIG_OUT / "04_grid_loss_curves.png", dpi=120, bbox_inches="tight")
plt.close()
print("[fig] 04_grid_loss_curves.png")


# ── 6. Grid final sample images ───────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for col, lr in enumerate(LRS):
    for row, d in enumerate(DSTEPS):
        rd = run_dir_for(lr, d)
        samples = sorted((rd / "samples").glob("grid_step*.png"))
        img = mpimg.imread(str(samples[-1]))
        ax = axes[row][col]
        ax.imshow(img, cmap="gray")
        ax.set_title(f"lr={lr}, d_steps={d}", fontsize=10)
        ax.axis("off")
fig.suptitle("GAN Grid: Final Sample Grids  (rows = d_steps, cols = lr)", fontsize=13)
plt.tight_layout()
plt.savefig(FIG_OUT / "05_grid_final_samples.png", dpi=120, bbox_inches="tight")
plt.close()
print("[fig] 05_grid_final_samples.png")


# ── 7. Failure mode: oscillation plot ─────────────────────────────────────────

log_u = all_logs[(0.0001, 2)]
st_u  = [r["step"] for r in log_u]
ld_u  = [r["lossD"] for r in log_u]
lg_u  = [r["lossG"] for r in log_u]

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(st_u, ld_u, alpha=0.35, color="steelblue")
ax.plot(st_u, smooth(ld_u, 30), color="steelblue", lw=2, label="lossD (smooth)")
ax.plot(st_u, lg_u, alpha=0.35, color="coral")
ax.plot(st_u, smooth(lg_u, 30), color="coral", lw=2, label="lossG (smooth)")
ax.set_title("Failure Mode: lr=0.0001, d_steps=2 — Discriminator Oscillation")
ax.set_xlabel("step"); ax.set_ylabel("loss")
ax.legend(); ax.set_ylim(-0.1, 8)
plt.tight_layout()
plt.savefig(FIG_OUT / "06_failure_oscillation.png", dpi=120)
plt.close()
print("[fig] 06_failure_oscillation.png")


# ── 8. GAN vs Diffusion side-by-side ─────────────────────────────────────────

best_gan = sorted((run_dir_for(0.0004, 2) / "samples").glob("grid_step*.png"))[-1]
best_diff = sorted((diff_run / "samples").glob("samples_step*.png"))[-1]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(mpimg.imread(str(best_gan)), cmap="gray")
axes[0].set_title("Best GAN: lr=0.0004, d_steps=2 (step 400)", fontsize=11)
axes[0].axis("off")
axes[1].imshow(mpimg.imread(str(best_diff)), cmap="gray")
axes[1].set_title("Diffusion: T=200, eps-target (step 468)", fontsize=11)
axes[1].axis("off")
plt.suptitle("GAN vs. Diffusion — Fashion-MNIST Samples After 1 Epoch", fontsize=12)
plt.tight_layout()
plt.savefig(FIG_OUT / "07_gan_vs_diffusion.png", dpi=120, bbox_inches="tight")
plt.close()
print("[fig] 07_gan_vs_diffusion.png")


# ── 9. Summary table ──────────────────────────────────────────────────────────

print("\n=== Summary Table ===")
print(f"{'lr':>8} {'d_steps':>8} {'finalD':>8} {'finalG':>8} {'stdD_last100':>14} {'diversity':>10} {'label':>28}")
print("-" * 90)
for lr in LRS:
    for d in DSTEPS:
        log3 = all_logs[(lr, d)]
        fd = log3[-1]["lossD"]
        fg = log3[-1]["lossG"]
        last100D = [r["lossD"] for r in log3[-100:]]
        std_d = float(np.std(last100D))
        rd = run_dir_for(lr, d)
        samples3 = sorted((rd / "samples").glob("grid_step*.png"))
        div = tile_diversity(samples3[-1])
        if fd < 0.15:
            label = "D dominates / collapse risk"
        elif fd > 1.0:
            label = "D failing / G winning"
        else:
            label = "balanced"
        print(f"{lr:>8} {d:>8} {fd:>8.4f} {fg:>8.4f} {std_d:>14.4f} {div:>10.4f} {label:>28}")

print(f"\nAll figures saved to: {FIG_OUT}")
