# DATA 37100 — Final Project

**Question:** In a small DCGAN on Fashion-MNIST, how do learning rate (`lr`) and discriminator update frequency (`d_steps`) jointly affect training stability, sample sharpness, and mode-collapse risk?

**Model families:** GAN (DCGAN), Diffusion
**Dataset:** Fashion-MNIST (28×28 grayscale)
**Controlled experiment:** 3 × 2 grid — `lr ∈ {0.0001, 0.0002, 0.0004}` × `d_steps ∈ {1, 2}`

---

## Setup

```bash
micromamba activate badvideo
```

Data is pre-downloaded under `./data/bigdata/`. No additional download needed for GAN runs.

---

## Run Commands

### GAN baseline (lr=0.0002, d_steps=1)

```bash
python src/gan_baseline.py \
  --dataset fashionmnist \
  --epochs 1 --max-steps 400 --batch-size 128 \
  --lr 0.0002 --d-steps 1 \
  --z-dim 128 --base-ch 64 \
  --beta1 0.5 --beta2 0.999 \
  --sample-every 100 --seed 42
```

### GAN controlled grid (6 runs)

```bash
python src/gan_baseline.py \
  --dataset fashionmnist \
  --epochs 1 --max-steps 400 --batch-size 128 \
  --z-dim 128 --base-ch 64 \
  --beta1 0.5 --beta2 0.999 \
  --sample-every 100 --seed 42 \
  --grid "lr=0.0001,0.0002,0.0004;d_steps=1,2"
```

### Diffusion baseline

```bash
python src/diffusion_baseline.py \
  --dataset fashion \
  --epochs 1 --T 200 --target eps \
  --base-ch 64 --beta2 0.02 \
  --sample-every 400 --seed 42 \
  --download
```

---

## Expected Runtime & Hardware

| Run | Steps | Time |
|-----|-------|------|
| GAN single run | 400 | ~11 s |
| GAN full grid (6 runs) | 2400 total | ~67 s |
| Diffusion baseline | 468 | ~11 s |

Tested on: **CUDA GPU** (NVIDIA). Outputs written to `./untrack/outputs/final/`.

The code falls back to MPS (Apple Silicon) or CPU automatically via `--device auto`.

---

## Outputs

```
untrack/outputs/final/
  gan/
    ds-fashionmnist_ep-1_bs-128_lr-{lr}_dsteps-{d}_z-128_ch-64/
      run_args.json       # hyperparameters
      train_log.csv       # step, lossD, lossG, sec_per_step
      samples/            # grid_step{N}.png at every sample_every steps
      checkpoint.pt
    baseline_loss_curves.png
    baseline_sample_progression.png
    grid_loss_curves.png
    grid_final_samples.png
    failure_loss_curve.png
    failure_lr0001_dsteps2.png
    results.csv
  diffusion/
    ds-fashion_T-200_target-eps_b2-0.02_ch-64/
      run_args.json
      summary.json
      samples/
      checkpoints/
    baseline_samples.png
    baseline_denoise.png
```

---

## Analysis

Open `analysis.ipynb` (all cells pre-executed). Sections:

1. GAN Baseline — loss curves and sample progression
2. Diffusion Baseline — samples and denoising trajectory
3. Controlled Experiment — 3×2 loss grid, summary table, sample grids, diversity proxy
4. Failure Mode Analysis — oscillation case (lr=0.0001, d_steps=2)
5. Interpretation Summary

---

## Repo Structure

```
DATA371_Final/
  README.md
  report.md               # summary report (~3-5 pages)
  analysis.ipynb          # full analysis with outputs
  src/
    gan_baseline.py
    diffusion_baseline.py
    lab07_diffusion_core.py
    run_analysis.py
  data/
    bigdata/              # Fashion-MNIST (not committed)
  untrack/                # all training outputs (not committed)
```

No large files (data, checkpoints) are committed to git.
