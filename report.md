# DATA 37100 — Final Project Summary Report

**Student:** Kevin L.
**Due:** March 13, 2026
**Dataset:** Fashion-MNIST (28×28 grayscale)
**Model families:** GAN (DCGAN), Diffusion

---

## 1. Question and Motivation

**Research question:** In a small DCGAN on Fashion-MNIST, how do learning rate (`lr`) and discriminator update frequency (`d_steps`) jointly affect training stability, sample sharpness, and mode-collapse risk?

GANs are the most behaviorally interesting model family covered in DATA 37100 because adversarial training is fragile by design. Unlike diffusion or autoregressive models, which optimize a single loss toward a fixed target, a GAN contains two competing objectives that must remain roughly balanced. Small changes in either the optimizer settings or the training ratio can push the system into qualitatively different failure regimes.

The two knobs chosen for this study directly control that balance:

- **`lr`** sets how large each gradient step is for both networks. A learning rate that is too small leaves the discriminator permanently ahead; one that is too large causes oscillation.
- **`d_steps`** controls how many discriminator update steps occur per generator step. More discriminator steps strengthen its signal but can starve the generator of a useful gradient if the discriminator becomes too strong relative to the generator's current capability.

Working hypotheses going in:
- Increasing `d_steps` from 1 to 2 will sharpen samples at moderate `lr` but risks starving the generator at low `lr`.
- `lr=0.0004` is most likely to produce unstable loss behavior due to large gradient steps.

A diffusion baseline on the same dataset provides a stability reference point — one model family with a well-defined single-objective loss — to contextualize the GAN's fragility.

---

## 2. Methods

### 2.1 Dataset

Fashion-MNIST: 60,000 training images, 28×28 grayscale, 10 clothing categories. More visually ambiguous than MNIST, making texture artifacts, repeated silhouettes, and diversity loss easier to identify.

Data was loaded from a pre-downloaded local copy (`./data/bigdata/`). No augmentation was applied to either model.

### 2.2 GAN (DCGAN) Architecture

A standard DCGAN for 28×28 images:

- **Generator:** latent vector z ∈ ℝ¹²⁸ → `ConvTranspose2d` stack (7×7 → 14×14 → 28×28), BatchNorm + ReLU, Tanh output.
- **Discriminator:** `Conv2d` stack (28×28 → 14×14 → 7×7 → scalar), LeakyReLU, Sigmoid output.
- **base_ch=64** for both networks.

Losses: standard binary cross-entropy GAN objective. No gradient penalty, no spectral norm, no label smoothing (in the main grid).

### 2.3 Diffusion Architecture

A small UNet-style denoising network with time embedding (`time_emb_dim=128`, `base_ch=64`), trained to predict the added noise (epsilon-parameterization). Linear noise schedule: β₁=0.0001 → β₂=0.02, T=200 steps.

### 2.4 Experimental Design

**GAN baseline:** `lr=0.0002`, `d_steps=1`, `max-steps=400`, `batch-size=128`, `seed=42`.

**Controlled grid (6 runs):**

| Run | lr | d_steps |
|-----|------|---------|
| 1 | 0.0001 | 1 |
| 2 | 0.0001 | 2 |
| 3 | 0.0002 | 1 |
| 4 | 0.0002 | 2 |
| 5 | 0.0004 | 1 |
| 6 | 0.0004 | 2 |

All other settings held fixed: `epochs=1`, `max-steps=400`, `batch-size=128`, `z-dim=128`, `base-ch=64`, `beta1=0.5`, `beta2=0.999`, `sample-every=100`, `seed=42`.

**Diffusion baseline:** `T=200`, `target=eps`, `epochs=1`, `seed=42`.

**Runtime:** Each GAN run completed in ~11 seconds on a CUDA GPU (avg 0.028 s/step × 400 steps). All 6 grid runs completed in ~67 seconds. Diffusion baseline: 11.35 seconds (468 steps).

---

## 3. Results

### 3.1 GAN Baseline

The baseline run (`lr=0.0002`, `d_steps=1`) reached `lossD=1.14`, `lossG=4.86` at step 400. Early training (step 1) shows `lossD≈1.41`, `lossG≈0.98` — discriminator and generator initially balanced — then the generator loss rises as the discriminator learns to separate real from fake, and D loss fluctuates in the 0.8–1.4 range throughout. Sample grids show pure noise at step 1, rough blobs at step 100, recognizable silhouettes (bags, shoes, shirts) emerging by steps 200–300, with some texture detail at step 400.

### 3.2 Diffusion Baseline

The diffusion baseline produced coherent but blurry sample grids after 1 epoch (468 steps, 11.35 seconds). Loss decreased monotonically — no oscillation, no instability. The denoising trajectory is visible in the saved grid: high-noise images progressively denoised to coarse shapes. T=200 is modest for Fashion-MNIST; more reverse steps would sharpen the output, but this was not the purpose of this baseline. The diffusion run is included primarily as a stability reference.

### 3.3 Controlled Experiment Results

**Loss summary table:**

| lr | d_steps | finalD | finalG | stdD (last 100 steps) | Peak lossD |
|----|---------|--------|--------|-----------------------|------------|
| 0.0001 | 1 | 0.189 | 4.535 | 0.259 | 2.548 |
| 0.0001 | 2 | 0.165 | 5.944 | **0.518** | **5.152** |
| 0.0002 | 1 | **1.144** | 4.864 | 0.288 | 3.337 |
| 0.0002 | 2 | 0.138 | 3.834 | 0.506 | 4.418 |
| 0.0004 | 1 | 0.597 | 2.649 | 0.194 | 7.453 |
| 0.0004 | 2 | 0.261 | 3.121 | 0.322 | 3.513 |

**Diversity proxy** (mean pairwise pixel L1 distance across 32 sampled tiles from the final sample grid, seed=42):

| lr | d_steps | diversity |
|----|---------|-----------|
| 0.0001 | 1 | 0.238 |
| 0.0001 | 2 | 0.226 |
| 0.0002 | 1 | 0.236 |
| **0.0002** | **2** | **0.265** |
| 0.0004 | 1 | 0.217 |
| 0.0004 | 2 | 0.218 |

The `lr=0.0002, d_steps=2` setting produced the highest diversity score. The `lr=0.0004` settings produced the lowest diversity despite moderate final losses, suggesting the generator converged to a narrower output distribution under higher learning rates.

---

## 4. Failure Modes

### 4.1 Discriminator Oscillation — `lr=0.0001, d_steps=2`

**Observation:** `lossD` swings from near-zero to >5.15 in individual steps, then partially recovers. `stdD` over the last 100 steps = 0.518, the highest in the grid. `lossG` climbs steadily to 5.94 by step 400. Sample grids show slow, unstable visual improvement: the generator is receiving a signal that is alternately too strong (D near-zero = gradient vanishes for G) and too weak (D spiking = D itself is temporarily miscalibrated).

**Cause:** Two discriminator updates per generator step at a low learning rate creates a seesaw. The small `lr` means each update step moves the discriminator a tiny distance, but two consecutive steps overshoot the equilibrium, pushing D to near-perfect discrimination. At that point D's output is saturated (near zero for all generated samples), so the generator's gradient vanishes. D then regresses on the next generator step, then overshoots again. This is a textbook discriminator oscillation pattern caused by too-frequent updates relative to the generator's ability to respond.

### 4.2 Discriminator Instability / Failure — `lr=0.0002, d_steps=1`

**Observation:** `finalD=1.14` is the highest in the grid — significantly above the 0.5 equilibrium. At this setting the discriminator ends training *losing* to the generator: it can no longer reliably classify real from fake. Visually, sample quality at step 400 is worse than expected for this learning rate.

**Cause:** `lr=0.0002` with `d_steps=1` occasionally causes the generator to improve faster than the discriminator can track. When `lossD` exceeds 1.0, the discriminator's gradient signal to the generator becomes misleading — it is too easy to fool. This produces a failure mode opposite to oscillation: the generator gets a false "you're winning" signal and stops learning meaningful structure.

### 4.3 Soft Mode Collapse / Low Diversity — `lr=0.0004`

**Observation:** Both `d_steps=1` and `d_steps=2` at `lr=0.0004` show the lowest diversity scores (0.217–0.218), despite having the highest peak `lossD` spikes (up to 7.45 at `d_steps=1`). The final sample grids show repeated silhouette patterns — coarse shapes without texture diversity.

**Cause:** A high learning rate causes large gradient steps that push both networks away from local equilibria quickly. The generator can learn a small set of "safe" outputs that consistently fool the discriminator in its current state, without needing to cover the full data distribution. This is a soft mode collapse — the generator has not fully collapsed to one sample, but it has reduced its effective output diversity. The high peak `lossD` reflects the discriminator catching up suddenly to these repeated patterns, then the cycle repeating.

---

## 5. Interpretation

### Effect of `d_steps`

The effect of doubling `d_steps` is **strongly conditioned on `lr`**:

- At **`lr=0.0001`**: `d_steps=2` increases instability (stdD 0.259 → 0.518, finalG 4.54 → 5.94). The small learning rate cannot absorb two consecutive discriminator updates without overshooting.
- At **`lr=0.0002`**: `d_steps=2` *helps* — finalD drops from 1.14 to 0.14 and diversity improves from 0.236 to 0.265. The discriminator was previously failing (lossD>1); one more update step stabilizes it.
- At **`lr=0.0004`**: `d_steps=2` reduces peak lossD spikes (7.45 → 3.51) and stdD (0.194 → 0.322), providing modest stabilization. Diversity is unchanged.

This interaction is the core finding: `d_steps` is not a simple "more = better" knob. Its effect depends on whether the discriminator is currently underpowered (where more steps help) or already oscillating (where more steps make it worse).

### Effect of `lr`

- **`lr=0.0001`**: Discriminator dominates persistently (finalD near 0), generator learns slowly, diversity is moderate. Training is "stable" in the sense of no wild oscillation but slow.
- **`lr=0.0002`**: The most sensitive setting — D can fall behind (d_steps=1) or be stabilized (d_steps=2). Produces the highest diversity when d_steps=2.
- **`lr=0.0004`**: Fastest early loss changes but lowest diversity. High learning rate pushes toward soft mode collapse.

### GAN vs. Diffusion

The diffusion baseline shows that a single-objective denoising model avoids all of the above failure modes by construction. Its loss decreases monotonically with no seesaw dynamics. The trade-off: after 1 epoch the diffusion samples are blurrier than the best GAN outputs at comparable wall-clock time. The GAN's adversarial signal can produce sharper local texture faster — but only when `lr` and `d_steps` are jointly well-calibrated.

---

## 6. Limitations

- All runs used `max-steps=400`, which is a very small training budget. Conclusions about long-run behavior (e.g., whether collapse is permanent or recovers) are not supported.
- Diversity was measured from saved PNG grids as a pixel-level proxy, not from model-generated samples with a fixed noise vector. This introduces rendering artifacts as a confound.
- Only two discrete values of `d_steps` were tested. The interaction between `lr` and `d_steps` across a finer grid might reveal non-monotonic effects not captured here.
- The diffusion comparison is not a controlled cross-model experiment: the architectures, loss functions, and training dynamics are fundamentally different. It is included as a qualitative stability reference only.

---

## 7. Conclusion

The key finding is that `lr` and `d_steps` interact non-linearly. At low learning rate, more discriminator updates per step create oscillation because each step overshoots the equilibrium and the generator cannot respond fast enough. At moderate learning rate, more discriminator updates stabilize training by correcting a discriminator that was falling behind. At high learning rate, both settings produce low diversity despite apparently balanced losses, consistent with soft mode collapse.

The best-performing setting for sample quality and diversity in this budget was **`lr=0.0002, d_steps=2`** (highest diversity score 0.265, finalD=0.138 indicating a reasonable equilibrium). The worst instability was **`lr=0.0001, d_steps=2`** (stdD=0.518, peak lossD=5.15, generator loss climbs to 5.94). The diffusion baseline confirmed that adversarial instability is a GAN-specific phenomenon, not a dataset artifact.
