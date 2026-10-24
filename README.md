# Synthetic Data Generation with Diffusion Models

**Student:** Bartłomiej Cichy  
**Course:** [Synthetic Data Generation with Diffusion Models](https://patrykbart.github.io/teaching/synthetic-data-generation-with-diffusion-models/)  
**Detailed Experiment Log:** [`EXPERIMENTS.md`](EXPERIMENTS.md)

---

## Quick Navigation

- [Repository Structure & Where to Look](#repository-structure)
- [Meeting 1 — Real-Data Baselines (Completed)](#meeting-1--real-data-baselines)
- [Meeting 2 — Text-to-Image Generation](#meeting-2--text-to-image-generation)
- [Meeting 3 — Generation from Training Photos](#meeting-3--generation-from-training-photos)
- [Meeting 4 — Editing Training Photos](#meeting-4--editing-training-photos)
- [Final Comparison](#final-comparison)
- [Commands to Reproduce](#commands-to-reproduce)

---

## Repository Structure

| Path | Contents & Purpose |
| :--- | :--- |
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | **Main laboratory logbook** with full tables, charts, and analysis per meeting |
| [`src/synthdata/balance.py`](src/synthdata/balance.py) | Implementations of `oversample()` and `augment()` |
| [`evidence/`](evidence/) | High-resolution evaluation charts (`m1_*.png`) |
| [`runs/`](runs/) | Metrics (`results.json`, `test-results.json`) and per-seed checkpoints |
| [`pools/`](pools/) | Generated synthetic image pools, manifests, and contact sheets |
| [`generate_m1_plots.py`](generate_m1_plots.py) | Script regenerating Meeting 1 publication plots from raw run data |

---

## Meeting 1 — Real-Data Baselines

- **Objective:** Evaluate how classical imbalance countermeasures (oversampling and label-preserving data augmentation) impact Egyptian Mau recall compared to imbalanced training data.
- **Implemented Code:**
  - `oversample()` in [`src/synthdata/balance.py`](src/synthdata/balance.py): Distributes repeated samples uniformly across the 15 minority photos to reach the 55-sample budget.
  - `augment()` in [`src/synthdata/balance.py`](src/synthdata/balance.py): Stochastic spatial rotations, perspective shifts, and color jitter applied dynamically per epoch.
- **Completed Runs:**
  - `runs/m1-real/` (imbalanced baseline, 15 real Egyptian Mau training photos)
  - `runs/m1-oversample/` (oversampling +55 photos)
  - `runs/m1-augment/` (data augmentation +55 photos)

### Summary of Results (5 Seeds)

| Run | Validation Mau Recall | Test Mau Recall | Test Macro-F1 (All 8 Breeds) | Test Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| `m1-real` | 48.0% ± 7.6% | 50.7% ± 8.0% | 0.8576 ± 0.0094 | 86.33% ± 0.80% |
| `m1-oversample` | 67.0% ± 5.7% | 68.67% ± 1.8% | 0.8798 ± 0.0099 | 88.08% ± 1.00% |
| `m1-augment` | 69.0% ± 6.5% | 70.67% ± 5.5% | 0.8789 ± 0.0122 | 88.00% ± 1.16% |

### Key Evidence & Charts (in [`evidence/`](evidence/))

- [`evidence/m1_mau_recall.png`](evidence/m1_mau_recall.png): Egyptian Mau recall across Validation & Test sets with individual seed distributions.
- [`evidence/m1_macro_f1.png`](evidence/m1_macro_f1.png): Macro-F1 across all 8 breeds.
- [`evidence/m1_per_breed_test_recall.png`](evidence/m1_per_breed_test_recall.png): Test recall by breed confirming non-minority classes remain intact (>85–98%).
- [`evidence/m1_mau_misclassifications.png`](evidence/m1_mau_misclassifications.png): Error destination breakdown (Bengal, Russian Blue, Abyssinian).

### Key Takeaway
> **Oversampling significantly improves recall (+18.0 pp on test), whereas data augmentation provides virtually no additional benefit over simple oversampling (+2.0 pp recall, identical macro-F1).**  
> *Why:* Pixel-level perturbations (rotations, crops, jitter) cannot synthesize novel cat poses, backgrounds, or lighting conditions. The model still repeatedly overfits the same 15 underlying cats. Full commentary in [`EXPERIMENTS.md`](EXPERIMENTS.md).

---

## Meeting 2 — Text-to-Image Generation

*(Upcoming / next milestone)*

- **Objective:** Generate a pool of synthetic Egyptian Mau photos using text-to-image diffusion models (Stable Diffusion 1.5) and evaluate downstream classification benefit.
- **Reference Pool:** [`pools/pool-mau-basic/`](pools/pool-mau-basic/) (Contact sheet: [`pools/pool-mau-basic/contact-sheet.jpg`](pools/pool-mau-basic/contact-sheet.jpg))
- **Status:** Detailed experiments and sweep comparisons to be added.

---

## Meeting 3 — Generation from Training Photos

*(Upcoming)*

- **Objective:** Evaluate `img2img` (denoising strength sweep) and `ip-adapter` (scale sweep) conditioned on the 15 Egyptian Mau training images.
- **Status:** To be completed.

---

## Meeting 4 — Editing Training Photos

*(Upcoming)*

- **Objective:** Evaluate cat inpainting with diverse backgrounds versus outpainting (varying zoom/scale).
- **Status:** To be completed.

---

## Final Comparison

*(To be compiled after completion of Meetings 1–4)*

- Comparative ranking across all methods on validation and test splits.
- Final recommendation on synthetic data utility.

---

## Commands to Reproduce

All environments and dependencies are managed via `uv`:

```bash
# 1. Run unit tests
uv run pytest

# 2. Reproduce Meeting 1 training runs (5 seeds each)
uv run synth-train --name m1-real --method real
uv run synth-train --name m1-oversample --method oversample
uv run synth-train --name m1-augment --method augment

# 3. Evaluate models on test set
uv run synth-test --run m1-real
uv run synth-test --run m1-oversample
uv run synth-test --run m1-augment

# 4. Re-generate Meeting 1 evidence plots
python generate_m1_plots.py
```
