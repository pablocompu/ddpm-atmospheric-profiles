# Synoptically-Conditioned Diffusion Model for Vertical Atmospheric Profiles

A denoising diffusion probabilistic model (DDPM) that generates vertical atmospheric profiles — temperature, mixing ratio, and wind speed — conditioned on synoptic-scale meteorological variables from ERA5. Developed as part of a Bachelor's thesis on deep generative modelling for meteorological applications at Tenerife (GCAS station, WMO 60018).

---

## Repository structure

```
.
├── unets_perfiles_synoptic.py   # 1D U-Net architecture
├── main_perfiles_synoptic.py    # GaussianDiffusion class + training loop
├── data_perfiles_synoptic.py    # PyTorch Dataset + normalisation utilities
├── train_synoptic_example.py    # High-level training entry point
├── evaluar_modelo.py            # Full test-set evaluation + figures
├── eval_figuras.py              # Cache-based figure generation
├── comparar_perfil.py           # Profile-by-profile visual comparison
└── plot_training_loss.py        # Training / validation loss curves
```

---

## Requirements

```
python >= 3.9
torch >= 2.0
numpy
pandas
scipy
matplotlib
tqdm
easydict
```

Install with:

```bash
pip install torch numpy pandas scipy matplotlib tqdm easydict
```

---

## Data

| File | Source | Description |
|---|---|---|
| `sondeos_interpolados.csv` | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21238090.svg)](https://doi.org/10.5281/zenodo.21238090) | Radiosonde profiles interpolated to a uniform 311-level height grid (105 – 3 205 m, 10 m step). Columns: `Fecha`, `HGHT`, `TEMP`, `MIXR`, `SKNT`. |
| `era5_boxtfe.csv` | This repository (`era5_boxtfe.zip`) | Daily ERA5 synoptic variables for the Tenerife region. Columns: `fecha`, plus 23 meteorological fields (pressure, temperature, humidity, wind, geopotential at multiple levels). |
| `eval_cache.pkl` | This repository | Pre-computed inference results for all 1 581 test profiles (see [Cache-based figure generation](#cache-based-figure-generation)). |

Download `sondeos_interpolados.csv` from Zenodo and unzip `era5_boxtfe.zip` before running any script. By default every script looks for these files in the **current working directory**. Pass `--profiles-csv` and `--synoptic-csv` to point to a different location.

---

## Training

### Quick start

```bash
python train_synoptic_example.py
```

This runs 50 epochs with the default hyperparameters (batch size 32, lr 1e-4, 1 000 diffusion steps). Checkpoints are saved to `./trained_models_perfiles_synoptic/`.

### Custom training

```bash
python main_perfiles_synoptic.py \
    --epochs 800 \
    --batch-size 64 \
    --lr 1e-4 \
    --diffusion-steps 1000 \
    --sampling-steps 250 \
    --save-dir ./trained_models_v2/
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--epochs` | 800 | Number of training epochs |
| `--batch-size` | 64 | Batch size per GPU |
| `--lr` | 1e-4 | Initial learning rate (cosine schedule with 5-epoch warm-up) |
| `--diffusion-steps` | 1000 | Total diffusion timesteps |
| `--sampling-steps` | 250 | Reverse-process steps at inference |
| `--save-dir` | `./trained_models_perfiles_synoptic/` | Checkpoint directory |
| `--ddim` | off | Use DDIM sampling instead of DDPM |

The best checkpoint (lowest validation loss) is saved as `model_best_ema.pt`.

---

## Evaluation

### Full test-set evaluation

```bash
python evaluar_modelo.py \
    --model-dir trained_models_v2 \
    --sampling-steps 250
```

Produces six figures:

| Figure | Content |
|---|---|
| `eval_rmse_por_altura.png` | Temperature RMSE profile |
| `eval_perfiles_mensuales.png` | Mean monthly temperature profiles |
| `eval_inversion_base_mensual.png` | Monthly inversion base height |
| `eval_inversion_delta_mensual.png` | Monthly inversion delta-T |
| `eval_inversion_espesor_mensual.png` | Monthly inversion depth |
| `eval_scatter_inversion.png` | Scatter: observed vs generated inversion base height |

Optional inversion height filter:

```bash
python evaluar_modelo.py --inv-min-altura 300 --inv-max-altura 2500
```

### Cache-based figure generation

Running the full evaluation (1 581 test profiles × 250 steps) requires a GPU. Use the cache workflow to regenerate figures without re-running inference:

```bash
# First run — saves results to cache
python eval_figuras.py --cache eval_cache.pkl --prefix eval_v7

# Subsequent runs — loads cache, regenerates figures in seconds
python eval_figuras.py --cache eval_cache.pkl --prefix eval_v7 \
    --inv-min-altura 300 --inv-max-altura 2500
```

An additional figure `eval_v7_inv_freq.png` shows the monthly inversion detection frequency (percentage of profiles with an inversion) for generated vs observed.

> **Note on `eval_cache.pkl`:** The cache file included in this repository was produced with `model_best_ema.pt` and the fixed train/test split (`random_seed=42`, `--seed 42`). It contains the inference results for all 1 581 test profiles. If you retrain the model or change the random seed the cached results will no longer correspond to your model and you should regenerate the cache with `--recompute`.

---

## Visual profile comparison

Generate a multi-panel comparison of individual profiles:

```bash
# Default: 6 panels (2 rows x 3 columns), temperature only
python comparar_perfil.py --temp-only --num-dias 6

# All three variables (temperature, mixing ratio, wind speed)
python comparar_perfil.py --num-dias 4

# Specific date
python comparar_perfil.py --temp-only --fecha 2005-07-15

# With ensemble selection (5 members, pick closest to median)
python comparar_perfil.py --temp-only --num-ensemble 5 --num-dias 6
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--temp-only` | off | Show temperature only in a 3-column grid |
| `--num-dias` | 6 | Number of dates to compare |
| `--sampling-steps` | 250 | Reverse-process steps |
| `--num-ensemble` | 1 | Ensemble size; selects member closest to median inversion base |
| `--fecha` | random | Fix one specific date (YYYY-MM-DD) |
| `--seed` | None | Random seed for reproducible date selection |
| `--output` | `comparacion_perfil.png` | Output filename |

---

## Training loss curve

Parse a training log file and plot loss + learning-rate curves:

```bash
python plot_training_loss.py
```

The script reads `training_v2.log` (expected in the same directory) and writes `training_loss.png`.

---

## Model architecture

The generator is a **1D U-Net** operating on sequences of length 311 (one value per height level):

```
Input  (B, 3, 311)
  |
  |-- Time embedding  (sinusoidal, 64-dim -> MLP -> 256-dim)
  |-- Synoptic embedding (23-var MLP -> 256-dim)  [sum]
  |
Encoder:  64 ch (311)  ->  128 ch (156)  ->  256 ch (78)
Bottleneck: 256 ch (39) + self-attention
Decoder: 128 ch (78) + attention  ->  64 ch (156)  ->  64 ch (311)
  |
Output (B, 3, 311)
```

Total parameters: ~4.5 M.

**Normalisation:**
- Profiles: linear scaling to [-1, 1] using fixed physical ranges (TEMP: -90 to 50 °C; MIXR: 0 to 55 g kg⁻¹; SKNT: 0 to 250 kt).
- Synoptic variables: z-score normalisation (μ = 0, σ = 1) computed over the full dataset.

---

## Reproducibility

The train/test split (80 % / 20 %) is determined by a fixed random seed (`random_seed=42`). All scripts accept a `--seed` argument for reproducible profile selection at inference time.

---

## License

This code is released for academic and research purposes. Please cite the associated thesis if you use it.
