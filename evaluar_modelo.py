"""
Full-test-set evaluation of the diffusion model.

Generates:
  1. Temperature RMSE by height
  2. Mean monthly temperature profiles (generated vs observed)
  3. Monthly inversion base height
  4. Monthly inversion delta-T
  5. Monthly inversion depth
  6. Scatter plot: observed vs generated inversion base height
"""

import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import random
import os
import glob
from collections import defaultdict
from tqdm import tqdm

from data_perfiles_synoptic import get_metadata, PerfilesSynopticDataset
from unets_perfiles_synoptic import unet_1d_perfiles
from main_perfiles_synoptic import GaussianDiffusion

# Default paths — override via CLI arguments
_DEFAULT_PROFILES_CSV = "sondeos_interpolados.csv"
_DEFAULT_SYNOPTIC_CSV = "era5_boxtfe.csv"


def load_model(model_path, metadata, device):
    """Load a trained checkpoint, auto-detecting attention blocks."""
    checkpoint  = torch.load(model_path, map_location=device)
    use_attention = any('attn_enc3' in k or 'attn_dec3' in k for k in checkpoint.keys())

    model = unet_1d_perfiles(
        image_size=metadata.image_size,
        in_channels=metadata.num_channels,
        out_channels=metadata.num_channels,
        use_synoptic_cond=True,
        num_synoptic_vars=metadata.num_synoptic_vars,
        use_attention=use_attention,
        num_heads=4,
    ).to(device)

    model.load_state_dict(checkpoint)
    model.eval()
    print(f"Model loaded (attention: {'yes' if use_attention else 'no'})")
    return model


def denormalize_profile(profile, dataset):
    """Denormalise a profile array from [-1, 1] to physical units."""
    result = {}
    for i, var in enumerate(['TEMP', 'MIXR', 'SKNT']):
        result[var] = dataset.denormalize(profile[i], var)
    return result


def intervalos(a, threshold0=0, threshold=0):
    """
    Return start and end indices of intervals where values drop below
    *threshold* and recover above *threshold0*.
    Based on the implementation by J. Carrillo.
    """
    inds, inde = [], []
    i = -1
    try:
        while i < len(a) - 1:
            i += 1
            if a[i] < threshold:
                inds.append(i)
                while (a[i] <= threshold0) and (i <= len(a) - 2):
                    i += 1
                else:
                    if i <= len(a) - 1:
                        inde.append(i)
        if inds[0] > inde[0]:
            inde = list(np.delete(inde, 0))
        if inds[len(inds) - 1] > inde[len(inde) - 1]:
            inds = list(np.delete(inds, len(inds) - 1))
    except Exception:
        pass
    return inds, inde


def detect_inversion(temp_profile, alturas):
    """
    Detect the strongest thermal inversion in a temperature profile.

    Returns (base_height, top_height, delta_T) or (None, None, None).
    Surface inversions (starting at the first level) are discarded.
    Minimum thresholds: delta_T >= 0.5 deg C, depth >= 50 m.
    """
    dT   = np.diff(temp_profile)
    dT   = np.append(dT, 0)
    inds, inde = intervalos(dT * -1, threshold0=0, threshold=0)

    if len(inds) == 0 or len(inde) == 0:
        return None, None, None

    if inds[0] == 0:
        inds, inde = inds[1:], inde[1:]

    if len(inds) == 0 or len(inde) == 0:
        return None, None, None

    best_delta, best_base, best_cima = -np.inf, None, None
    for s, e in zip(inds, inde):
        e_safe = min(e, len(temp_profile) - 1)
        dt = temp_profile[e_safe] - temp_profile[s]
        if dt > best_delta:
            best_delta, best_base, best_cima = dt, s, e_safe

    if best_base is None or best_delta < 0.5:
        return None, None, None
    if alturas[best_cima] - alturas[best_base] < 50:
        return None, None, None

    return alturas[best_base], alturas[best_cima], best_delta


def generate_single_profile(model, diffusion, dataset, metadata, fecha, device, sampling_steps):
    """Generate one profile for a given date using the reverse diffusion process."""
    synoptic_conditions = dataset.get_synoptic_conditions(fecha)
    synoptic_tensor     = torch.tensor(synoptic_conditions, dtype=torch.float32)

    with torch.no_grad():
        xT            = torch.randn(1, metadata.num_channels, metadata.image_size).to(device)
        synoptic_batch = synoptic_tensor.unsqueeze(0).to(device)
        generated     = diffusion.sample_from_reverse_process(
            model, xT, sampling_steps, {'synoptic': synoptic_batch}, ddim=False
        )
        generated = generated.cpu().numpy()[0]

    profile_gen    = denormalize_profile(generated, dataset)
    perfil_real_df = dataset.df[dataset.df['Fecha'] == fecha].sort_values('HGHT')
    return profile_gen, perfil_real_df, synoptic_conditions


def generate_ensemble_profile(model, diffusion, dataset, metadata, fecha, device,
                               sampling_steps, num_ensemble, alturas):
    """
    Generate *num_ensemble* profiles and select the member closest to the
    ensemble median inversion base height (without consulting observations).
    """
    synoptic_conditions = dataset.get_synoptic_conditions(fecha)
    synoptic_tensor     = torch.tensor(synoptic_conditions, dtype=torch.float32)
    synoptic_batch      = synoptic_tensor.unsqueeze(0).to(device)

    profiles, bases = [], []
    for _ in range(num_ensemble):
        with torch.no_grad():
            xT        = torch.randn(1, metadata.num_channels, metadata.image_size).to(device)
            generated = diffusion.sample_from_reverse_process(
                model, xT, sampling_steps, {'synoptic': synoptic_batch}, ddim=False
            )
            generated = generated.cpu().numpy()[0]
        p = denormalize_profile(generated, dataset)
        profiles.append(p)
        base, _, _ = detect_inversion(p['TEMP'], alturas)
        bases.append(base if base is not None else np.nan)

    median_base = np.nanmedian(bases)
    if np.isnan(median_base):
        best_idx = 0
    else:
        dists    = [abs(b - median_base) if not np.isnan(b) else np.inf for b in bases]
        best_idx = int(np.argmin(dists))

    perfil_real_df = dataset.df[dataset.df['Fecha'] == fecha].sort_values('HGHT')
    return profiles[best_idx], perfil_real_df, synoptic_conditions


def main():
    parser = argparse.ArgumentParser("Evaluate diffusion model on the full test set")
    parser.add_argument("--model",           type=str, default=None)
    parser.add_argument("--model-dir",       type=str, default="trained_models_v2")
    parser.add_argument("--profiles-csv",    type=str, default=_DEFAULT_PROFILES_CSV,
                        help="Path to sondeos_interpolados.csv")
    parser.add_argument("--synoptic-csv",    type=str, default=_DEFAULT_SYNOPTIC_CSV,
                        help="Path to era5_boxtfe.csv")
    parser.add_argument("--sampling-steps",  type=int, default=250)
    parser.add_argument("--num-ensemble",    type=int, default=1)
    parser.add_argument("--inv-min-altura",  type=float, default=None)
    parser.add_argument("--inv-max-altura",  type=float, default=None)
    parser.add_argument("--output-rmse",     type=str, default="eval_rmse_por_altura.png")
    parser.add_argument("--output-meses",    type=str, default="eval_perfiles_mensuales.png")
    parser.add_argument("--output-inv-base", type=str, default="eval_inversion_base_mensual.png")
    parser.add_argument("--output-inv-delta",type=str, default="eval_inversion_delta_mensual.png")
    parser.add_argument("--output-inv-espesor", type=str, default="eval_inversion_espesor_mensual.png")
    parser.add_argument("--output-scatter",  type=str, default="eval_scatter_inversion.png")
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--max-samples",     type=int, default=None)
    parser.add_argument("--month",           type=int, default=None)
    args = parser.parse_args()

    if args.model is None:
        best = os.path.join(args.model_dir, "model_best_ema.pt")
        if os.path.exists(best):
            args.model = best
        else:
            models = sorted(
                glob.glob(os.path.join(args.model_dir, "model_epoch_*_ema.pt")),
                key=lambda x: int(x.split('_epoch_')[1].split('_')[0]),
            )
            if models:
                args.model = models[-1]
            else:
                print(f"No model found in {args.model_dir}")
                return
    print(f"Model: {args.model}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    dataset  = PerfilesSynopticDataset(
        csv_path=args.profiles_csv,
        synoptic_csv_path=args.synoptic_csv,
        split='test',
    )
    metadata  = get_metadata("perfiles_synoptic")
    model     = load_model(args.model, metadata, device)
    diffusion = GaussianDiffusion(timesteps=1000, device=device)

    from scipy.interpolate import interp1d

    alturas   = dataset.alturas
    num_alt   = len(alturas)
    fechas    = dataset.perfiles_validos

    if args.month is not None:
        fechas = [f for f in fechas if int(f.split('-')[1]) == args.month]
    if args.max_samples is not None:
        fechas = fechas[:args.max_samples]

    n = len(fechas)
    print(f"Evaluating {n} profiles...")

    variables  = ['TEMP', 'MIXR', 'SKNT']
    month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    month_abbr  = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']

    errores_por_altura     = np.full((n, num_alt), np.nan)
    rmse_total             = {v: [] for v in variables}
    mae_total              = {v: [] for v in variables}
    perfiles_por_mes_gen   = {m: [] for m in range(1, 13)}
    perfiles_por_mes_real  = {m: [] for m in range(1, 13)}
    inversion_por_mes_gen  = defaultdict(list)
    inversion_por_mes_real = defaultdict(list)
    scatter_base_real      = []
    scatter_base_gen       = []

    def _valida(base):
        if base is None:
            return False
        if args.inv_min_altura is not None and base < args.inv_min_altura:
            return False
        if args.inv_max_altura is not None and base > args.inv_max_altura:
            return False
        return True

    use_ensemble = args.num_ensemble > 1
    for i, fecha in enumerate(tqdm(fechas, desc="Generating profiles")):
        if use_ensemble:
            profile_gen, real_df, _ = generate_ensemble_profile(
                model, diffusion, dataset, metadata, fecha, device,
                args.sampling_steps, args.num_ensemble, alturas,
            )
        else:
            profile_gen, real_df, _ = generate_single_profile(
                model, diffusion, dataset, metadata, fecha, device, args.sampling_steps,
            )

        if len(real_df) == 0:
            continue

        real_interp = {}
        for var in variables:
            f = interp1d(real_df['HGHT'].values, real_df[var].values,
                         kind='linear', fill_value='extrapolate', bounds_error=False)
            real_interp[var] = f(alturas)

        errores_por_altura[i] = (profile_gen['TEMP'] - real_interp['TEMP']) ** 2
        for var in variables:
            rmse_total[var].append(np.sqrt(np.mean((profile_gen[var] - real_interp[var]) ** 2)))
            mae_total[var].append(np.mean(np.abs(profile_gen[var] - real_interp[var])))

        mes = int(fecha.split('-')[1])
        perfiles_por_mes_gen[mes].append(profile_gen['TEMP'])
        perfiles_por_mes_real[mes].append(real_interp['TEMP'])

        base_gen,  cima_gen,  delta_gen  = detect_inversion(profile_gen['TEMP'],  alturas)
        base_real, cima_real, delta_real = detect_inversion(real_interp['TEMP'],  alturas)
        if _valida(base_gen):
            inversion_por_mes_gen[mes].append((base_gen,  cima_gen,  delta_gen))
        if _valida(base_real):
            inversion_por_mes_real[mes].append((base_real, cima_real, delta_real))
        if _valida(base_gen) and _valida(base_real):
            scatter_base_real.append(base_real)
            scatter_base_gen.append(base_gen)

    print("Evaluation complete.")

    # ── Figure 1: RMSE by height ──────────────────────────────────────────────
    rmse_por_altura = np.sqrt(np.nanmean(errores_por_altura, axis=0))
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(rmse_por_altura, alturas, color='#e74c3c', linewidth=2.5)
    ax.fill_betweenx(alturas, 0, rmse_por_altura, alpha=0.3, color='#e74c3c')
    ax.set_xlabel('Temperature RMSE (deg C)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Height (m)',               fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(0.97, 0.03,
            f'min  {rmse_por_altura.min():.2f} deg C @ {alturas[np.argmin(rmse_por_altura)]:.0f} m\n'
            f'max  {rmse_por_altura.max():.2f} deg C @ {alturas[np.argmax(rmse_por_altura)]:.0f} m\n'
            f'mean {rmse_por_altura.mean():.2f} deg C',
            transform=ax.transAxes, fontsize=10, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))
    plt.tight_layout()
    plt.savefig(args.output_rmse, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {args.output_rmse}")

    # ── Figure 2: Monthly mean profiles ──────────────────────────────────────
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    all_t = []
    for m in range(1, 13):
        for pg in [perfiles_por_mes_gen[m], perfiles_por_mes_real[m]]:
            if pg:
                mu = np.mean(pg, axis=0); sg = np.std(pg, axis=0)
                all_t.extend([mu - sg, mu + sg])
    xmin   = min(t.min() for t in all_t)
    xmax   = max(t.max() for t in all_t)
    margin = (xmax - xmin) * 0.05
    xlims  = (xmin - margin, xmax + margin)

    for mes in range(1, 13):
        ax = axes[mes - 1]
        if perfiles_por_mes_gen[mes]:
            mu_g = np.mean(perfiles_por_mes_gen[mes],  axis=0)
            sg_g = np.std(perfiles_por_mes_gen[mes],   axis=0)
            mu_r = np.mean(perfiles_por_mes_real[mes], axis=0)
            sg_r = np.std(perfiles_por_mes_real[mes],  axis=0)
            ax.plot(mu_g, alturas, color='#3498db', lw=2,    label='Generated')
            ax.fill_betweenx(alturas, mu_g - sg_g, mu_g + sg_g, alpha=0.20, color='#3498db')
            ax.plot(mu_r, alturas, color='#e74c3c', lw=1.5, ls='--', label='Observed')
            ax.fill_betweenx(alturas, mu_r - sg_r, mu_r + sg_r, alpha=0.15, color='#e74c3c')
            rmse_m = np.sqrt(np.mean((mu_g - mu_r) ** 2))
            ax.text(0.97, 0.03, f'RMSE {rmse_m:.2f} deg C\nn={len(perfiles_por_mes_gen[mes])}',
                    transform=ax.transAxes, fontsize=9, va='bottom', ha='right',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', color='gray')
        ax.set_title(month_names[mes - 1], fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.set_xlim(xlims)
        if mes in [1, 5, 9]:
            ax.set_ylabel('Height (m)', fontsize=10)
        if mes >= 9:
            ax.set_xlabel('Temp (deg C)', fontsize=10)
        if mes == 1:
            ax.legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    plt.savefig(args.output_meses, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {args.output_meses}")

    # ── Inversion figures ─────────────────────────────────────────────────────
    hay_inv = any(len(inversion_por_mes_gen[m]) > 0 for m in range(1, 13))
    meses_x = list(range(1, 13))

    def _agg(inv_dict, idx):
        mu, sg = [], []
        for m in range(1, 13):
            vals = [v[idx] for v in inv_dict[m] if v[idx] is not None]
            mu.append(np.nanmean(vals) if vals else np.nan)
            sg.append(np.nanstd(vals)  if vals else 0.0)
        return np.array(mu), np.array(sg)

    def _inv_fig(mu_g, sg_g, mu_r, sg_r, ylabel, outfile):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(meses_x, mu_g, 'o-',  color='#3498db', lw=2,   label='Generated')
        ax.fill_between(meses_x, mu_g - sg_g, mu_g + sg_g, alpha=0.20, color='#3498db')
        ax.plot(meses_x, mu_r, 's--', color='#e74c3c', lw=1.5, label='Observed')
        ax.fill_between(meses_x, mu_r - sg_r, mu_r + sg_r, alpha=0.15, color='#e74c3c')
        ax.set_xticks(meses_x); ax.set_xticklabels(month_abbr, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=12); ax.set_xlabel('Month', fontsize=12)
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(outfile, dpi=300, bbox_inches='tight'); plt.close(fig)
        print(f"Saved: {outfile}")

    if hay_inv:
        bm_g, bs_g = _agg(inversion_por_mes_gen,  0)
        bm_r, bs_r = _agg(inversion_por_mes_real, 0)
        _inv_fig(bm_g, bs_g, bm_r, bs_r, 'Inversion base height (m)', args.output_inv_base)

        dm_g, ds_g = _agg(inversion_por_mes_gen,  2)
        dm_r, ds_r = _agg(inversion_por_mes_real, 2)
        _inv_fig(dm_g, ds_g, dm_r, ds_r, 'Inversion delta-T (deg C)', args.output_inv_delta)

        esp_g = defaultdict(list)
        esp_r = defaultdict(list)
        for m in range(1, 13):
            for base, cima, _ in inversion_por_mes_gen[m]:
                esp_g[m].append(cima - base)
            for base, cima, _ in inversion_por_mes_real[m]:
                esp_r[m].append(cima - base)

        def _agg_scalar(d):
            mu, sg = [], []
            for m in range(1, 13):
                mu.append(np.nanmean(d[m]) if d[m] else np.nan)
                sg.append(np.nanstd(d[m])  if d[m] else 0.0)
            return np.array(mu), np.array(sg)

        em_g, es_g = _agg_scalar(esp_g)
        em_r, es_r = _agg_scalar(esp_r)
        _inv_fig(em_g, es_g, em_r, es_r, 'Inversion depth (m)', args.output_inv_espesor)

    if len(scatter_base_real) >= 2:
        x_obs = np.array(scatter_base_real)
        y_mod = np.array(scatter_base_gen)
        corr  = np.corrcoef(x_obs, y_mod)[0, 1]
        p     = np.polyfit(x_obs, y_mod, 1)
        xl    = np.linspace(x_obs.min(), x_obs.max(), 200)
        lims  = [min(x_obs.min(), y_mod.min()) - 50, max(x_obs.max(), y_mod.max()) + 50]

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(x_obs, y_mod, alpha=0.4, s=15, color='#3498db', edgecolors='none')
        ax.plot(xl, np.polyval(p, xl), color='#e74c3c', lw=2,
                label=f'Regression (r={corr:.2f})')
        ax.plot(lims, lims, 'k--', lw=1, alpha=0.5, label='1:1')
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel('Observed inversion base height (m)', fontsize=12)
        ax.set_ylabel('Generated inversion base height (m)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(args.output_scatter, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {args.output_scatter}")

    # ── Summary metrics ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    units = {'TEMP': 'deg C', 'MIXR': 'g/kg', 'SKNT': 'kt'}
    for var in variables:
        rmse_mean   = np.mean(rmse_total[var])
        rmse_median = np.median(rmse_total[var])
        rmse_std    = np.std(rmse_total[var])
        mae_mean    = np.mean(mae_total[var])
        mae_median  = np.median(mae_total[var])
        u = units[var]
        print(f"\n{var}:")
        print(f"  RMSE: mean={rmse_mean:.3f} +/- {rmse_std:.3f} {u}, median={rmse_median:.3f} {u}")
        print(f"  MAE : mean={mae_mean:.3f} {u},  median={mae_median:.3f} {u}")
    print("=" * 60)


if __name__ == "__main__":
    main()
