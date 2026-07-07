"""
Cache-based evaluation script for iterating figure styles without re-running inference.

Usage:
    # First run: execute inference and cache results
    python eval_figuras.py --cache eval_cache.pkl

    # Subsequent runs: skip inference, regenerate figures only (seconds)
    python eval_figuras.py --cache eval_cache.pkl

    # Try different output prefixes without touching the cache
    python eval_figuras.py --cache eval_cache.pkl --prefix eval_v7

    # Force re-inference even if a cache file exists
    python eval_figuras.py --cache eval_cache.pkl --recompute
"""

import argparse
import os
import pickle
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from evaluar_modelo import (
    load_model,
    denormalize_profile,
    detect_inversion,
    generate_single_profile,
    generate_ensemble_profile,
)
from data_perfiles_synoptic import get_metadata, PerfilesSynopticDataset
from main_perfiles_synoptic import GaussianDiffusion


# Default data paths — override via CLI arguments
_DEFAULT_PROFILES_CSV = "sondeos_interpolados.csv"
_DEFAULT_SYNOPTIC_CSV = "era5_boxtfe.csv"

# ── Constants ─────────────────────────────────────────────────────────────────
VARIABLES    = ['TEMP', 'MIXR', 'SKNT']
MESES_NOMBRE = ['January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December']
MESES_ABREV  = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']

# ─────────────────────────────────────────────────────────────────────────────
# INFERENCIA
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(args):
    """Run the model on the test set and return a dictionary of raw results."""
    import random
    from scipy.interpolate import interp1d

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # Find model checkpoint
    if args.model is None:
        import glob
        best = os.path.join(args.model_dir, 'model_best_ema.pt')
        if os.path.exists(best):
            args.model = best
        else:
            candidates = sorted(
                glob.glob(os.path.join(args.model_dir, 'model_epoch_*_ema.pt')),
                key=lambda x: int(x.split('_epoch_')[1].split('_')[0])
            )
            if candidates:
                args.model = candidates[-1]
            else:
                raise FileNotFoundError(f'No se encontraron modelos en {args.model_dir}')
    print(f'Modelo: {args.model}')

    dataset = PerfilesSynopticDataset(
        csv_path=getattr(args, 'profiles_csv', _DEFAULT_PROFILES_CSV),
        synoptic_csv_path=getattr(args, 'synoptic_csv', _DEFAULT_SYNOPTIC_CSV),
        split='test',
    )
    metadata  = get_metadata('perfiles_synoptic')
    model     = load_model(args.model, metadata, device)
    diffusion = GaussianDiffusion(timesteps=1000, device=device)

    alturas   = dataset.alturas
    n_alt     = len(alturas)
    fechas    = dataset.perfiles_validos

    if args.month is not None:
        fechas = [f for f in fechas if int(f.split('-')[1]) == args.month]
    if args.max_samples is not None:
        fechas = fechas[:args.max_samples]

    n = len(fechas)
    print(f'Evaluating {n} profiles...')

    errores_por_altura    = np.full((n, n_alt), np.nan)
    rmse_total            = {v: [] for v in VARIABLES}
    mae_total             = {v: [] for v in VARIABLES}
    perfiles_por_mes_gen  = {m: [] for m in range(1, 13)}
    perfiles_por_mes_real = {m: [] for m in range(1, 13)}
    inversion_por_mes_gen  = defaultdict(list)
    inversion_por_mes_real = defaultdict(list)
    scatter_base_real  = []
    scatter_base_gen   = []
    scatter_delta_real = []
    scatter_delta_gen  = []
    scatter_cima_real  = []
    scatter_cima_gen   = []

    def _valida(base):
        if base is None:
            return False
        if args.inv_min_altura is not None and base < args.inv_min_altura:
            return False
        if args.inv_max_altura is not None and base > args.inv_max_altura:
            return False
        return True

    use_ensemble = args.num_ensemble > 1
    for i, fecha in enumerate(tqdm(fechas, desc='Generando')):
        if use_ensemble:
            pg, real_df, _ = generate_ensemble_profile(
                model, diffusion, dataset, metadata, fecha, device,
                args.sampling_steps, args.num_ensemble, alturas,
            )
        else:
            pg, real_df, _ = generate_single_profile(
                model, diffusion, dataset, metadata, fecha, device, args.sampling_steps,
            )

        if len(real_df) == 0:
            continue

        real_interp = {}
        for var in VARIABLES:
            from scipy.interpolate import interp1d as _i1d
            f = _i1d(real_df['HGHT'].values, real_df[var].values,
                     kind='linear', fill_value='extrapolate', bounds_error=False)
            real_interp[var] = f(alturas)

        errores_por_altura[i] = (pg['TEMP'] - real_interp['TEMP']) ** 2

        for var in VARIABLES:
            rmse_total[var].append(np.sqrt(np.mean((pg[var] - real_interp[var]) ** 2)))
            mae_total[var].append(np.mean(np.abs(pg[var] - real_interp[var])))

        mes = int(fecha.split('-')[1])
        perfiles_por_mes_gen[mes].append(pg['TEMP'])
        perfiles_por_mes_real[mes].append(real_interp['TEMP'])

        b_gen,  _, d_gen  = detect_inversion(pg['TEMP'],           alturas)
        b_real, c_real, d_real = detect_inversion(real_interp['TEMP'], alturas)
        c_gen = alturas[np.argmax(pg['TEMP'][int(np.searchsorted(alturas, b_gen)):]) + int(np.searchsorted(alturas, b_gen))] if b_gen is not None else None

        if _valida(b_gen):
            inversion_por_mes_gen[mes].append((b_gen, c_gen, d_gen))
        if _valida(b_real):
            inversion_por_mes_real[mes].append((b_real, c_real, d_real))
        if _valida(b_gen) and _valida(b_real):
            scatter_base_real.append(b_real)
            scatter_base_gen.append(b_gen)
            scatter_delta_real.append(d_real)
            scatter_delta_gen.append(d_gen)
            scatter_cima_real.append(c_real)
            scatter_cima_gen.append(c_gen)

    return {
        'errores_por_altura':    errores_por_altura,
        'rmse_total':            rmse_total,
        'mae_total':             mae_total,
        'perfiles_por_mes_gen':  dict(perfiles_por_mes_gen),
        'perfiles_por_mes_real': dict(perfiles_por_mes_real),
        'inversion_por_mes_gen': dict(inversion_por_mes_gen),
        'inversion_por_mes_real':dict(inversion_por_mes_real),
        'scatter_base_real':      scatter_base_real,
        'scatter_base_gen':       scatter_base_gen,
        'scatter_delta_real':     scatter_delta_real,
        'scatter_delta_gen':      scatter_delta_gen,
        'scatter_cima_real':      scatter_cima_real,
        'scatter_cima_gen':       scatter_cima_gen,
        'num_samples':           n,
        'alturas':               alturas,
        'model_name':            os.path.basename(args.model),
    }


# ─────────────────────────────────────────────────────────────────────────────
# FIGURAS  ← edita aquí libremente sin tocar la inferencia
# ─────────────────────────────────────────────────────────────────────────────

def make_figures(data, prefix, sampling_steps=250,
                 inv_min_altura=None, inv_max_altura=None):
    """Generate all figures from a cached results dictionary."""

    alturas    = data['alturas']
    n          = data['num_samples']
    model_name = data.get('model_name', 'model')
    subtitle   = f'{model_name} | {n} profiles'

    # Filtrar inversiones por rango de altura (se aplica sobre la caché)
    def _valida(base):
        if base is None:
            return False
        if inv_min_altura is not None and base < inv_min_altura:
            return False
        if inv_max_altura is not None and base > inv_max_altura:
            return False
        return True

    def _filtrar(inv_dict):
        return {m: [v for v in vlist if _valida(v[0])]
                for m, vlist in inv_dict.items()}

    inv_gen  = _filtrar(data['inversion_por_mes_gen'])
    inv_real = _filtrar(data['inversion_por_mes_real'])
    hay_inversiones = any(len(inv_gen.get(m, [])) > 0 for m in range(1, 13))

    # Re-filtrar scatter para que sea consistente con los filtros actuales
    scatter_pairs = [
        (r, g) for r, g in zip(data['scatter_base_real'], data['scatter_base_gen'])
        if _valida(r) and _valida(g)
    ]

    meses_x = list(range(1, 13))

    def _agregar(inv_dict, idx):
        medias, stds = [], []
        for m in range(1, 13):
            vals = [v[idx] for v in inv_dict.get(m, []) if v[idx] is not None]
            medias.append(np.nanmean(vals) if vals else np.nan)
            stds.append(np.nanstd(vals) if vals else 0.0)
        return np.array(medias), np.array(stds)

    # ── 1. RMSE por altura ────────────────────────────────────────────────────
    rmse_por_altura = np.sqrt(np.nanmean(data['errores_por_altura'], axis=0))

    fig, ax = plt.subplots(figsize=(7, 9))
    ax.plot(rmse_por_altura, alturas, color='#e74c3c', linewidth=2.5)
    ax.fill_betweenx(alturas, 0, rmse_por_altura, alpha=0.25, color='#e74c3c')
    ax.set_xlabel('Temperature RMSE (°C)', fontsize=12)
    ax.set_ylabel('Height (m)', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    rmse_min = rmse_por_altura.min(); rmse_max = rmse_por_altura.max()
    ax.text(0.97, 0.03,
            f'min {rmse_min:.2f}°C @ {alturas[np.argmin(rmse_por_altura)]:.0f} m\n'
            f'max {rmse_max:.2f}°C @ {alturas[np.argmax(rmse_por_altura)]:.0f} m\n'
            f'mean {rmse_por_altura.mean():.2f}°C',
            transform=ax.transAxes, fontsize=9, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='#ccc'))
    plt.tight_layout()
    out = f'{prefix}_rmse.png'
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f'  {out}')

    # ── 2. Perfiles mensuales ─────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    # Rango común de temperatura
    all_t = []
    for m in range(1, 13):
        for p in [data['perfiles_por_mes_gen'][m], data['perfiles_por_mes_real'][m]]:
            if p:
                mu = np.mean(p, axis=0); sg = np.std(p, axis=0)
                all_t.extend([mu - sg, mu + sg])
    xmin = min(t.min() for t in all_t)
    xmax = max(t.max() for t in all_t)
    margin = (xmax - xmin) * 0.05
    xlims = (xmin - margin, xmax + margin)

    for mes in range(1, 13):
        ax = axes[mes - 1]
        pg  = data['perfiles_por_mes_gen'][mes]
        pr  = data['perfiles_por_mes_real'][mes]
        if pg:
            mu_g = np.mean(pg, axis=0); sg_g = np.std(pg, axis=0)
            mu_r = np.mean(pr, axis=0); sg_r = np.std(pr, axis=0)
            ax.plot(mu_g, alturas, color='#3498db', lw=2,   label='Generated')
            ax.fill_betweenx(alturas, mu_g - sg_g, mu_g + sg_g, alpha=0.20, color='#3498db')
            ax.plot(mu_r, alturas, color='#e74c3c', lw=1.5, ls='--', label='Observed')
            ax.fill_betweenx(alturas, mu_r - sg_r, mu_r + sg_r, alpha=0.15, color='#e74c3c')
            rmse_m = np.sqrt(np.mean((mu_g - mu_r) ** 2))
            ax.text(0.97, 0.03, f'RMSE {rmse_m:.2f} °C\nn={len(pg)}',
                    transform=ax.transAxes, fontsize=8, va='bottom', ha='right',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', color='gray')

        ax.set_title(MESES_NOMBRE[mes - 1], fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.set_xlim(xlims)
        if mes in [1, 5, 9]:
            ax.set_ylabel('Height (m)', fontsize=9)
        if mes >= 9:
            ax.set_xlabel('Temp (°C)', fontsize=9)
        if mes == 1:
            ax.legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    out = f'{prefix}_meses.png'
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f'  {out}')

    if not hay_inversiones:
        print('  (no inversions detected — skipping inversion figures)')
        return

    # ── 3. Inversion base height ──────────────────────────────────────────────
    bm_g, bs_g = _agregar(inv_gen,  0)
    bm_r, bs_r = _agregar(inv_real, 0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(meses_x, bm_g, 'o-', color='#3498db', lw=2, label='Generated')
    ax.fill_between(meses_x, bm_g - bs_g, bm_g + bs_g, alpha=0.2, color='#3498db')
    ax.plot(meses_x, bm_r, 's--', color='#e74c3c', lw=1.5, label='Observed')
    ax.fill_between(meses_x, bm_r - bs_r, bm_r + bs_r, alpha=0.15, color='#e74c3c')
    ax.set_xticks(meses_x); ax.set_xticklabels(MESES_ABREV, fontsize=10)
    ax.set_ylabel('Inversion base height (m)', fontsize=12)
    ax.set_xlabel('Month', fontsize=12)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    out = f'{prefix}_inv_base.png'
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f'  {out}')

    # ── 4. Inversion ΔT ──────────────────────────────────────────────────────
    dm_g, ds_g = _agregar(inv_gen,  2)
    dm_r, ds_r = _agregar(inv_real, 2)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(meses_x, dm_g, 'o-', color='#3498db', lw=2, label='Generated')
    ax.fill_between(meses_x, dm_g - ds_g, dm_g + ds_g, alpha=0.2, color='#3498db')
    ax.plot(meses_x, dm_r, 's--', color='#e74c3c', lw=1.5, label='Observed')
    ax.fill_between(meses_x, dm_r - ds_r, dm_r + ds_r, alpha=0.15, color='#e74c3c')
    ax.set_xticks(meses_x); ax.set_xticklabels(MESES_ABREV, fontsize=10)
    ax.set_ylabel('Inversion ΔT (°C)', fontsize=12)
    ax.set_xlabel('Month', fontsize=12)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    out = f'{prefix}_inv_delta.png'
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f'  {out}')

    # ── 5. Inversion depth ───────────────────────────────────────────────────
    def _espesor(inv_dict):
        d = defaultdict(list)
        for m in range(1, 13):
            for base, cima, _ in inv_dict.get(m, []):
                if base is not None and cima is not None:
                    d[m].append(cima - base)
        return d

    def _agg_scalar(d):
        mu, sg = [], []
        for m in range(1, 13):
            v = d[m]
            mu.append(np.nanmean(v) if v else np.nan)
            sg.append(np.nanstd(v)  if v else 0.0)
        return np.array(mu), np.array(sg)

    em_g, es_g = _agg_scalar(_espesor(inv_gen))
    em_r, es_r = _agg_scalar(_espesor(inv_real))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(meses_x, em_g, 'o-', color='#3498db', lw=2, label='Generated')
    ax.fill_between(meses_x, em_g - es_g, em_g + es_g, alpha=0.2, color='#3498db')
    ax.plot(meses_x, em_r, 's--', color='#e74c3c', lw=1.5, label='Observed')
    ax.fill_between(meses_x, em_r - es_r, em_r + es_r, alpha=0.15, color='#e74c3c')
    ax.set_xticks(meses_x); ax.set_xticklabels(MESES_ABREV, fontsize=10)
    ax.set_ylabel('Inversion depth (m)', fontsize=12)
    ax.set_xlabel('Month', fontsize=12)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    out = f'{prefix}_inv_espesor.png'
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f'  {out}')

    # ── 6. Scatter observed vs generated base height ──────────────────────────
    if len(scatter_pairs) >= 2:
        x_obs = np.array([p[0] for p in scatter_pairs])
        y_mod = np.array([p[1] for p in scatter_pairs])
        corr = np.corrcoef(x_obs, y_mod)[0, 1]
        rmse = np.sqrt(np.mean((y_mod - x_obs) ** 2))
        bias = np.mean(y_mod - x_obs)
        p    = np.polyfit(x_obs, y_mod, 1)
        xl   = np.linspace(x_obs.min(), x_obs.max(), 200)
        lims = [min(x_obs.min(), y_mod.min()) - 50,
                max(x_obs.max(), y_mod.max()) + 50]

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(x_obs, y_mod, alpha=0.4, s=14, color='#3498db', edgecolors='none')
        ax.plot(xl, np.polyval(p, xl), color='#e74c3c', lw=2,
                label=f'Regression (r={corr:.2f})')
        ax.plot(lims, lims, 'k--', lw=1, alpha=0.5, label='1:1')
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel('Observed inversion base height (m)', fontsize=12)
        ax.set_ylabel('Generated inversion base height (m)', fontsize=12)
        stats_text = f'RMSE = {rmse:.0f} m\nBias = {bias:+.0f} m\nn = {len(x_obs)}'
        ax.text(0.97, 0.03, stats_text, transform=ax.transAxes,
                fontsize=10, va='bottom', ha='right',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout()
        out = f'{prefix}_scatter.png'
        plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
        print(f'  {out}')

    # ── 7. Inversion detection frequency per month ────────────────────────────
    freq_gen  = []
    freq_real = []
    for m in range(1, 13):
        n_gen  = len(data['perfiles_por_mes_gen'].get(m, []))
        n_real = len(data['perfiles_por_mes_real'].get(m, []))
        freq_gen.append(
            len(inv_gen.get(m, []))  / n_gen  * 100 if n_gen  > 0 else np.nan
        )
        freq_real.append(
            len(inv_real.get(m, [])) / n_real * 100 if n_real > 0 else np.nan
        )

    freq_gen  = np.array(freq_gen)
    freq_real = np.array(freq_real)

    x     = np.arange(1, 13)
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, freq_gen,  width, color='#3498db', alpha=0.85, label='Generated')
    ax.bar(x + width / 2, freq_real, width, color='#e74c3c', alpha=0.85, label='Observed')
    ax.set_xticks(x)
    ax.set_xticklabels(MESES_ABREV, fontsize=10)
    ax.set_ylabel('Profiles with inversion (%)', fontsize=12)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    out = f'{prefix}_inv_freq.png'
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f'  {out}')

    # ── 8. Print LaTeX table for inversion error metrics ─────────────────────
    _print_inversion_table(data, _valida)


def _print_inversion_table(data, valida_fn):
    """Print LaTeX table with RMSE / MAE / Bias for inversion properties."""
    if 'scatter_delta_real' not in data:
        print('  (cache does not contain paired delta/cima — re-run with --recompute)')
        return

    # Rebuild filtered paired arrays
    base_r  = np.array(data['scatter_base_real'])
    base_g  = np.array(data['scatter_base_gen'])
    delta_r = np.array(data['scatter_delta_real'])
    delta_g = np.array(data['scatter_delta_gen'])
    cima_r  = np.array(data['scatter_cima_real'], dtype=float)
    cima_g  = np.array(data['scatter_cima_gen'],  dtype=float)

    mask = np.array([valida_fn(b) for b in base_r]) & np.array([valida_fn(b) for b in base_g])
    base_r, base_g   = base_r[mask],  base_g[mask]
    delta_r, delta_g = delta_r[mask], delta_g[mask]
    cima_r,  cima_g  = cima_r[mask],  cima_g[mask]

    esp_r = cima_r - base_r
    esp_g = cima_g - base_g

    def _stats(gen, obs, label, unit):
        err   = gen - obs
        rmse  = np.sqrt(np.mean(err ** 2))
        mae   = np.mean(np.abs(err))
        bias  = np.mean(err)
        n     = len(err)
        return rmse, mae, bias, n

    rows = [
        ('Inversion base height (m)', 'm', base_g,  base_r),
        (r'Inversion $\Delta T$ (\degree C)', '°C', delta_g, delta_r),
        ('Inversion depth (m)',       'm', esp_g,   esp_r),
    ]

    print('\n' + '─' * 60)
    print('LaTeX table — inversion error metrics')
    print('─' * 60)
    print(r'\begin{table}[H]')
    print(r'  \centering')
    print(r'  \begin{tabular}{|l|c|c|c|c|}')
    print(r'    \hline')
    print(r'    Property & N & RMSE & MAE & Bias \\')
    print(r'    \hline')
    for label, unit, gen, obs in rows:
        rmse, mae, bias, n = _stats(gen, obs, label, unit)
        fmt = '.0f' if unit == 'm' else '.2f'
        print(f'    {label} & ${n}$ & ${rmse:{fmt}}$ & ${mae:{fmt}}$ & ${bias:+{fmt}}$ \\\\')
        print(r'    \hline')
    print(r'  \end{tabular}')
    print(r'  \caption{Error metrics for inversion properties over paired profiles '
          r'(inversion detected in both generated and observed, base height 300--2500\,m).}')
    print(r'  \label{tab:inversion_metrics}')
    print(r'\end{table}')
    print('─' * 60 + '\n')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cache',          type=str,   default=None,
                        help='.pkl file to save/load inference results')
    parser.add_argument('--recompute',      action='store_true',
                        help='Force re-inference even if a cache file exists')
    parser.add_argument('--model',          type=str,   default=None)
    parser.add_argument('--model-dir',      type=str,   default='trained_models_v2')
    parser.add_argument('--profiles-csv',   type=str,   default=_DEFAULT_PROFILES_CSV,
                        help='Path to sondeos_interpolados.csv')
    parser.add_argument('--synoptic-csv',   type=str,   default=_DEFAULT_SYNOPTIC_CSV,
                        help='Path to era5_boxtfe.csv')
    parser.add_argument('--sampling-steps', type=int,   default=250)
    parser.add_argument('--num-ensemble',   type=int,   default=1)
    parser.add_argument('--seed',           type=int,   default=42)
    parser.add_argument('--max-samples',    type=int,   default=None)
    parser.add_argument('--month',          type=int,   default=None)
    parser.add_argument('--inv-min-altura', type=float, default=None)
    parser.add_argument('--inv-max-altura', type=float, default=None)
    parser.add_argument('--prefix',         type=str,   default='eval',
                        help='Output filename prefix (default: eval)')
    args = parser.parse_args()

    # ── Caché ─────────────────────────────────────────────────────────────────
    cache_exists = args.cache and os.path.exists(args.cache)

    if cache_exists and not args.recompute:
        print(f'[cache] Loading {args.cache} ...')
        with open(args.cache, 'rb') as fh:
            data = pickle.load(fh)
        print(f'  {data["num_samples"]} profiles. Inference skipped.\n')
    else:
        data = run_inference(args)
        if args.cache:
            with open(args.cache, 'wb') as fh:
                pickle.dump(data, fh)
            print(f'[cache] Saved to {args.cache} '
                  f'({os.path.getsize(args.cache)/1e6:.1f} MB)\n')

    # ── Figures ───────────────────────────────────────────────────────────────
    print(f'Generating figures with prefix "{args.prefix}_" ...')
    make_figures(data, prefix=args.prefix,
                 inv_min_altura=args.inv_min_altura,
                 inv_max_altura=args.inv_max_altura)
    print('Done.')


if __name__ == '__main__':
    main()