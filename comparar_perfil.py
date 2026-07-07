"""
Generate and compare vertical profiles: model output vs radiosonde observations.

Produces a multi-panel figure showing temperature (and optionally mixing ratio
and wind speed) profiles for a set of randomly selected test dates.
"""

import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import random
import os
import glob

from data_perfiles_synoptic import get_metadata, PerfilesSynopticDataset
from unets_perfiles_synoptic import unet_1d_perfiles
from main_perfiles_synoptic import GaussianDiffusion

# Default data paths — override via CLI arguments
_DEFAULT_PROFILES_CSV = "sondeos_interpolados.csv"
_DEFAULT_SYNOPTIC_CSV = "era5_boxtfe.csv"


def load_model(model_path, metadata, device):
    """Load a trained checkpoint, auto-detecting attention blocks."""
    checkpoint    = torch.load(model_path, map_location=device)
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
    Surface inversions are discarded. Minimum thresholds: delta_T >= 0.5 deg C,
    depth >= 50 m.
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

    n_with_inv = int(np.sum(~np.isnan(bases)))
    bases_str  = [f"{b:.0f}" if not np.isnan(b) else "N/A" for b in bases]
    median_str = f"{median_base:.0f}" if not np.isnan(median_base) else "N/A"
    print(f"    ensemble {num_ensemble} -> bases: {bases_str} "
          f"-> median={median_str} -> selected idx={best_idx} "
          f"({n_with_inv}/{num_ensemble} with inversion)")

    perfil_real_df = dataset.df[dataset.df['Fecha'] == fecha].sort_values('HGHT')
    return profiles[best_idx], perfil_real_df, synoptic_conditions


def main():
    parser = argparse.ArgumentParser("Compare generated vs observed vertical profiles")
    parser.add_argument("--model",          type=str,   default=None)
    parser.add_argument("--model-dir",      type=str,   default="trained_models_v2")
    parser.add_argument("--profiles-csv",   type=str,   default=_DEFAULT_PROFILES_CSV,
                        help="Path to sondeos_interpolados.csv")
    parser.add_argument("--synoptic-csv",   type=str,   default=_DEFAULT_SYNOPTIC_CSV,
                        help="Path to era5_boxtfe.csv")
    parser.add_argument("--fecha",          type=str,   default=None,
                        help="Specific date (YYYY-MM-DD)")
    parser.add_argument("--num-dias",       type=int,   default=6,
                        help="Number of dates to compare")
    parser.add_argument("--sampling-steps", type=int,   default=250)
    parser.add_argument("--num-ensemble",   type=int,   default=1)
    parser.add_argument("--output",         type=str,   default="comparacion_perfil.png")
    parser.add_argument("--seed",           type=int,   default=None)
    parser.add_argument("--temp-only",      action="store_true",
                        help="Show temperature only in a 3-column grid")
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

    if args.seed is not None:
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

    num_dias = args.num_dias
    fechas   = []
    if args.fecha and args.fecha in dataset.perfiles_validos:
        fechas.append(args.fecha)
    available = [f for f in dataset.perfiles_validos if f not in fechas]
    random.shuffle(available)
    fechas.extend(available[:num_dias - len(fechas)])
    print(f"Selected dates: {fechas}")

    alturas      = dataset.alturas
    use_ensemble = args.num_ensemble > 1

    perfiles_gen  = []
    perfiles_real = []
    synoptic_list = []

    for i, fecha in enumerate(fechas):
        print(f"  [{i+1}/{num_dias}] {fecha}")
        if use_ensemble:
            pg, pr, sc = generate_ensemble_profile(
                model, diffusion, dataset, metadata, fecha, device,
                args.sampling_steps, args.num_ensemble, alturas,
            )
        else:
            pg, pr, sc = generate_single_profile(
                model, diffusion, dataset, metadata, fecha, device, args.sampling_steps,
            )
        perfiles_gen.append(pg)
        perfiles_real.append(pr)
        synoptic_list.append(sc)

    print("Profiles generated.")

    # ── Pre-compute metrics ───────────────────────────────────────────────────
    from scipy.interpolate import interp1d

    variables  = ['TEMP', 'MIXR', 'SKNT']
    xlabels    = ['Temperature (deg C)', 'Mixing Ratio (g/kg)', 'Wind Speed (kt)']
    colors_gen = ['#3498db', '#2ecc71', '#e74c3c']
    colors_obs = ['#1a5276', '#196f3d', '#922b21']

    metricas           = {var: {'rmse': [], 'mae': []} for var in variables}
    inv_base_gen_list  = []
    inv_base_real_list = []
    computed           = []

    for fecha, pg, pr, sc in zip(fechas, perfiles_gen, perfiles_real, synoptic_list):
        real_interp = {}
        if len(pr) > 0:
            for var in variables:
                f = interp1d(pr['HGHT'].values, pr[var].values,
                             kind='linear', fill_value='extrapolate', bounds_error=False)
                real_interp[var] = f(alturas)

        base_gen,  cima_gen,  delta_gen  = detect_inversion(pg['TEMP'], alturas)
        base_real, cima_real, delta_real = (
            detect_inversion(real_interp['TEMP'], alturas)
            if real_interp else (None, None, None)
        )
        if base_gen is not None and base_real is not None:
            inv_base_gen_list.append(base_gen)
            inv_base_real_list.append(base_real)

        rmse_vals, mae_vals = {}, {}
        for var in variables:
            if real_interp:
                rmse = np.sqrt(np.mean((pg[var] - real_interp[var]) ** 2))
                mae  = np.mean(np.abs(pg[var] - real_interp[var]))
                metricas[var]['rmse'].append(rmse)
                metricas[var]['mae'].append(mae)
                rmse_vals[var], mae_vals[var] = rmse, mae
            else:
                rmse_vals[var] = mae_vals[var] = None

        computed.append(dict(
            fecha=fecha, profile_gen=pg, perfil_real_df=pr,
            real_interp=real_interp,
            base_gen=base_gen,  cima_gen=cima_gen,  delta_gen=delta_gen,
            base_real=base_real, cima_real=cima_real, delta_real=delta_real,
            rmse=rmse_vals, mae=mae_vals,
        ))

    # ── Plot ─────────────────────────────────────────────────────────────────
    if args.temp_only:
        C_GEN = '#2471a3'
        C_OBS = '#c0392b'
        ncols = 3
        nrows = (num_dias + ncols - 1) // ncols
        YMAX  = 2500

        xlims = (4, 22)

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(5.5 * ncols, 7 * nrows),
            sharex=True, sharey=True, squeeze=False,
        )

        for idx, c in enumerate(computed):
            r, col_i = divmod(idx, ncols)
            ax = axes[r, col_i]

            ax.plot(c['profile_gen']['TEMP'], alturas,
                    color=C_GEN, linewidth=2.0, label='Generated', alpha=0.9)
            if len(c['perfil_real_df']) > 0:
                ax.plot(c['perfil_real_df']['TEMP'].values,
                        c['perfil_real_df']['HGHT'].values,
                        color=C_OBS, linewidth=1.6, linestyle='--',
                        label='Observed', alpha=0.9)

            if c['base_gen'] is not None:
                ax.axhspan(c['base_gen'], c['cima_gen'], alpha=0.12, color=C_GEN, zorder=0)
                ax.axhline(c['base_gen'], color=C_GEN, lw=1.4, ls='-',  alpha=0.85)
                ax.axhline(c['cima_gen'], color=C_GEN, lw=1.0, ls='--', alpha=0.70)
            if c['base_real'] is not None:
                ax.axhspan(c['base_real'], c['cima_real'], alpha=0.10, color=C_OBS, zorder=0)
                ax.axhline(c['base_real'], color=C_OBS, lw=1.4, ls='-',  alpha=0.85)
                ax.axhline(c['cima_real'], color=C_OBS, lw=1.0, ls='--', alpha=0.70)

            info = [c['fecha']]
            if c['base_gen'] is not None:
                info.append(f"Gen: {c['base_gen']:.0f}-{c['cima_gen']:.0f} m  dT={c['delta_gen']:.1f} C")
            if c['base_real'] is not None:
                info.append(f"Obs: {c['base_real']:.0f}-{c['cima_real']:.0f} m  dT={c['delta_real']:.1f} C")
            ax.text(0.03, 0.97, '\n'.join(info),
                    transform=ax.transAxes, fontsize=8.5, va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                              alpha=0.88, edgecolor='lightgray'))

            if c['rmse']['TEMP'] is not None:
                ax.text(0.97, 0.03,
                        f"RMSE: {c['rmse']['TEMP']:.2f} C\nMAE:  {c['mae']['TEMP']:.2f} C",
                        transform=ax.transAxes, fontsize=8.5, va='bottom', ha='right',
                        bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                                  alpha=0.88, edgecolor='lightgray'))

            if r == nrows - 1:
                ax.set_xlabel('Temperature (deg C)', fontsize=11)
            if col_i == 0:
                ax.set_ylabel('Height (m)', fontsize=11)
            ax.tick_params(labelsize=9)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if idx == 0:
                ax.legend(loc='lower left', fontsize=9, framealpha=0.88)

        for extra in range(num_dias, nrows * ncols):
            axes[extra // ncols, extra % ncols].set_visible(False)

        axes[0, 0].set_xlim(xlims)
        axes[0, 0].set_ylim(0, YMAX)
        plt.tight_layout()
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"Saved: {args.output}")

    else:
        # N rows x 3 columns
        _XLIMS = {'TEMP': (4, 22), 'MIXR': (0, 10), 'SKNT': (0, 25)}

        fig, axes = plt.subplots(num_dias, 3, figsize=(15, 4 * num_dias))
        if num_dias == 1:
            axes = axes[np.newaxis, :]

        for row, c in enumerate(computed):
            for col, (var, xlabel, cg, cr) in enumerate(
                zip(variables, xlabels, colors_gen, colors_obs)
            ):
                ax = axes[row, col]
                ax.plot(c['profile_gen'][var], alturas,
                        color=cg, linewidth=2, label='Generated', alpha=0.9)
                if c['real_interp']:
                    ax.plot(c['real_interp'][var], alturas,
                            color=cr, linewidth=1.5, linestyle='--',
                            label='Observed', alpha=0.8)
                    if c['rmse'][var] is not None:
                        unit = {'TEMP': 'C', 'MIXR': 'g/kg', 'SKNT': 'kt'}[var]
                        ax.text(0.97, 0.03,
                                f"RMSE: {c['rmse'][var]:.2f} {unit}\nMAE: {c['mae'][var]:.2f} {unit}",
                                transform=ax.transAxes, fontsize=8,
                                va='bottom', ha='right',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                          alpha=0.8, edgecolor='gray'))

                if col == 0:
                    if c['base_gen'] is not None:
                        ax.axhline(c['base_gen'],  color=cg, lw=1.2, ls=':', label='Gen base')
                        ax.axhline(c['cima_gen'],  color=cg, lw=1.2, ls='-.', label='Gen top')
                    if c['base_real'] is not None:
                        ax.axhline(c['base_real'], color=cr, lw=1.2, ls=':', label='Obs base')
                        ax.axhline(c['cima_real'], color=cr, lw=1.2, ls='-.', label='Obs top')

                if row == 0:
                    ax.set_title(xlabel, fontsize=11, fontweight='bold')
                if row == num_dias - 1:
                    ax.set_xlabel(xlabel.split('(')[0].strip(), fontsize=10)
                if col == 0:
                    ax.set_ylabel(f"{c['fecha']}\n\nHeight (m)", fontsize=9)

                ax.set_xlim(_XLIMS[var])
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                if row == 0:
                    ax.legend(loc='upper right', fontsize=7)

        plt.tight_layout()
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"Saved: {args.output}")

    # ── Scatter: inversion base heights ──────────────────────────────────────
    if len(inv_base_gen_list) >= 2:
        x_obs = np.array(inv_base_real_list)
        y_mod = np.array(inv_base_gen_list)
        corr  = np.corrcoef(x_obs, y_mod)[0, 1]
        p     = np.polyfit(x_obs, y_mod, 1)
        xl    = np.linspace(x_obs.min(), x_obs.max(), 100)
        lims  = [min(x_obs.min(), y_mod.min()) - 50, max(x_obs.max(), y_mod.max()) + 50]

        fig_sc, ax_sc = plt.subplots(figsize=(6, 6))
        ax_sc.scatter(x_obs, y_mod, color='#3498db', alpha=0.7, edgecolors='white', s=60)
        ax_sc.plot(xl, np.polyval(p, xl), color='#e74c3c', lw=1.5,
                   label=f'Regression (r={corr:.2f})')
        ax_sc.plot(lims, lims, 'k--', lw=1, alpha=0.5, label='1:1')
        ax_sc.set_xlim(lims); ax_sc.set_ylim(lims)
        ax_sc.set_xlabel('Observed inversion base height (m)', fontsize=12)
        ax_sc.set_ylabel('Generated inversion base height (m)', fontsize=12)
        ax_sc.legend(fontsize=10)
        ax_sc.grid(True, alpha=0.3, linestyle='--')
        ax_sc.spines['top'].set_visible(False)
        ax_sc.spines['right'].set_visible(False)
        scatter_out = args.output.replace('.png', '_scatter_inv.png')
        plt.tight_layout()
        plt.savefig(scatter_out, dpi=150, bbox_inches='tight')
        plt.close(fig_sc)
        print(f"Scatter saved: {scatter_out}")

    # ── Summary metrics ───────────────────────────────────────────────────────
    print(f"\nError metrics (mean over {num_dias} days):")
    units = {'TEMP': 'C', 'MIXR': 'g/kg', 'SKNT': 'kt'}
    for var in variables:
        if metricas[var]['rmse']:
            rmse_mean = np.mean(metricas[var]['rmse'])
            mae_mean  = np.mean(metricas[var]['mae'])
            rmse_std  = np.std(metricas[var]['rmse'])
            u = units[var]
            print(f"  {var}: RMSE={rmse_mean:.2f}+/-{rmse_std:.2f} {u}, MAE={mae_mean:.2f} {u}")


if __name__ == "__main__":
    main()
