"""
PyTorch Dataset for synoptically-conditioned vertical meteorological profiles.

Loads radiosonde profiles (sondeos_interpolados.csv) and ERA5 synoptic
conditions (era5_boxtfe.csv), interpolates profiles to a uniform 311-level
height grid, and returns normalised tensors suitable for diffusion model
training.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from easydict import EasyDict
from scipy.interpolate import interp1d

# Default data paths — override via get_dataset() arguments if needed.
_DEFAULT_PROFILES_CSV = "sondeos_interpolados.csv"
_DEFAULT_SYNOPTIC_CSV = "era5_boxtfe.csv"


def get_metadata(dataset, root=None):
    """Return dataset metadata for the vertical-profile dataset."""
    if dataset in ["perfiles_meteo", "vertical_profiles", "perfiles_synoptic", "perfiles_synoptic_test"]:
        return EasyDict({
            "image_size":        311,   # height levels (105 m to 3205 m, 10 m step)
            "seq_length":        311,
            "channels":          3,     # TEMP, MIXR, SKNT
            "num_channels":      3,
            "num_classes":       None,
            "num_synoptic_vars": 23,
        })
    raise ValueError(f"Unknown dataset: {dataset}")


class PerfilesSynopticDataset(Dataset):
    """
    Vertical-profile dataset with synoptic conditioning.

    Each item is a tuple of:
        profile   : (3, 311) float32 tensor — TEMP, MIXR, SKNT in [-1, 1]
        label     : scalar 0 (dummy, for API compatibility)
        synoptic  : (23,) float32 tensor — z-score-normalised ERA5 variables
    """

    def __init__(
        self,
        csv_path,
        synoptic_csv_path,
        altura_min=105,
        altura_max=3205,
        altura_step=10,
        split='train',
        test_ratio=0.2,
        random_seed=42,
    ):
        """
        Args:
            csv_path          : path to sondeos_interpolados.csv
            synoptic_csv_path : path to era5_boxtfe.csv
            altura_min        : lowest height level (m)
            altura_max        : highest height level (m)
            altura_step       : vertical resolution (m)
            split             : 'train', 'test', or 'all'
            test_ratio        : fraction of profiles reserved for the test split
            random_seed       : seed for the reproducible train/test shuffle
        """
        print(f"Loading profiles from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        self.split = split
        print(f"  {len(self.df)} rows loaded")

        print(f"Loading synoptic conditions from {synoptic_csv_path}...")
        self.synoptic_df = pd.read_csv(synoptic_csv_path)
        print(f"  {len(self.synoptic_df)} synoptic records loaded")

        self.synoptic_vars = [c for c in self.synoptic_df.columns if c != 'fecha']
        print(f"  {len(self.synoptic_vars)} synoptic variables")

        # Filter physically implausible values
        len_original = len(self.df)
        self.df = self.df[
            (self.df['TEMP'] >= -100) & (self.df['TEMP'] <= 50) &
            (self.df['MIXR'] >= 0)   & (self.df['MIXR'] <= 50) &
            (self.df['SKNT'] >= 0)   & (self.df['SKNT'] <= 200)
        ]
        print(f"  {len_original - len(self.df)} anomalous rows removed")

        self.alturas     = np.arange(altura_min, altura_max + altura_step, altura_step)
        self.num_alturas = len(self.alturas)

        # Fixed physical normalisation ranges (avoids data leakage across splits)
        self.stats = {
            'TEMP': {'min': -90.0, 'max':  50.0},
            'MIXR': {'min':   0.0, 'max':  55.0},
            'SKNT': {'min':   0.0, 'max': 250.0},
        }

        # Z-score normalisation statistics for synoptic variables
        self.synoptic_stats = {
            var: {'mean': self.synoptic_df[var].mean(), 'std': self.synoptic_df[var].std()}
            for var in self.synoptic_vars
        }

        # Identify profiles that have matching synoptic conditions
        self.fechas = self.df['Fecha'].unique()
        self.perfiles_validos = []

        for fecha in self.fechas:
            perfil_df = self.df[self.df['Fecha'] == fecha]
            if (
                self.get_synoptic_conditions(fecha) is not None
                and len(perfil_df['HGHT'].values) >= 50
            ):
                self.perfiles_validos.append(fecha)

        print(f"  {len(self.perfiles_validos)} valid profiles found")

        if len(self.perfiles_validos) == 0:
            raise ValueError(
                "No valid profiles found. Verify that dates match between the two CSV files."
            )

        # Reproducible train / test split
        if split != 'all':
            self.perfiles_validos = sorted(self.perfiles_validos)
            np.random.seed(random_seed)
            indices = np.random.permutation(len(self.perfiles_validos))
            n_test  = int(len(self.perfiles_validos) * test_ratio)
            n_train = len(self.perfiles_validos) - n_test

            if split == 'train':
                selected = indices[:n_train]
                print(f"Split TRAIN: {n_train} profiles")
            elif split == 'test':
                selected = indices[n_train:]
                print(f"Split TEST: {n_test} profiles")
            else:
                raise ValueError(f"Unknown split '{split}'. Use 'train', 'test', or 'all'.")

            self.perfiles_validos = [self.perfiles_validos[i] for i in selected]
        else:
            print(f"No split: using all {len(self.perfiles_validos)} profiles")

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------

    def normalize(self, value, var_name):
        """Map profile variable from [min, max] to [-1, 1]."""
        vmin = self.stats[var_name]['min']
        vmax = self.stats[var_name]['max']
        return 2 * (value - vmin) / (vmax - vmin) - 1

    def denormalize(self, value, var_name):
        """Invert normalize(): map from [-1, 1] back to physical units."""
        vmin = self.stats[var_name]['min']
        vmax = self.stats[var_name]['max']
        return (value + 1) / 2 * (vmax - vmin) + vmin

    def normalize_synoptic(self, value, var_name):
        """Z-score normalisation for a synoptic variable."""
        mean = self.synoptic_stats[var_name]['mean']
        std  = self.synoptic_stats[var_name]['std']
        return (value - mean) / (std + 1e-8)

    def denormalize_synoptic(self, value, var_name):
        """Invert normalize_synoptic()."""
        mean = self.synoptic_stats[var_name]['mean']
        std  = self.synoptic_stats[var_name]['std']
        return value * std + mean

    # ------------------------------------------------------------------

    def get_synoptic_conditions(self, fecha):
        """Return normalised synoptic vector for a given date, or None."""
        if isinstance(fecha, str):
            fecha_dt = pd.to_datetime(fecha)
        else:
            fecha_dt = fecha

        row = self.synoptic_df[self.synoptic_df['fecha'] == fecha_dt]
        if len(row) == 0:
            fecha_str = (
                fecha_dt.strftime('%Y-%m-%d') if hasattr(fecha_dt, 'strftime')
                else str(fecha)
            )
            row = self.synoptic_df[
                self.synoptic_df['fecha'].astype(str).str.startswith(fecha_str)
            ]
        if len(row) == 0:
            return None

        return np.array(
            [self.normalize_synoptic(row[v].values[0], v) for v in self.synoptic_vars],
            dtype=np.float32,
        )

    def __len__(self):
        return len(self.perfiles_validos)

    def __getitem__(self, idx):
        fecha    = self.perfiles_validos[idx]
        perfil   = self.df[self.df['Fecha'] == fecha].copy().sort_values('HGHT')

        alturas_orig = perfil['HGHT'].values
        temp_orig    = perfil['TEMP'].values
        mixr_orig    = perfil['MIXR'].values
        sknt_orig    = perfil['SKNT'].values

        # Interpolate to the uniform 311-level grid when needed
        if len(alturas_orig) != self.num_alturas or not np.allclose(alturas_orig, self.alturas, atol=1):
            def _interp(y):
                f = interp1d(alturas_orig, y, kind='linear',
                             fill_value='extrapolate', bounds_error=False)
                return f(self.alturas)

            temp = np.clip(_interp(temp_orig), self.stats['TEMP']['min'], self.stats['TEMP']['max'])
            mixr = np.clip(_interp(mixr_orig), self.stats['MIXR']['min'], self.stats['MIXR']['max'])
            sknt = np.clip(_interp(sknt_orig), self.stats['SKNT']['min'], self.stats['SKNT']['max'])
        else:
            temp, mixr, sknt = temp_orig, mixr_orig, sknt_orig

        perfil_tensor = torch.tensor(
            np.stack([
                self.normalize(temp, 'TEMP'),
                self.normalize(mixr, 'MIXR'),
                self.normalize(sknt, 'SKNT'),
            ], axis=0),
            dtype=torch.float32,
        )

        synoptic_conditions = self.get_synoptic_conditions(fecha)
        if synoptic_conditions is None:
            synoptic_conditions = np.zeros(len(self.synoptic_vars), dtype=np.float32)

        return (
            perfil_tensor,
            torch.tensor(0, dtype=torch.long),
            torch.tensor(synoptic_conditions, dtype=torch.float32),
        )


def get_dataset(dataset_name, data_dir=None, metadata=None, use_synoptic=True,
                root=None, split='train',
                profiles_csv=None, synoptic_csv=None):
    """
    Instantiate the dataset by name.

    Args:
        dataset_name  : one of 'perfiles_synoptic', 'vertical_profiles', etc.
        data_dir      : directory containing the CSV files (overrides defaults)
        profiles_csv  : explicit path to sondeos_interpolados.csv
        synoptic_csv  : explicit path to era5_boxtfe.csv
        split         : 'train' (80 %), 'test' (20 %), or 'all'
    """
    if profiles_csv is None:
        profiles_csv = (
            os.path.join(data_dir, _DEFAULT_PROFILES_CSV)
            if data_dir else _DEFAULT_PROFILES_CSV
        )
    if synoptic_csv is None:
        synoptic_csv = (
            os.path.join(data_dir, _DEFAULT_SYNOPTIC_CSV)
            if data_dir else _DEFAULT_SYNOPTIC_CSV
        )

    if dataset_name in ["perfiles_meteo", "vertical_profiles", "perfiles_synoptic"]:
        if use_synoptic:
            return PerfilesSynopticDataset(profiles_csv, synoptic_csv, split=split)
        else:
            from data_perfiles import PerfilesMeteoDataset
            return PerfilesMeteoDataset(profiles_csv)

    if dataset_name == "perfiles_synoptic_test":
        return PerfilesSynopticDataset(profiles_csv, synoptic_csv, split='test')

    raise ValueError(f"Unsupported dataset: {dataset_name}")


# needed by get_dataset
import os


def fix_legacy_dict(d):
    """Pass-through for checkpoint compatibility."""
    return d
