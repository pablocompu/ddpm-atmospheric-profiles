#!/usr/bin/env python3
"""
Example training script for the synoptically-conditioned diffusion model.

Basic usage:
    python train_synoptic_example.py

Custom parameters:
    python train_synoptic_example.py --epochs 100 --batch-size 64 --lr 0.0002

Synoptic variables used (23):
    Pressure / temperature : msl, d2m, t2m, t850, t700, t500
    Humidity               : q850, q700, q500
    Zonal wind             : u10, u850, u700, u500
    Meridional wind        : v10, v850, v700, v500
    Vertical velocity      : w850, w700, w500
    Geopotential           : z850, z700, z500
"""

import sys
import os

if __name__ == "__main__":
    print("=" * 70)
    print("  DIFFUSION MODEL TRAINING WITH SYNOPTIC CONDITIONING")
    print("=" * 70)
    print()
    print("  MODEL ARCHITECTURE:")
    print("    1D U-Net with 3 resolution levels (311 -> 156 -> 78 -> 39)")
    print("    Time embedding     : 256 dimensions (sinusoidal)")
    print("    Synoptic embedding : 256 dimensions (3-layer MLP)")
    print("    Combination        : additive (time_emb + synoptic_emb)")
    print()
    print("  SYNOPTIC CONDITIONS:")
    print("    23 meteorological variables from ERA5")
    print("    Normalisation: z-score (mean=0, std=1)")
    print("    MLP architecture  : 23 -> 512 -> 512 -> 256 dims")
    print()
    print("  DEFAULT HYPERPARAMETERS:")
    print("    Epochs          : 50")
    print("    Batch size      : 32")
    print("    Learning rate   : 0.0001")
    print("    Diffusion steps : 1000")
    print("    Sampling steps  : 250")
    print()
    print("  OUTPUT:")
    print("    Checkpoints saved to: ./trained_models_perfiles_synoptic/")
    print("    Samples generated every 10 epochs")
    print()
    print("=" * 70)
    print()

    default_args = [
        "--arch", "unet_1d_perfiles",
        "--dataset", "perfiles_synoptic",
        "--synoptic-cond",
        "--epochs", "50",
        "--batch-size", "32",
        "--lr", "0.0001",
        "--diffusion-steps", "1000",
        "--sampling-steps", "250",
        "--save-dir", "./trained_models_perfiles_synoptic/",
    ]

    args = default_args + sys.argv[1:]
    sys.argv = [sys.argv[0]] + args

    from main_perfiles_synoptic import main

    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError during training: {e}")
        raise
