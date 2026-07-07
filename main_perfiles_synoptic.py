"""
Training script for the synoptically-conditioned diffusion model.
"""

import argparse
import copy
import math
import numpy as np
import os
import torch
import torch.distributed as dist
from time import time
from tqdm import tqdm
from easydict import EasyDict

import unets_perfiles_synoptic
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from data_perfiles_synoptic import get_metadata, get_dataset, fix_legacy_dict

unsqueeze1x = lambda x: x[:, None, None]


class GaussianDiffusion:
    """Gaussian diffusion process for 1D sequences."""

    def __init__(self, timesteps=1000, device="cuda:0"):
        self.timesteps = timesteps
        self.device = device
        self.alpha_bar_scheduler = (
            lambda t: math.cos((t / self.timesteps + 0.008) / 1.008 * math.pi / 2) ** 2
        )
        self.scalars = self.get_all_scalars(
            self.alpha_bar_scheduler, self.timesteps, self.device
        )
        self.clamp_x0 = lambda x: x.clamp(-1, 1)
        self.get_x0_from_xt_eps = lambda xt, eps, t, scalars: (
            self.clamp_x0(
                1
                / unsqueeze1x(scalars.alpha_bar[t].sqrt())
                * (xt - unsqueeze1x((1 - scalars.alpha_bar[t]).sqrt()) * eps)
            )
        )
        self.get_pred_mean_from_x0_xt = (
            lambda xt, x0, t, scalars: unsqueeze1x(
                (scalars.alpha_bar[t].sqrt() * scalars.beta[t])
                / ((1 - scalars.alpha_bar[t]) * scalars.alpha[t].sqrt())
            )
            * x0
            + unsqueeze1x(
                (scalars.alpha[t] - scalars.alpha_bar[t])
                / ((1 - scalars.alpha_bar[t]) * scalars.alpha[t].sqrt())
            )
            * xt
        )

    def get_all_scalars(self, alpha_bar_scheduler, timesteps, device, betas=None):
        all_scalars = {}
        if betas is None:
            all_scalars["beta"] = torch.from_numpy(
                np.array(
                    [
                        min(
                            1 - alpha_bar_scheduler(t + 1) / alpha_bar_scheduler(t),
                            0.999,
                        )
                        for t in range(timesteps)
                    ]
                )
            ).to(device)
        else:
            all_scalars["beta"] = betas
        all_scalars["beta_log"] = torch.log(all_scalars["beta"])
        all_scalars["alpha"] = 1 - all_scalars["beta"]
        all_scalars["alpha_bar"] = torch.cumprod(all_scalars["alpha"], dim=0)
        all_scalars["beta_tilde"] = (
            all_scalars["beta"][1:]
            * (1 - all_scalars["alpha_bar"][:-1])
            / (1 - all_scalars["alpha_bar"][1:])
        )
        all_scalars["beta_tilde"] = torch.cat(
            [all_scalars["beta_tilde"][0:1], all_scalars["beta_tilde"]]
        )
        all_scalars["beta_tilde_log"] = torch.log(all_scalars["beta_tilde"])
        return EasyDict(dict([(k, v.float()) for (k, v) in all_scalars.items()]))

    def sample_from_forward_process(self, x0, t):
        """Single forward-process step: adds noise at level t."""
        eps = torch.randn_like(x0)
        xt = (
            unsqueeze1x(self.scalars.alpha_bar[t].sqrt()) * x0
            + unsqueeze1x((1 - self.scalars.alpha_bar[t]).sqrt()) * eps
        )
        return xt.float(), eps

    def sample_from_reverse_process(
        self, model, xT, timesteps=None, model_kwargs={}, ddim=False
    ):
        """Iterative reverse-process sampling."""
        model.eval()
        final = xT

        timesteps = timesteps or self.timesteps
        new_timesteps = np.linspace(
            0, self.timesteps - 1, num=timesteps, endpoint=True, dtype=int
        )
        alpha_bar = self.scalars["alpha_bar"][new_timesteps]
        new_betas = 1 - (
            alpha_bar / torch.nn.functional.pad(alpha_bar, [1, 0], value=1.0)[:-1]
        )
        scalars = self.get_all_scalars(
            self.alpha_bar_scheduler, timesteps, self.device, new_betas
        )

        for i, t in zip(np.arange(timesteps)[::-1], new_timesteps[::-1]):
            with torch.no_grad():
                current_t = torch.tensor([t] * len(final), device=final.device)
                current_sub_t = torch.tensor([i] * len(final), device=final.device)
                pred_epsilon = model(final, current_t, **model_kwargs)
                pred_x0 = self.get_x0_from_xt_eps(
                    final, pred_epsilon, current_sub_t, scalars
                )
                pred_mean = self.get_pred_mean_from_x0_xt(
                    final, pred_x0, current_sub_t, scalars
                )
                if i == 0:
                    final = pred_mean
                else:
                    if ddim:
                        alpha_ratio = (
                            scalars.alpha_bar[current_sub_t - 1].sqrt()
                            / scalars.alpha_bar[current_sub_t].sqrt()
                        )
                        alpha_complement = (
                            1 - scalars.alpha_bar[current_sub_t - 1].sqrt()
                            / scalars.alpha_bar[current_sub_t].sqrt()
                        )
                        while len(alpha_ratio.shape) < len(final.shape):
                            alpha_ratio = alpha_ratio[:, None]
                            alpha_complement = alpha_complement[:, None]
                        final = alpha_ratio * final + alpha_complement * pred_x0
                    else:
                        noise = torch.randn_like(final)
                        beta_tilde_sqrt = scalars.beta_tilde[current_sub_t].sqrt()
                        while len(beta_tilde_sqrt.shape) < len(final.shape):
                            beta_tilde_sqrt = beta_tilde_sqrt[:, None]
                        final = pred_mean + beta_tilde_sqrt * noise
                final = final.detach()
        return final


class loss_logger:
    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.loss = []
        self.start_time = time()
        self.ema_loss = None
        self.ema_w = 0.9

    def log(self, v, display=False):
        self.loss.append(v)
        if self.ema_loss is None:
            self.ema_loss = v
        else:
            self.ema_loss = self.ema_w * self.ema_loss + (1 - self.ema_w) * v
        if display:
            print(
                f"Steps: {len(self.loss)}/{self.max_steps} \t loss (ema): {self.ema_loss:.3f} "
                + f"\t Time elapsed: {(time() - self.start_time)/3600:.3f} hr"
            )


def train_one_epoch(
    model,
    dataloader,
    diffusion,
    optimizer,
    logger,
    lrs,
    args,
    grad_clip=1.0,
):
    model.train()
    for step, batch_data in enumerate(dataloader):
        profiles, labels, synoptic = batch_data
        assert (profiles.max().item() <= 1.01) and (-1.01 <= profiles.min().item())

        profiles = profiles.to(args.device)
        synoptic = synoptic.to(args.device)

        t = torch.randint(diffusion.timesteps, (len(profiles),), dtype=torch.int64).to(
            args.device
        )
        xt, eps = diffusion.sample_from_forward_process(profiles, t)
        pred_eps = model(xt, t, y=None, synoptic=synoptic)
        loss = ((pred_eps - eps) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if lrs is not None:
            lrs.step()

        if args.local_rank == 0:
            new_dict = model.state_dict()
            for (k, v) in args.ema_dict.items():
                args.ema_dict[k] = (
                    args.ema_w * args.ema_dict[k] + (1 - args.ema_w) * new_dict[k]
                )
            logger.log(loss.item(), display=not step % 100)

    return logger.ema_loss


def sample_N_profiles(
    N,
    model,
    diffusion,
    xT=None,
    sampling_steps=250,
    batch_size=64,
    num_channels=3,
    profile_length=311,
    synoptic_conditions=None,
    args=None,
):
    """Generate N profiles from the model."""
    samples, num_samples = [], 0

    if torch.cuda.device_count() > 1 and torch.distributed.is_initialized():
        num_processes, group = dist.get_world_size(), dist.group.WORLD
    else:
        num_processes, group = 1, None

    with tqdm(total=math.ceil(N / (batch_size * num_processes))) as pbar:
        while num_samples < N:
            if xT is None:
                xT = (
                    torch.randn(batch_size, num_channels, profile_length)
                    .float()
                    .to(args.device)
                )
            model_kwargs = {}
            if synoptic_conditions is not None:
                if synoptic_conditions.shape[0] == 1:
                    synoptic_batch = synoptic_conditions.repeat(len(xT), 1).to(args.device)
                else:
                    synoptic_batch = synoptic_conditions[:len(xT)].to(args.device)
                model_kwargs['synoptic'] = synoptic_batch

            gen_profiles = diffusion.sample_from_reverse_process(
                model, xT, sampling_steps, model_kwargs, args.ddim
            )

            if num_processes > 1:
                samples_list = [torch.zeros_like(gen_profiles) for _ in range(num_processes)]
                dist.all_gather(samples_list, gen_profiles, group)
                samples.append(torch.cat(samples_list).detach().cpu().numpy())
            else:
                samples.append(gen_profiles.detach().cpu().numpy())

            num_samples += len(xT) * num_processes
            pbar.update(1)
            xT = None

    samples = np.concatenate(samples)[:N]
    return samples


@torch.no_grad()
def validate(model, dataloader, diffusion, args):
    """Evaluate the model on a validation/test set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch_data in dataloader:
        profiles, labels, synoptic = batch_data
        profiles = profiles.to(args.device)
        synoptic = synoptic.to(args.device)

        t = torch.randint(diffusion.timesteps, (len(profiles),), dtype=torch.int64).to(args.device)
        xt, eps = diffusion.sample_from_forward_process(profiles, t)
        pred_eps = model(xt, t, y=None, synoptic=synoptic)
        loss = ((pred_eps - eps) ** 2).mean()
        total_loss += loss.item()
        num_batches += 1

    model.train()
    return total_loss / num_batches if num_batches > 0 else float('inf')


def main():
    parser = argparse.ArgumentParser("Diffusion model with synoptic conditioning")
    parser.add_argument("--arch", type=str, default="unet_1d_perfiles")
    parser.add_argument("--synoptic-cond", action="store_true", default=True)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--sampling-steps", type=int, default=250)
    parser.add_argument("--ddim", action="store_true", default=False)
    parser.add_argument("--dataset", type=str, default="perfiles_synoptic")
    parser.add_argument("--data-dir", type=str, default="./")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--ema_w", type=float, default=0.9995)
    parser.add_argument("--pretrained-ckpt", type=str, help="Pretrained model checkpoint")
    parser.add_argument("--sampling-only", action="store_true", default=False)
    parser.add_argument("--num-sampled-profiles", type=int, default=100)
    parser.add_argument("--save-dir", type=str, default="./trained_models_perfiles_synoptic/")
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--seed", default=112233, type=int)
    args = parser.parse_args()

    metadata = get_metadata(args.dataset)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        args.device = "cuda:{}".format(args.local_rank)
        torch.cuda.set_device(args.device)
        print(f"Device: {args.device}")
    else:
        args.device = "cpu"
        print("Device: cpu (no GPU available)")

    torch.manual_seed(args.seed + args.local_rank)
    np.random.seed(args.seed + args.local_rank)
    if args.local_rank == 0:
        print(args)

    model = unets_perfiles_synoptic.__dict__[args.arch](
        image_size=metadata.image_size,
        in_channels=metadata.num_channels,
        out_channels=metadata.num_channels,
        use_synoptic_cond=args.synoptic_cond,
        num_synoptic_vars=metadata.num_synoptic_vars,
        use_attention=True,
        num_heads=4,
    ).to(args.device)

    if args.local_rank == 0:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters : {num_params:,}")
        print(f"Synoptic cond.   : {args.synoptic_cond}")
        print(f"Synoptic vars    : {metadata.num_synoptic_vars}")

    diffusion = GaussianDiffusion(args.diffusion_steps, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    if args.pretrained_ckpt:
        print(f"Loading checkpoint: {args.pretrained_ckpt}")
        d = fix_legacy_dict(torch.load(args.pretrained_ckpt, map_location=args.device))
        model.load_state_dict(d, strict=False)

    ngpus = torch.cuda.device_count()
    if ngpus > 1:
        if args.local_rank == 0:
            print(f"Using {ngpus} GPUs")
        args.batch_size = args.batch_size // ngpus
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    if args.sampling_only:
        print(f"Generating {args.num_sampled_profiles} profiles...")
        sampled_profiles = sample_N_profiles(
            args.num_sampled_profiles, model, diffusion, None,
            args.sampling_steps, args.batch_size,
            metadata.num_channels, metadata.image_size,
            None, args,
        )
        output_file = os.path.join(
            args.save_dir,
            f"generated_profiles_{args.num_sampled_profiles}.npz",
        )
        np.savez(output_file, profiles=sampled_profiles)
        print(f"Saved: {output_file} | shape: {sampled_profiles.shape}")
        return

    train_set = get_dataset(args.dataset, args.data_dir, metadata, use_synoptic=args.synoptic_cond, split='train')
    sampler = DistributedSampler(train_set) if ngpus > 1 else None
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=sampler is None, sampler=sampler,
        num_workers=4, pin_memory=True,
    )
    test_set = get_dataset(args.dataset, args.data_dir, metadata, use_synoptic=args.synoptic_cond, split='test')
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
    )

    if args.local_rank == 0:
        print(f"Train set : {len(train_set)} profiles ({len(train_loader)} batches)")
        print(f"Test set  : {len(test_set)} profiles ({len(test_loader)} batches)")

    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * 5

    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

    logger = loss_logger(len(train_loader) * args.epochs)
    args.ema_dict = copy.deepcopy(model.state_dict())
    os.makedirs(args.save_dir, exist_ok=True)

    _fixed_batch = next(iter(test_loader))
    fixed_synoptic = _fixed_batch[2][:10].to(args.device)
    best_val_loss = float('inf')

    print(f"Starting training ({args.epochs} epochs)...\n")
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        if args.local_rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{args.epochs} (lr={current_lr:.2e})")

        train_one_epoch(model, train_loader, diffusion, optimizer, logger, scheduler, args, grad_clip=1.0)

        if args.local_rank == 0 and (epoch + 1) % 10 == 0:
            val_loss = validate(model, test_loader, diffusion, args)
            print(f"  Validation loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(args.save_dir, "model_best.pt"))
                torch.save(args.ema_dict,      os.path.join(args.save_dir, "model_best_ema.pt"))
                print(f"  New best model saved (val_loss={val_loss:.4f})")

            print("  Generating validation samples...")
            sampled_profiles = sample_N_profiles(
                len(fixed_synoptic), model, diffusion, None,
                args.sampling_steps, args.batch_size,
                metadata.num_channels, metadata.image_size,
                fixed_synoptic, args,
            )
            sample_file = os.path.join(args.save_dir, f"samples_epoch_{epoch+1}.npz")
            np.savez(sample_file, profiles=sampled_profiles)
            print(f"  Samples saved: {sample_file}\n")

        if args.local_rank == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"model_epoch_{epoch+1}.pt"))
            torch.save(args.ema_dict,      os.path.join(args.save_dir, f"model_epoch_{epoch+1}_ema.pt"))

    if args.local_rank == 0:
        print(f"Training complete. Best val_loss: {best_val_loss:.4f}")
        print(f"Checkpoints saved in: {args.save_dir}")


if __name__ == "__main__":
    main()
