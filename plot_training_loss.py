"""
Plots training and validation loss curves from training_v2.log.
Produces a two-panel figure: (top) loss curves, (bottom) learning-rate schedule.
"""

import re
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

LOG_FILE = os.path.join(os.path.dirname(__file__), 'training_v2.log')
OUTPUT   = os.path.join(os.path.dirname(__file__), 'training_loss.png')

# ── Parse log ────────────────────────────────────────────────────────────────
step_re  = re.compile(r'Steps:\s+(\d+)/\d+\s+loss \(ema\):\s+([\d.]+)')
epoch_re = re.compile(r'Época\s+(\d+)/\d+\s+\(lr=([\d.e+-]+)\)')
val_re   = re.compile(r'Validation loss:\s+([\d.]+)')

train_steps, train_loss = [], []
epochs_lr,   lr_vals    = [], []
val_epochs,  val_loss   = [], []

current_epoch = None

with open(LOG_FILE) as fh:
    for line in fh:
        m = epoch_re.search(line)
        if m:
            current_epoch = int(m.group(1))
            epochs_lr.append(current_epoch)
            lr_vals.append(float(m.group(2)))
            continue

        m = step_re.search(line)
        if m and current_epoch is not None:
            train_steps.append(current_epoch)
            train_loss.append(float(m.group(2)))
            continue

        m = val_re.search(line)
        if m and current_epoch is not None:
            val_epochs.append(current_epoch)
            val_loss.append(float(m.group(1)))

train_steps = np.array(train_steps)
train_loss  = np.array(train_loss)
val_epochs  = np.array(val_epochs)
val_loss    = np.array(val_loss)
epochs_lr   = np.array(epochs_lr)
lr_vals     = np.array(lr_vals)

print(f'Train entries : {len(train_steps)}')
print(f'Val   entries : {len(val_epochs)}')
print(f'LR    entries : {len(epochs_lr)}')
print(f'Train loss range : {train_loss.min():.4f} – {train_loss.max():.4f}')
print(f'Val   loss range : {val_loss.min():.4f} – {val_loss.max():.4f}')

# ── Smooth train loss (running mean over 10 epochs) ───────────────────────────
def smooth(x, w=10):
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode='valid')

train_smooth = smooth(train_loss, w=10)
train_smooth_x = train_steps[9:]  # aligned after convolution

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax_loss, ax_lr) = plt.subplots(
    2, 1, figsize=(9, 6.5),
    gridspec_kw={'height_ratios': [3, 1]},
    sharex=True,
)
fig.subplots_adjust(hspace=0.08)

# ── Top panel: loss ───────────────────────────────────────────────────────────
ax_loss.plot(train_steps, train_loss,
             color='#aec6e8', linewidth=0.8, alpha=0.55, label='Train loss (EMA)')
ax_loss.plot(train_smooth_x, train_smooth,
             color='#2980b9', linewidth=2.0, label='Train loss (smoothed, $w{=}10$)')
ax_loss.plot(val_epochs, val_loss,
             color='#e74c3c', linewidth=1.8, marker='o',
             markersize=4, label='Validation loss')

ax_loss.set_ylabel('MSE Loss', fontsize=11)
ax_loss.set_yscale('log')
ax_loss.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax_loss.yaxis.set_minor_formatter(mticker.NullFormatter())
ax_loss.grid(True, which='major', alpha=0.3, linestyle='--')
ax_loss.grid(True, which='minor', alpha=0.12, linestyle=':')
ax_loss.spines['top'].set_visible(False)
ax_loss.spines['right'].set_visible(False)
ax_loss.tick_params(labelsize=9)
ax_loss.legend(fontsize=9, loc='upper right', framealpha=0.9)

# Annotate best val loss
best_idx  = np.argmin(val_loss)
best_ep   = val_epochs[best_idx]
best_val  = val_loss[best_idx]
ax_loss.annotate(
    f'Best val: {best_val:.4f}\n(epoch {best_ep})',
    xy=(best_ep, best_val),
    xytext=(best_ep - 120, best_val * 2.5),
    fontsize=8,
    arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.0),
    color='#e74c3c',
)

# ── Bottom panel: learning rate ───────────────────────────────────────────────
ax_lr.plot(epochs_lr, lr_vals * 1e4,
           color='#27ae60', linewidth=1.5)
ax_lr.set_ylabel(r'LR $(\times 10^{-4})$', fontsize=10)
ax_lr.set_xlabel('Epoch', fontsize=11)
ax_lr.set_xlim(1, train_steps[-1] + 5)
ax_lr.set_ylim(bottom=0)
ax_lr.grid(True, alpha=0.3, linestyle='--')
ax_lr.spines['top'].set_visible(False)
ax_lr.spines['right'].set_visible(False)
ax_lr.tick_params(labelsize=9)

plt.savefig(OUTPUT, dpi=150, bbox_inches='tight')
print(f'Figure saved: {OUTPUT}')
plt.close(fig)