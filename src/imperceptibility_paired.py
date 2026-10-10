"""Is the imperceptibility gap paired-significant, or just a difference of means?

Means can separate while per-identity pairs do not, so the comparison is redone
on identities where both methods have an artifact, with the same one-sided
Wilcoxon signed-rank test the paper uses for its certificate comparison.
"""
import glob
import os
import re

import lpips
import numpy as np
import torch
from PIL import Image
from scipy.stats import wilcoxon
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

ROOT = "/mnt/data/certimmune"
R = os.path.join(ROOT, "results")
net = lpips.LPIPS(net="alex", verbose=False).cuda().eval()


def clean(i):
    p = os.path.join(ROOT, "data/celebahq_sample/identity_%s.png" % i)
    return np.asarray(Image.open(p).convert("RGB").resize((512, 512))).astype(np.uint8) \
        if os.path.exists(p) else None


def metrics(c, f):
    a = np.asarray(Image.open(f).convert("RGB")).astype(np.uint8)
    if a.shape != c.shape:
        return None

    def t(x):
        return (torch.from_numpy(x).float().permute(2, 0, 1)[None] / 127.5 - 1.0).cuda()
    with torch.no_grad():
        l = float(net(t(c), t(a)).item())
    return psnr(c, a, data_range=255), ssim(c, a, channel_axis=2, data_range=255), l


def collect(pat, rx):
    out = {}
    for f in sorted(glob.glob(pat)):
        m = re.search(rx, os.path.basename(f))
        if not m:
            continue
        i = m.group(1)
        c = clean(i)
        if c is None:
            continue
        v = metrics(c, f)
        if v:
            out[i] = v
    return out


ours = collect(os.path.join(R, "immunized_identity_*_eps8.png"), r"identity_(\d\d)_eps8\.png")
fast = collect(os.path.join(R, "delta_immunized_fasteot80_*.png"), r"fasteot80_(\d\d)\.png")
pg = collect(os.path.join(R, "photoguard_baseline_*_immunized.png"), r"baseline_(\d\d)_immunized\.png")

for label, a, b in (("Ours (EOT) vs PhotoGuard", ours, pg),
                    ("fast-EOT vs PhotoGuard", fast, pg)):
    common = sorted(set(a) & set(b))
    print("\n%s  (%d paired identities)" % (label, len(common)))
    for k, name, better in ((0, "PSNR ", "greater"), (1, "SSIM ", "greater"), (2, "LPIPS", "less")):
        x = np.array([a[i][k] for i in common])
        y = np.array([b[i][k] for i in common])
        stat, p = wilcoxon(x, y, alternative=better)
        wins = int((x > y).sum()) if better == "greater" else int((x < y).sum())
        print("   %-6s ours %7.4f  photoguard %7.4f   better on %2d/%2d   one-sided p=%.2e"
              % (name, x.mean(), y.mean(), wins, len(common), p))
