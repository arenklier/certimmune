"""Imperceptibility of the published perturbations: PSNR, SSIM, LPIPS, and the budget.

The clean images are 1024x1024 and the perturbed ones 512x512, so the clean side
is resized with the pipeline's own call (delta_optimize.pil_to_pixel_tensor uses
Image.resize((512,512)) with PIL's default resample) before anything is measured;
resizing any other way would measure the resampler, not the perturbation.

The maximum absolute channel difference is reported alongside, in 1/255 units,
as a direct check that the shipped artifacts respect the epsilon they claim.
"""
import glob
import json
import os
import sys

import lpips
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

ROOT = "/mnt/data/certimmune"
CLEAN = os.path.join(ROOT, "data/celebahq_sample/identity_%s.png")
R = os.path.join(ROOT, "results")

net = lpips.LPIPS(net="alex", verbose=False).cuda().eval()


def load_clean(ident):
    p = CLEAN % ident
    if not os.path.exists(p):
        return None
    return np.asarray(Image.open(p).convert("RGB").resize((512, 512))).astype(np.uint8)


def load(p):
    return np.asarray(Image.open(p).convert("RGB")).astype(np.uint8)


def lp(a, b):
    def t(x):
        x = torch.from_numpy(x).float().permute(2, 0, 1)[None] / 127.5 - 1.0
        return x.cuda()
    with torch.no_grad():
        return float(net(t(a), t(b)).item())


METHODS = {
    "Ours (EOT, 30 steps)": os.path.join(R, "immunized_identity_%s_eps8.png"),
    "Ours (fast-EOT, 80 steps)": os.path.join(R, "delta_immunized_fasteot80_%s.png"),
    "PhotoGuard (faithful)": os.path.join(R, "photoguard_baseline_%s_immunized.png"),
}

rows = {}
for name, pat in METHODS.items():
    vals = []
    for f in sorted(glob.glob(pat % "*")):
        base = os.path.basename(f)
        ident = None
        for tok in base.replace(".png", "").split("_"):
            if tok.isdigit() and len(tok) == 2:
                ident = tok
        if ident is None:
            continue
        c = load_clean(ident)
        if c is None:
            continue
        a = load(f)
        if a.shape != c.shape:
            continue
        d = np.abs(a.astype(np.int16) - c.astype(np.int16)).max()
        vals.append((psnr(c, a, data_range=255),
                     ssim(c, a, channel_axis=2, data_range=255),
                     lp(c, a), d))
    if vals:
        v = np.array([x[:3] for x in vals])
        dmax = max(x[3] for x in vals)
        rows[name] = dict(n=len(vals),
                          psnr=(v[:, 0].mean(), v[:, 0].std()),
                          ssim=(v[:, 1].mean(), v[:, 1].std()),
                          lpips=(v[:, 2].mean(), v[:, 2].std()),
                          linf_max=int(dmax))

print("%-28s %4s %16s %16s %16s %8s" % ("method", "n", "PSNR (dB)", "SSIM", "LPIPS", "max|d|"))
for k, r in rows.items():
    print("%-28s %4d %8.2f+-%.2f %8.4f+-%.4f %8.4f+-%.4f %8d"
          % (k, r["n"], r["psnr"][0], r["psnr"][1], r["ssim"][0], r["ssim"][1],
             r["lpips"][0], r["lpips"][1], r["linf_max"]))

print()
print("epsilon sweep on identity_00 (ours, EOT):")
for eps, p in (("4/255", "immunized_identity_00_eps4.png"),
               ("8/255", "immunized_identity_00_eps8_n20.png"),
               ("16/255", "immunized_identity_00_eps16.png")):
    f = os.path.join(R, p)
    if not os.path.exists(f):
        print("   %-7s (missing)" % eps)
        continue
    c, a = load_clean("00"), load(f)
    d = int(np.abs(a.astype(np.int16) - c.astype(np.int16)).max())
    print("   %-7s PSNR %6.2f  SSIM %.4f  LPIPS %.4f  max|d| %d" %
          (eps, psnr(c, a, data_range=255), ssim(c, a, channel_axis=2, data_range=255), lp(c, a), d))

json.dump({k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in r.items()}
           for k, r in rows.items()},
          open(os.path.join(R, "imperceptibility.json"), "w"), indent=1)
print("\nwritten: results/imperceptibility.json")
