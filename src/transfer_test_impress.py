"""Empirical (UNCERTIFIED) transfer test: does our EOT-delta perturbation
retain any protective effect against IMPRESS (Cao, Li, Wang, Jia, Li, Chen,
NeurIPS 2023), a published purifier NOT in this paper's certified family
A = {DiffPure, GrIDPure}?

This is explicitly NOT a certification -- no grid, no Clopper-Pearson bound,
no Lipschitz argument. It is a single-point empirical probe intended only to
soften (not remove) the paper's closed-world-purifier-family limitation
(Sec 9, item 7): evidence that the perturbation is not *maximally* overfit
to the two certified architectures, without claiming any formal guarantee
against IMPRESS or any other purifier outside A.

Uses IMPRESS's own official code (github.com/AAAAAAsuka/Impress), specifically
the impress() function from impress.py, which is model-architecture-agnostic
(any VAE with a .forward()->.sample interface) -- applied here to THIS
project's own SD1.5 VAE (matching how DiffPure/GrIDPure are applied
throughout the rest of the paper), not IMPRESS's own SD2.1-inpainting demo
pipeline. Hyperparameters (pur_eps=0.1, pur_iters=100, pur_lr=0.01,
pur_alpha=0.1, pur_noise=0.1) match IMPRESS's own released reproduction
script's CLI defaults (pg_mask_pur_helen.py), not re-tuned by us.
"""

import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party", "Impress"))
from impress import impress  # noqa: E402

from common import DEVICE, load_sd_pipeline, load_face_app, arcface_embed, cosine_sim  # noqa: E402
from delta_optimize import pil_to_pixel_tensor, pixel_to_vae_input  # noqa: E402


def vae_forward_wrapper(vae):
    """impress() calls model(X_p).sample -- diffusers' AutoencoderKL.forward()
    returns a DecoderOutput via full encode+decode when called directly, but
    the *class* __call__ signature differs; use a thin lambda that does a
    full encode-mean/decode roundtrip matching what 'reconstruct this image'
    means for a VAE, consistent with impress()'s own consistency-loss intent
    (X_p should be a fixed point of the VAE's own encode-decode map)."""
    def _fwd(x):
        posterior = vae.encode(x).latent_dist
        rec = vae.decode(posterior.mean).sample
        class _Out:
            pass
        out = _Out()
        out.sample = rec
        return out
    return _fwd


def run_impress_purify(vae, x_01, pur_eps=0.1, pur_iters=100, pur_lr=0.01, pur_alpha=0.1, pur_noise=0.1):
    x_vae = pixel_to_vae_input(x_01).float()  # impress()'s Adam optimizer needs fp32, not fp16
    vae_fp32 = vae.float()
    model_fn = vae_forward_wrapper(vae_fp32)
    x_p = impress(x_vae, model=model_fn, eps=pur_eps, iters=pur_iters, lr=pur_lr,
                  clamp_min=-1, clamp_max=1, pur_alpha=pur_alpha, noise=pur_noise)
    x_p_01 = (x_p / 2 + 0.5).clamp(0, 1)
    return x_p_01


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base, "..", "results")

    pipe = load_sd_pipeline()
    face_app = load_face_app()

    clean_img = Image.open(os.path.join(base, "..", "data", "celebahq_sample", "identity_00.png")).convert("RGB")
    clean_pixel = pil_to_pixel_tensor(clean_img)
    protected_img = Image.open(os.path.join(results_dir, "delta_immunized_eot.png")).convert("RGB")
    protected_pixel = pil_to_pixel_tensor(protected_img)

    print("=== IMPRESS-purifying UNPROTECTED (clean) image ===")
    torch.manual_seed(0)
    purified_clean = run_impress_purify(pipe.vae, clean_pixel)

    print("=== IMPRESS-purifying our EOT-delta PROTECTED image ===")
    torch.manual_seed(0)
    purified_protected = run_impress_purify(pipe.vae, protected_pixel)

    with torch.no_grad():
        clean_emb = arcface_embed(face_app, clean_img)
        sim_unprotected_purified = cosine_sim(
            arcface_embed(face_app, Image.fromarray(
                (purified_clean[0].permute(1, 2, 0).float().cpu().numpy() * 255).round().astype(np.uint8))),
            clean_emb)
        sim_protected_purified = cosine_sim(
            arcface_embed(face_app, Image.fromarray(
                (purified_protected[0].permute(1, 2, 0).float().cpu().numpy() * 255).round().astype(np.uint8))),
            clean_emb)

    print(f"\nArcFace similarity to clean identity_00, AFTER IMPRESS purification:")
    print(f"  unprotected input -> IMPRESS -> sim = {sim_unprotected_purified:.4f}")
    print(f"  our EOT-delta input -> IMPRESS -> sim = {sim_protected_purified:.4f}")

    os.makedirs(results_dir, exist_ok=True)
    Image.fromarray((purified_clean[0].permute(1, 2, 0).float().cpu().numpy() * 255)
                     .round().astype(np.uint8)).save(os.path.join(results_dir, "impress_purified_unprotected.png"))
    Image.fromarray((purified_protected[0].permute(1, 2, 0).float().cpu().numpy() * 255)
                     .round().astype(np.uint8)).save(os.path.join(results_dir, "impress_purified_eotdelta.png"))
    with open(os.path.join(results_dir, "transfer_test_impress.json"), "w") as f:
        json.dump({"sim_unprotected_after_impress": float(sim_unprotected_purified),
                   "sim_eotdelta_after_impress": float(sim_protected_purified),
                   "note": "UNCERTIFIED empirical probe, not a certification -- see docstring"}, f, indent=2)
    print(f"\nSaved to {results_dir}")


if __name__ == "__main__":
    main()
