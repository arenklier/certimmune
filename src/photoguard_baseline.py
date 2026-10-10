"""PhotoGuard baseline (Salman et al. 2023, "Raising the Cost of Malicious
AI-Powered Image Editing") for comparison against our EOT-over-Theta delta.

v2 (2026-07-23): ported from the OFFICIAL repo (github.com/MadryLab/photoguard,
notebooks/demo_simple_attack_img2img.ipynb), replacing our earlier simplified
analog. The real "Simple photo-guarding (Encoder Attack)" recipe:
  - loss = ||VAE.encode(x).latent_dist.mean||  (the LATENT'S OWN NORM, not
    distance from the clean latent -- our v1 analog attacked a different,
    self-invented objective; this is the actual published one)
  - gradient DESCENT on that norm (drives the latent toward a degenerate,
    near-zero-norm representation, not "away from clean" in any direction)
  - random init within the epsilon-ball, decaying step size, up to 1000 PGD
    iterations in the paper's own strongest demo (vs. our earlier 30 -- we
    keep more iterations here since a VAE-encoder-only forward/backward is
    cheap, unlike our full diffusion-chain delta optimization)
Still STATIC: no purification-awareness, matching how PhotoGuard actually
works in the literature (this was true of v1 too, unchanged).
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image

from common import DEVICE, load_sd_pipeline, load_face_app, get_fixed_timesteps
from delta_optimize import pil_to_pixel_tensor, pixel_to_vae_input, evaluate_grid


def optimize_photoguard(pipe, clean_pixel, epsilon, steps, step_size, seed=0):
    """Faithful port of the official pgd() from MadryLab/photoguard's
    demo_simple_attack_img2img.ipynb. Official recipe works in [-1,1] pixel
    space (eps=0.06, step_size=0.02); we keep our [0,1]-space delta
    convention and the SAME epsilon budget used by every other method in
    this comparison (e.g. 8/255) for a fair, apples-to-apples evaluation,
    while replicating their exact mechanics: random init in the epsilon-ball,
    a decaying step size, and critically the loss itself -- ||latent.mean||
    (the latent's OWN norm), NOT distance from the clean latent (that was
    our earlier, non-faithful v1 analog's self-invented objective)."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    delta = (torch.rand(clean_pixel.shape, generator=gen, device=DEVICE) * 2 * epsilon - epsilon)
    delta = delta.detach().requires_grad_(True)

    for step in range(steps):
        actual_step_size = step_size - (step_size - step_size / 100) / steps * step  # official decay

        x_adv = (clean_pixel + delta).clamp(0, 1)
        latent = pipe.vae.encode(pixel_to_vae_input(x_adv)).latent_dist.mean
        loss = latent.float().norm()  # official: latent's own norm, gradient DESCENT drives it toward 0
        loss.backward()

        with torch.no_grad():
            delta -= actual_step_size * delta.grad.sign()  # official direction: descent, not ascent
            delta.clamp_(-epsilon, epsilon)
        delta.grad = None

        if step % 100 == 0 or step == steps - 1:
            print(f"    step {step:4d}: latent norm = {loss.item():.4f}  step_size={actual_step_size:.5f}")

    return delta.detach()


def main():
    ap = argparse.ArgumentParser()
    base = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--identity", default=os.path.join(base, "..", "data", "celebahq_sample", "identity_00.png"))
    ap.add_argument("--epsilon", type=float, default=8 / 255)
    ap.add_argument("--steps", type=int, default=1000, help="official demo uses 1000 for its strongest result")
    ap.add_argument("--step-size", type=float, default=0.01, help="official 0.02 in [-1,1]-space == 0.01 in [0,1]")
    ap.add_argument("--eval-grid", type=int, nargs="+", default=[100, 150, 200, 250, 300, 350, 400])
    ap.add_argument("--eval-seeds", type=int, default=5)
    ap.add_argument("--out-dir", default=os.path.join(base, "..", "results"))
    ap.add_argument("--seed", type=int, default=0,
                    help="initialisation seed for delta; 0 reproduces every file already on disk")
    ap.add_argument("--tag", default="", help="suffix for output filenames, e.g. _01 (default: none, keeps original names)")
    args = ap.parse_args()

    pipe = load_sd_pipeline()
    face_app = load_face_app()
    fixed_timesteps = get_fixed_timesteps(pipe, 50)
    text_input = pipe.tokenizer([""], padding="max_length", max_length=pipe.tokenizer.model_max_length,
                                 truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        text_emb = pipe.text_encoder(text_input.input_ids)[0]

    clean_img = Image.open(args.identity).convert("RGB")
    clean_pixel = pil_to_pixel_tensor(clean_img)

    print(f"=== PhotoGuard (OFFICIAL recipe port): VAE-latent-norm encoder attack (epsilon={args.epsilon:.4f}) ===")
    delta = optimize_photoguard(pipe, clean_pixel, args.epsilon, args.steps, args.step_size, seed=args.seed)
    adv_pixel = (clean_pixel + delta).clamp(0, 1)
    torch.cuda.empty_cache()

    print(f"\n=== Evaluating across the FULL Theta grid {args.eval_grid} (n_seeds={args.eval_seeds}) ===")
    grid_unprotected = evaluate_grid(pipe, face_app, clean_img, clean_pixel, args.eval_grid,
                                      fixed_timesteps, text_emb, args.eval_seeds)
    grid_pg = evaluate_grid(pipe, face_app, clean_img, adv_pixel, args.eval_grid,
                             fixed_timesteps, text_emb, args.eval_seeds)

    print(f"\n{'t*':>6} {'unprotected':>12} {'photoguard':>12}")
    for t in args.eval_grid:
        print(f"{t:>6} {grid_unprotected[t]:>12.4f} {grid_pg[t]:>12.4f}")
    worst_unprotected = max(grid_unprotected.values())
    worst_pg = max(grid_pg.values())
    print(f"\nworst-case over Theta: unprotected={worst_unprotected:.4f}  photoguard={worst_pg:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    img = Image.fromarray((adv_pixel[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))
    img.save(os.path.join(args.out_dir, f"photoguard_baseline{args.tag}_immunized.png"))
    import json
    with open(os.path.join(args.out_dir, f"photoguard_baseline{args.tag}_grid.json"), "w") as f:
        json.dump({"unprotected": grid_unprotected, "photoguard": grid_pg,
                   "worst_case": {"unprotected": worst_unprotected, "photoguard": worst_pg}}, f, indent=2)
    print(f"\nSaved to {args.out_dir}")


if __name__ == "__main__":
    main()
