"""AntiPure baseline (Yang, Cao, Duan, He, ICCV 2025) for comparison against
our EOT-over-Theta delta.

ANALOG, NOT A FAITHFUL PORT -- unlike PhotoGuard/Anti-DreamBooth, AntiPure's
official code was not publicly available as of 2026-08 (~11 months after the
ICCV 2025 / Sept-2025-arXiv release; verified via WebSearch across arXiv,
CVF openaccess, and GitHub). This is implemented from the paper's stated
equations (arXiv:2509.13922) rather than from released code, and several
details the paper does not give a closed-form for are our own honest,
documented choices -- flagged inline and in the paper's Limitations section.
Treat this the same way the project treats the Glaze baseline: a labeled
analog, not a reproduction to be held to the same fidelity standard as
PhotoGuard/Anti-DreamBooth.

Loss (from the paper, Sec 5.3 + Eq. 9-11, Appendix A/C):
  L_pgd = E_{eps,t}[ L_ddpm + lambda1 * L_fre + lambda2 * L_errt ]
  L_ddpm = || eps_theta(x_t, t) - eps ||^2            (standard noise-pred MSE)
  L_fre  = sigmoid( mean_over_patches( sum_{m,n=s/2}^{s-1} PatchDCT(x0_hat, s)_{m,n} ) )
           where x0_hat = (x_t - sqrt(1-abar_t)*eps_theta(x_t,t)) / sqrt(abar_t)
  L_errt = -|| eps_theta(x_t, t_err) - eps_theta(x_t, t) ||^2
  Optimized via PGD, MAXIMIZING L_pgd (gradient ASCENT; paper Eq. 16:
  x_{k+1} = Proj(x_k + alpha * sign(grad_x L))), lambda1 = lambda2 = 0.5.

Our deviations from the paper (undeterminable from the paper text alone,
without code):
  - The paper's adaptive weights e^{abar_t-1} (on L_fre) and e^{L_errt} (on
    L_errt) are dropped in favor of the plain lambda1/lambda2=0.5 weighting
    the paper also states -- the self-referential exponential terms are not
    fully specified (e.g. no clipping/normalization given) and risk an
    unstable reproduction more than a faithful one.
  - t_err ("an erroneous timestep... at a higher timestep") has no formula
    given; we use t_err = min(t + 300, 999).
  - t is sampled from OUR project's Theta=[100,400] range (matching every
    other baseline's evaluation grid in this project) rather than the
    paper's own t^p=10 GrIDPure-specific schedule, so the comparison is
    apples-to-apples with PhotoGuard/Anti-DreamBooth/ours, not a replication
    of AntiPure's own reported numbers.
  - PatchDCT operates on the SD1.5 latent (4x64x64), not pixel space --
    x0_hat is only algebraically available as a latent from the UNet's own
    prediction equation. Patch size s=8 (8x8 patches over a 64x64 latent, 64
    patches; "bottom-right quarter" = indices [4:8, 4:8] of each patch),
    computed per-channel via a fixed differentiable DCT-II basis matrix and
    averaged across the 4 latent channels.
  - epsilon = 8/255 (this project's shared budget for every baseline),
    not the paper's own eta=16/255.
"""

import argparse
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from common import DEVICE, load_sd_pipeline, load_face_app, get_fixed_timesteps
from delta_optimize import pil_to_pixel_tensor, pixel_to_vae_input, evaluate_grid


def _dct_matrix(n, device, dtype):
    """Standard orthonormal DCT-II basis matrix, n x n."""
    k = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(0)
    basis = torch.cos(math.pi / n * (i + 0.5) * k)
    basis[0, :] *= 1.0 / math.sqrt(2.0)
    basis *= math.sqrt(2.0 / n)
    return basis.to(dtype)


def patch_dct_highfreq_loss(x0_hat, patch=8):
    """L_fre: unfold into patch x patch blocks (per latent channel), 2D
    DCT-II each block, sum the bottom-right quadrant (indices [patch/2:,
    patch/2:]), average over all patches/channels, sigmoid-normalize."""
    b, c, h, w = x0_hat.shape
    assert h % patch == 0 and w % patch == 0
    D = _dct_matrix(patch, x0_hat.device, x0_hat.dtype)
    blocks = x0_hat.unfold(2, patch, patch).unfold(3, patch, patch)  # b,c,H/p,W/p,p,p
    blocks = blocks.reshape(-1, patch, patch)
    dct = torch.einsum("ij,njk,lk->nil", D, blocks, D)  # DCT-II each block
    half = patch // 2
    high = dct[:, half:, half:].sum(dim=(1, 2))
    return torch.sigmoid(high.mean())


def optimize_antipure(pipe, clean_pixel, epsilon, steps, step_size,
                       instance_prompt="a photo of sks person", seed=0,
                       theta_lo=100, theta_hi=400, lambda1=0.5, lambda2=0.5):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    delta = torch.zeros_like(clean_pixel, requires_grad=True)

    text_input = pipe.tokenizer([instance_prompt], padding="max_length",
                                 max_length=pipe.tokenizer.model_max_length,
                                 truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        text_emb = pipe.text_encoder(text_input.input_ids)[0]

    alphas_cumprod = pipe.scheduler.alphas_cumprod.to(DEVICE)
    num_train_timesteps = pipe.scheduler.config.num_train_timesteps

    for step in range(steps):
        x_adv = (clean_pixel + delta).clamp(0, 1)
        latents = pipe.vae.encode(pixel_to_vae_input(x_adv)).latent_dist.sample() * pipe.vae.config.scaling_factor

        noise = torch.randn(latents.shape, generator=gen, device=DEVICE, dtype=latents.dtype)
        t_val = torch.randint(theta_lo, theta_hi + 1, (1,), device=DEVICE, generator=gen).long()
        noisy_latents = pipe.scheduler.add_noise(latents, noise, t_val)

        eps_pred_t = pipe.unet(noisy_latents, t_val, encoder_hidden_states=text_emb).sample
        loss_ddpm = F.mse_loss(eps_pred_t.float(), noise.float())

        abar_t = alphas_cumprod[t_val].view(-1, 1, 1, 1).to(eps_pred_t.dtype)
        x0_hat = (noisy_latents - torch.sqrt(1 - abar_t) * eps_pred_t) / torch.sqrt(abar_t)
        loss_fre = patch_dct_highfreq_loss(x0_hat.float(), patch=8)

        t_err = torch.clamp(t_val + 300, max=num_train_timesteps - 1)
        with torch.no_grad():
            eps_pred_t_detached = eps_pred_t.detach()
        eps_pred_terr = pipe.unet(noisy_latents, t_err, encoder_hidden_states=text_emb).sample
        loss_errt = -F.mse_loss(eps_pred_terr.float(), eps_pred_t_detached.float())

        loss = loss_ddpm + lambda1 * loss_fre + lambda2 * loss_errt  # MAXIMIZE this
        loss.backward()

        with torch.no_grad():
            delta += step_size * delta.grad.sign()  # ascent, per paper Eq. 16
            delta.clamp_(-epsilon, epsilon)
        delta.grad = None

        if step % 10 == 0 or step == steps - 1:
            print(f"    step {step:3d}: L_pgd (maximizing) = {loss.item():.4f}  "
                  f"(ddpm={loss_ddpm.item():.4f} fre={loss_fre.item():.4f} errt={loss_errt.item():.4f})")

    return delta.detach()


def main():
    ap = argparse.ArgumentParser()
    base = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--identity", default=os.path.join(base, "..", "data", "celebahq_sample", "identity_00.png"))
    ap.add_argument("--epsilon", type=float, default=8 / 255)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--step-size", type=float, default=1 / 255)
    ap.add_argument("--eval-grid", type=int, nargs="+", default=[100, 150, 200, 250, 300, 350, 400])
    ap.add_argument("--eval-seeds", type=int, default=5)
    ap.add_argument("--out-dir", default=os.path.join(base, "..", "results"))
    ap.add_argument("--lambda1", type=float, default=0.5, help="PFG (patch-frequency guidance) weight")
    ap.add_argument("--lambda2", type=float, default=0.5, help="ETG (erroneous-timestep guidance) weight")
    ap.add_argument("--tag", default="", help="suffix for output filenames, e.g. '_etgonly' (default: none, "
                                                "matches original combined-loss baseline filenames)")
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

    print(f"=== AntiPure (ANALOG from paper equations, no official code): "
          f"PFG(lambda1={args.lambda1})+ETG(lambda2={args.lambda2}) attack (epsilon={args.epsilon:.4f}) ===")
    delta = optimize_antipure(pipe, clean_pixel, args.epsilon, args.steps, args.step_size,
                               lambda1=args.lambda1, lambda2=args.lambda2)
    adv_pixel = (clean_pixel + delta).clamp(0, 1)
    torch.cuda.empty_cache()

    print(f"\n=== Evaluating across the FULL Theta grid {args.eval_grid} (n_seeds={args.eval_seeds}) ===")
    grid_unprotected = evaluate_grid(pipe, face_app, clean_img, clean_pixel, args.eval_grid,
                                      fixed_timesteps, text_emb, args.eval_seeds)
    grid_ap = evaluate_grid(pipe, face_app, clean_img, adv_pixel, args.eval_grid,
                             fixed_timesteps, text_emb, args.eval_seeds)

    print(f"\n{'t*':>6} {'unprotected':>12} {'antipure-analog':>16}")
    for t in args.eval_grid:
        print(f"{t:>6} {grid_unprotected[t]:>12.4f} {grid_ap[t]:>16.4f}")
    worst_unprotected = max(grid_unprotected.values())
    worst_ap = max(grid_ap.values())
    print(f"\nworst-case over Theta: unprotected={worst_unprotected:.4f}  antipure-analog={worst_ap:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    img = Image.fromarray((adv_pixel[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))
    img.save(os.path.join(args.out_dir, f"antipure_baseline{args.tag}_immunized.png"))
    import json
    with open(os.path.join(args.out_dir, f"antipure_baseline{args.tag}_grid.json"), "w") as f:
        json.dump({"unprotected": grid_unprotected, "antipure": grid_ap,
                   "worst_case": {"unprotected": worst_unprotected, "antipure": worst_ap},
                   "lambda1": args.lambda1, "lambda2": args.lambda2}, f, indent=2)
    print(f"\nSaved to {args.out_dir}")


if __name__ == "__main__":
    main()
