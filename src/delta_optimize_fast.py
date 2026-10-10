"""Fast-EOT variant: same EOT-over-Theta objective as delta_optimize.py's
optimize_delta_eot, but replacing the expensive multi-step reverse-denoising
purification simulation (10-40 UNet calls per theta sample, depending on
t_star) with a SINGLE UNet call's one-step x0-hat estimate -- the same
closed-form predicted-clean-latent estimator (Tweedie's formula / DDPM's
standard x0-prediction) the AntiPure analog's PFG mechanism already uses in
antipure_baseline.py's patch_dct_highfreq_loss.

Motivation (raised by a fresh-eyes review of the paper's own §7.3/§9): our
current EOT-delta costs ~60x PhotoGuard's per-step cost, most of which
Sec 5B attributes to backpropping through the *iterative* reverse-denoising
loop (k=3 branches alone imply ~20x; the remainder is purification-chain
*depth*, i.e. the number of UNet calls per branch). This script tests
whether a cheap, single-UNet-call approximation of "what the purifier would
do" is enough signal to still produce a tight certified perturbation --
trading purification fidelity (a full multi-step simulation) for per-step
cost (one UNet call instead of up to ~40).

HONEST CAVEAT, stated up front: this is a cruder approximation of the
purifier's actual output than the full reverse-denoising loop the paper's
main method uses and the certification protocol evaluates against (the
*certification* step, certify.py/certify_gridpure.py, always runs the REAL
purifier -- this script only changes what's cheap to differentiate through
during *optimization*; certified numbers reported for this variant are
against the true purifiers, not the one-step approximation). Whether the
resulting delta is still tight is an empirical question this script answers,
not an assumption.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from common import DEVICE, load_sd_pipeline, load_face_app, get_fixed_timesteps
from delta_optimize import (pil_to_pixel_tensor, pixel_to_vae_input, facenet_embed,
                             load_facenet, evaluate_grid)


def differentiable_diffpure_onestep(pipe, x_01, t_star, text_emb, noise):
    """One-step approximation of DiffPure's purified output: a SINGLE UNet
    call's x0-hat estimate instead of iterating the full reverse schedule."""
    vae_in = pixel_to_vae_input(x_01)
    latent = pipe.vae.encode(vae_in).latent_dist.mean * pipe.vae.config.scaling_factor
    t_star_tensor = torch.tensor([t_star], device=DEVICE)
    noisy_latent = pipe.scheduler.add_noise(latent, noise, t_star_tensor)

    alphas_cumprod = pipe.scheduler.alphas_cumprod.to(DEVICE)
    abar_t = alphas_cumprod[t_star_tensor].view(-1, 1, 1, 1).to(noisy_latent.dtype)

    noise_pred = pipe.unet(noisy_latent, t_star_tensor, encoder_hidden_states=text_emb).sample
    x0_hat = (noisy_latent - torch.sqrt(1 - abar_t) * noise_pred) / torch.sqrt(abar_t)

    lat = x0_hat / pipe.vae.config.scaling_factor
    out = pipe.vae.decode(lat).sample
    out_01 = (out.float() / 2 + 0.5).clamp(0, 1)
    return out_01


def optimize_delta_eot_fast(pipe, facenet, clean_pixel, theta_min, theta_max, k_theta, text_emb,
                             epsilon, steps, step_size, seed=0):
    """Same structure/semantics as delta_optimize.optimize_delta_eot (memory-safe
    per-branch backward), but using the one-step purification approximation."""
    delta = torch.zeros_like(clean_pixel, requires_grad=True)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)

    with torch.no_grad():
        clean_emb = facenet_embed(facenet, clean_pixel).detach()

    history = []
    for step in range(steps):
        thetas = torch.randint(theta_min, theta_max + 1, (k_theta,), device=DEVICE, generator=gen).tolist()
        sim_values = []
        for t_star in thetas:
            x_adv = (clean_pixel + delta).clamp(0, 1)
            noise = torch.randn(1, 4, 64, 64, generator=gen, device=DEVICE, dtype=torch.float16)
            purified = differentiable_diffpure_onestep(pipe, x_adv, t_star, text_emb, noise)
            purified_emb = facenet_embed(facenet, purified)
            sim = F.cosine_similarity(purified_emb, clean_emb).mean()
            (sim / k_theta).backward()
            sim_values.append(sim.item())

        mean_sim = float(np.mean(sim_values))

        with torch.no_grad():
            delta -= step_size * delta.grad.sign()
            delta.clamp_(-epsilon, epsilon)
        delta.grad = None

        history.append({"mean_sim": mean_sim, "thetas": thetas})
        if step % 5 == 0 or step == steps - 1:
            print(f"    step {step:3d}: mean_sim={mean_sim:.4f}  thetas={thetas}")

    return delta.detach(), history


def main():
    ap = argparse.ArgumentParser()
    base = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--identity", default=os.path.join(base, "..", "data", "celebahq_sample", "identity_00.png"))
    ap.add_argument("--epsilon", type=float, default=8 / 255)
    ap.add_argument("--theta-min", type=int, default=100)
    ap.add_argument("--theta-max", type=int, default=400)
    ap.add_argument("--k-theta", type=int, default=3)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--step-size", type=float, default=1 / 255)
    ap.add_argument("--eval-grid", type=int, nargs="+", default=[100, 150, 200, 250, 300, 350, 400])
    ap.add_argument("--eval-seeds", type=int, default=5)
    ap.add_argument("--out-dir", default=os.path.join(base, "..", "results"))
    ap.add_argument("--seed", type=int, default=0,
                    help="optimization seed. 0 reproduces the runs already on disk; other values "
                         "give an independent draw of the same budget.")
    ap.add_argument("--tag", default="_fasteot")
    ap.add_argument("--time-only", action="store_true", help="just time N steps, skip eval/save")
    args = ap.parse_args()

    pipe = load_sd_pipeline()
    face_app = load_face_app()
    facenet = load_facenet()
    fixed_timesteps = get_fixed_timesteps(pipe, 50)
    text_input = pipe.tokenizer([""], padding="max_length", max_length=pipe.tokenizer.model_max_length,
                                 truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        text_emb = pipe.text_encoder(text_input.input_ids)[0]

    clean_img = Image.open(args.identity).convert("RGB")
    clean_pixel = pil_to_pixel_tensor(clean_img)

    if args.time_only:
        import time
        torch.cuda.synchronize()
        t0 = time.time()
        delta, history = optimize_delta_eot_fast(pipe, facenet, clean_pixel, args.theta_min, args.theta_max,
                                                   args.k_theta, text_emb, args.epsilon, args.steps,
                                                   args.step_size)
        torch.cuda.synchronize()
        t1 = time.time()
        print(f"\n=== TIMING: {args.steps} fast-EOT steps (k_theta={args.k_theta}) took {t1 - t0:.3f}s "
              f"-> {(t1 - t0) / args.steps:.4f}s/step ===")
        return

    print(f"=== Fast-EOT (one-step x0-hat approximation), k_theta={args.k_theta}, "
          f"Theta=[{args.theta_min},{args.theta_max}], epsilon={args.epsilon:.4f} ===")
    delta, history = optimize_delta_eot_fast(pipe, facenet, clean_pixel, args.theta_min, args.theta_max,
                                              args.k_theta, text_emb, args.epsilon, args.steps,
                                              args.step_size, seed=args.seed)
    adv_pixel = (clean_pixel + delta).clamp(0, 1)
    torch.cuda.empty_cache()

    print(f"\n=== Evaluating (REAL multi-step purifier, not the one-step approx) across grid "
          f"{args.eval_grid} (n_seeds={args.eval_seeds}) ===")
    grid_unprotected = evaluate_grid(pipe, face_app, clean_img, clean_pixel, args.eval_grid,
                                      fixed_timesteps, text_emb, args.eval_seeds)
    grid_fast = evaluate_grid(pipe, face_app, clean_img, adv_pixel, args.eval_grid,
                               fixed_timesteps, text_emb, args.eval_seeds)

    print(f"\n{'t*':>6} {'unprotected':>12} {'fast-eot':>12}")
    for t in args.eval_grid:
        print(f"{t:>6} {grid_unprotected[t]:>12.4f} {grid_fast[t]:>12.4f}")
    worst_unprotected = max(grid_unprotected.values())
    worst_fast = max(grid_fast.values())
    print(f"\nworst-case over Theta: unprotected={worst_unprotected:.4f}  fast-eot={worst_fast:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    img = Image.fromarray((adv_pixel[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))
    img.save(os.path.join(args.out_dir, f"delta_immunized{args.tag}.png"))
    import json
    with open(os.path.join(args.out_dir, f"delta{args.tag}_grid.json"), "w") as f:
        json.dump({"unprotected": grid_unprotected, "fast_eot": grid_fast,
                   "worst_case": {"unprotected": worst_unprotected, "fast_eot": worst_fast},
                   "steps": args.steps, "k_theta": args.k_theta}, f, indent=2)
    print(f"\nSaved to {args.out_dir}")


if __name__ == "__main__":
    main()
