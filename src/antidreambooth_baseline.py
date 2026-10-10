"""Anti-DreamBooth baseline (Le et al., ICCV 2023) for comparison against our
EOT-over-Theta delta.

v2 (2026-07-23): ported the CORE per-step attack loss from the OFFICIAL repo
(github.com/VinAIResearch/Anti-DreamBooth, attacks/aspl.py: pgd_attack()),
replacing our v1 facenet-embedding analog. The real ASPL loss is the
*diffusion training loss itself* -- predict the injected noise at a random
timestep, conditioned on the instance prompt, and maximize (not minimize)
that prediction error via gradient-ascent PGD -- run against a periodically
re-fine-tuned surrogate DreamBooth model in the full published algorithm
(the "alternating" part of Alternating Surrogate and Perturbation Learning).

SCOPE (honest, load-bearing): we port the per-step PGD loss exactly, but
against a FROZEN pretrained UNet rather than a surrogate re-fine-tuned every
few outer steps -- the latter requires unrolling backprop through actual
LoRA/DreamBooth training steps inside the attack loop (the same OOM-prone
pattern we hit and fixed for our own EOT delta, but for K outer iterations of
real fine-tuning, not one purification pass -- materially more expensive).
This static-surrogate variant is closer to Anti-DreamBooth's OWN "FSMG"
(Fully-trained Surrogate Model Guidance) ablation than to full ASPL, but is a
genuinely faithful port of the actual diffusion-objective attack, unlike our
v1 analog which attacked an unrelated face-recognition embedding space.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from common import DEVICE, load_sd_pipeline, load_face_app, get_fixed_timesteps
from delta_optimize import pil_to_pixel_tensor, pixel_to_vae_input, evaluate_grid


def optimize_antidreambooth(pipe, clean_pixel, epsilon, steps, step_size,
                             instance_prompt="a photo of sks person", seed=0):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    delta = torch.zeros_like(clean_pixel, requires_grad=True)

    text_input = pipe.tokenizer([instance_prompt], padding="max_length",
                                 max_length=pipe.tokenizer.model_max_length,
                                 truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        text_emb = pipe.text_encoder(text_input.input_ids)[0]

    for step in range(steps):
        x_adv = (clean_pixel + delta).clamp(0, 1)
        latents = pipe.vae.encode(pixel_to_vae_input(x_adv)).latent_dist.sample() * pipe.vae.config.scaling_factor

        noise = torch.randn(latents.shape, generator=gen, device=DEVICE, dtype=latents.dtype)
        t = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (1,), device=DEVICE, generator=gen).long()
        noisy_latents = pipe.scheduler.add_noise(latents, noise, t)

        model_pred = pipe.unet(noisy_latents, t, encoder_hidden_states=text_emb).sample
        loss = F.mse_loss(model_pred.float(), noise.float())  # official ASPL loss: diffusion noise-pred MSE
        loss.backward()

        with torch.no_grad():
            delta += step_size * delta.grad.sign()  # official direction: ASCENT (maximize training loss)
            delta.clamp_(-epsilon, epsilon)
        delta.grad = None

        if step % 5 == 0 or step == steps - 1:
            print(f"    step {step:3d}: diffusion noise-pred loss (maximizing) = {loss.item():.4f}")

    return delta.detach()


def main():
    ap = argparse.ArgumentParser()
    base = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--identity", default=os.path.join(base, "..", "data", "celebahq_sample", "identity_00.png"))
    ap.add_argument("--epsilon", type=float, default=8 / 255)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--step-size", type=float, default=1 / 255)
    ap.add_argument("--eval-grid", type=int, nargs="+", default=[100, 150, 200, 250, 300, 350, 400])
    ap.add_argument("--eval-seeds", type=int, default=5)
    ap.add_argument("--out-dir", default=os.path.join(base, "..", "results"))
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

    print(f"=== Anti-DreamBooth (OFFICIAL ASPL core loss, static surrogate): "
          f"diffusion noise-pred attack (epsilon={args.epsilon:.4f}) ===")
    delta = optimize_antidreambooth(pipe, clean_pixel, args.epsilon, args.steps, args.step_size)
    adv_pixel = (clean_pixel + delta).clamp(0, 1)
    torch.cuda.empty_cache()

    print(f"\n=== Evaluating across the FULL Theta grid {args.eval_grid} (n_seeds={args.eval_seeds}) ===")
    grid_unprotected = evaluate_grid(pipe, face_app, clean_img, clean_pixel, args.eval_grid,
                                      fixed_timesteps, text_emb, args.eval_seeds)
    grid_adb = evaluate_grid(pipe, face_app, clean_img, adv_pixel, args.eval_grid,
                              fixed_timesteps, text_emb, args.eval_seeds)

    print(f"\n{'t*':>6} {'unprotected':>12} {'anti-dreambooth':>16}")
    for t in args.eval_grid:
        print(f"{t:>6} {grid_unprotected[t]:>12.4f} {grid_adb[t]:>16.4f}")
    worst_unprotected = max(grid_unprotected.values())
    worst_adb = max(grid_adb.values())
    print(f"\nworst-case over Theta: unprotected={worst_unprotected:.4f}  anti-dreambooth-style={worst_adb:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    img = Image.fromarray((adv_pixel[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))
    img.save(os.path.join(args.out_dir, "antidreambooth_baseline_immunized.png"))
    import json
    with open(os.path.join(args.out_dir, "antidreambooth_baseline_grid.json"), "w") as f:
        json.dump({"unprotected": grid_unprotected, "antidreambooth": grid_adb,
                   "worst_case": {"unprotected": worst_unprotected, "antidreambooth": worst_adb}}, f, indent=2)
    print(f"\nSaved to {args.out_dir}")


if __name__ == "__main__":
    main()
