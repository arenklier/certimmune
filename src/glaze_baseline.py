"""Glaze-style baseline (Shan et al., USENIX Security 2023) for comparison
against our EOT-over-Theta delta (certification_framework.md "delta Optimizasyonu").

Glaze's actual mechanism is a targeted style-mimicry cloak (move the image's
feature representation toward a DIFFERENT chosen art style) under an LPIPS
perceptual budget, using proprietary internals we don't have access to. The
standard simplified analog used for comparison purposes in papers that don't
reproduce Glaze's exact style-transfer machinery: an UNTARGETED feature-space
cloak -- maximize the distance between a perturbed image's embedding (here,
CLIP's image encoder, a natural public stand-in for "a general perceptual
feature extractor") and its own clean embedding, under the SAME L_inf budget
(epsilon=8/255) our method uses.

Critically, this is a STATIC perturbation -- optimized once, with NO
purification-awareness and NO EOT over a range of purification strengths --
exactly matching how Glaze (and PhotoGuard, Anti-DreamBooth) actually work in
the literature. This is the precise comparison point our certification
framework's motivation rests on: does a purification-oblivious baseline hold
up as well as an EOT-optimized delta under an adaptive purification-strength
adversary?
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPImageProcessor

from common import DEVICE, load_sd_pipeline, load_face_app, get_fixed_timesteps
from delta_optimize import pil_to_pixel_tensor, evaluate_grid


def load_clip():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().to(DEVICE)
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor


def clip_embed(model, processor, img_01):
    """img_01: (1,3,H,W) tensor in [0,1], differentiable end-to-end (bypasses
    the processor's own image-loading path, applies its normalization
    manually so gradients flow back to img_01)."""
    mean = torch.tensor(processor.image_mean, device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor(processor.image_std, device=DEVICE).view(1, 3, 1, 1)
    x = F.interpolate(img_01, size=(224, 224), mode="bicubic", align_corners=False)
    x = (x - mean) / std
    return model.get_image_features(pixel_values=x)


def optimize_glaze(clip_model, clip_processor, clean_pixel, epsilon, steps, step_size, seed=0):
    delta = torch.zeros_like(clean_pixel, requires_grad=True)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)

    with torch.no_grad():
        clean_emb = clip_embed(clip_model, clip_processor, clean_pixel).detach()

    for step in range(steps):
        x_adv = (clean_pixel + delta).clamp(0, 1)
        emb = clip_embed(clip_model, clip_processor, x_adv)
        sim = F.cosine_similarity(emb, clean_emb).mean()  # minimize: push feature-space away from clean
        sim.backward()

        with torch.no_grad():
            delta -= step_size * delta.grad.sign()
            delta.clamp_(-epsilon, epsilon)
        delta.grad = None

        if step % 5 == 0 or step == steps - 1:
            print(f"    step {step:3d}: CLIP cosine-sim(perturbed, clean) = {sim.item():.4f}")

    return delta.detach()


def main():
    ap = argparse.ArgumentParser()
    base = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--identity", default=os.path.join(base, "..", "data", "celebahq_sample", "identity_00.png"))
    ap.add_argument("--epsilon", type=float, default=8 / 255)
    ap.add_argument("--steps", type=int, default=30)
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

    clip_model, clip_processor = load_clip()

    clean_img = Image.open(args.identity).convert("RGB")
    clean_pixel = pil_to_pixel_tensor(clean_img)

    print(f"=== Glaze-style baseline: static CLIP feature-space cloak (epsilon={args.epsilon:.4f}) ===")
    delta = optimize_glaze(clip_model, clip_processor, clean_pixel, args.epsilon, args.steps, args.step_size)
    adv_pixel = (clean_pixel + delta).clamp(0, 1)
    torch.cuda.empty_cache()

    print(f"\n=== Evaluating across the FULL Theta grid {args.eval_grid} (n_seeds={args.eval_seeds}) ===")
    grid_unprotected = evaluate_grid(pipe, face_app, clean_img, clean_pixel, args.eval_grid,
                                      fixed_timesteps, text_emb, args.eval_seeds)
    grid_glaze = evaluate_grid(pipe, face_app, clean_img, adv_pixel, args.eval_grid,
                                fixed_timesteps, text_emb, args.eval_seeds)

    print(f"\n{'t*':>6} {'unprotected':>12} {'glaze-style':>12}")
    for t in args.eval_grid:
        print(f"{t:>6} {grid_unprotected[t]:>12.4f} {grid_glaze[t]:>12.4f}")
    worst_unprotected = max(grid_unprotected.values())
    worst_glaze = max(grid_glaze.values())
    print(f"\nworst-case over Theta: unprotected={worst_unprotected:.4f}  glaze-style={worst_glaze:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    img = Image.fromarray((adv_pixel[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))
    img.save(os.path.join(args.out_dir, "glaze_baseline_immunized.png"))
    import json
    with open(os.path.join(args.out_dir, "glaze_baseline_grid.json"), "w") as f:
        json.dump({"unprotected": grid_unprotected, "glaze": grid_glaze,
                   "worst_case": {"unprotected": worst_unprotected, "glaze": worst_glaze}}, f, indent=2)
    print(f"\nSaved to {args.out_dir}")


if __name__ == "__main__":
    main()
