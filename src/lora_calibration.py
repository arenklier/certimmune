"""Calibration study (certification_framework.md Sec 5): does the cheap
frozen-encoder-distance proxy (already measured in the A1 pilots) track REAL
attack success -- fine-tune a LoRA on a (possibly purified) identity image,
generate images, measure actual ArcFace identity-similarity?

v2 (2026-07-22): adds DreamBooth PRIOR-PRESERVATION LOSS. v1 (instance-only,
no prior preservation) catastrophically failed even on the CLEAN condition:
300 AND 1000 training steps both produced near-zero identity similarity
(-0.02 to +0.02, i.e. chance level), and visual inspection of generated
images showed severe language/style drift -- the model produced a stylized
monochrome painting instead of a photorealistic face, unrelated to the
training identity. This is the classic DreamBooth failure mode that prior
preservation exists to prevent: without a countervailing loss term keeping
the model's general "a photo of a person" prior intact, a rank-4 LoRA
trained hundreds of steps on ONE image drifts the whole text-to-image
mapping, not just the "sks" token region. v2 alternates each step between
the instance loss (identity image + instance prompt) and a class loss
(generic person images, generated once from the un-tuned base model, +
class prompt), matching the original DreamBooth recipe.

Simplifications still flagged (fix before the full experiment matrix,
README #11):
- single image per identity (no multi-image photo SET DreamBooth training --
  the framework's Sec 5 "vekil-granuleriti" note calls for set-level training)
- small class-image pool (8, vs. DreamBooth's usual ~100-200) -- a pilot-scale
  prior-preservation set, not a publication-grade one
- LoRA on UNet attention projections only (to_q/k/v/out), rank 4

This validates the PIPELINE and the proxy-vs-real correlation direction, not
final publication-grade attack-success numbers.
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

from common import (
    DEVICE, load_sd_pipeline, load_face_app, get_empty_text_embedding, get_fixed_timesteps,
    encode_to_latent, diffpure_pass, arcface_embed, cosine_sim,
)
from PIL import Image


def add_lora(pipe, rank=4):
    config = LoraConfig(
        r=rank, lora_alpha=rank,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
    )
    pipe.unet = get_peft_model(pipe.unet, config)
    # keep LoRA params in fp32 for training stability; frozen base stays fp16
    for name, p in pipe.unet.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return pipe


@torch.no_grad()
def generate_class_images(pipe, class_prompt, n_images, seed_offset=5000):
    """Generated ONCE from the un-tuned base model, before any LoRA is
    attached -- these anchor the general 'a photo of a person' prior during
    training so the instance concept doesn't drag the whole model with it."""
    imgs = []
    for i in range(n_images):
        gen = torch.Generator(device=DEVICE).manual_seed(seed_offset + i)
        img = pipe(class_prompt, num_inference_steps=30, guidance_scale=7.5, generator=gen).images[0]
        imgs.append(img)
    return imgs


def _encode_prompt(pipe, prompt):
    text_input = pipe.tokenizer([prompt], padding="max_length",
                                 max_length=pipe.tokenizer.model_max_length,
                                 truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        return pipe.text_encoder(text_input.input_ids)[0]


def train_lora_instance(pipe, instance_images, instance_prompt, steps, lr, seed,
                         class_images=None, class_prompt=None, prior_loss_weight=1.0):
    """instance_images: list of PIL images (the identity's photo SET -- real
    DreamBooth regime, fixing the single-image limitation documented in
    certification_framework.md "Kalibrasyon Pipeline Denemeleri"). Each step
    randomly samples one instance image, same pattern already used for class
    images below."""
    from diffusers import DDPMScheduler
    ddpm = DDPMScheduler.from_config(pipe.scheduler.config)

    instance_text_emb = _encode_prompt(pipe, instance_prompt)
    with torch.no_grad():
        instance_latents = [encode_to_latent(pipe, img) for img in instance_images]

    class_latents, class_text_emb = [], None
    if class_images:
        class_text_emb = _encode_prompt(pipe, class_prompt)
        with torch.no_grad():
            class_latents = [encode_to_latent(pipe, img) for img in class_images]

    params = [p for p in pipe.unet.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)

    pipe.unet.train()
    losses = []
    for step in range(steps):
        i_idx = torch.randint(0, len(instance_latents), (1,), device=DEVICE, generator=gen).item()
        instance_latent = instance_latents[i_idx]
        noise = torch.randn(instance_latent.shape, generator=gen, device=DEVICE, dtype=torch.float16)
        t = torch.randint(0, ddpm.config.num_train_timesteps, (1,), device=DEVICE, generator=gen).long()
        noisy = ddpm.add_noise(instance_latent, noise, t)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            pred = pipe.unet(noisy, t, encoder_hidden_states=instance_text_emb).sample
            inst_loss = F.mse_loss(pred.float(), noise.float())

        total_loss = inst_loss
        if class_latents:
            c_idx = torch.randint(0, len(class_latents), (1,), device=DEVICE, generator=gen).item()
            c_latent = class_latents[c_idx]
            c_noise = torch.randn(c_latent.shape, generator=gen, device=DEVICE, dtype=torch.float16)
            c_t = torch.randint(0, ddpm.config.num_train_timesteps, (1,), device=DEVICE, generator=gen).long()
            c_noisy = ddpm.add_noise(c_latent, c_noise, c_t)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                c_pred = pipe.unet(c_noisy, c_t, encoder_hidden_states=class_text_emb).sample
                class_loss = F.mse_loss(c_pred.float(), c_noise.float())
            total_loss = inst_loss + prior_loss_weight * class_loss

        opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(total_loss.item())
        if step % 100 == 0 or step == steps - 1:
            extra = f" (inst={inst_loss.item():.4f})" if class_latents else ""
            print(f"    step {step:4d}: loss={total_loss.item():.4f}{extra}")
    pipe.unet.eval()
    return losses


@torch.no_grad()
def generate_and_score(pipe, prompt, face_app, emb_clean, n_images, seed_offset):
    sims = []
    for i in range(n_images):
        gen = torch.Generator(device=DEVICE).manual_seed(seed_offset + i)
        img = pipe(prompt, num_inference_steps=30, guidance_scale=7.5, generator=gen).images[0]
        emb = arcface_embed(face_app, img)
        sims.append(cosine_sim(emb_clean, emb) if emb is not None else 0.0)
    return sims


def load_identity_set(identity_dir):
    paths = sorted(glob.glob(os.path.join(identity_dir, "*.png")))
    return [Image.open(p).convert("RGB") for p in paths]


def set_level_reference_embedding(face_app, images):
    """Prototype embedding for a multi-image identity set: mean of the
    per-image (already L2-normalized) ArcFace embeddings, renormalized -- a
    standard multi-shot face-verification 'template' technique. Fixes the
    single-image-only proxy the earlier A1 pilots used (certification_
    framework.md Sec 5's 'vekil-granuleriti' note)."""
    embs = [arcface_embed(face_app, img) for img in images]
    embs = [e for e in embs if e is not None]
    if not embs:
        return None
    mean = np.mean(embs, axis=0)
    return mean / (np.linalg.norm(mean) + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--identity-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "metacloak_identities", "identity_3"))
    ap.add_argument("--conditions", type=str, nargs="+", default=["clean", "diffpure_t200", "diffpure_t400"])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--rank", type=int, default=4,
                    help="LoRA rank. Was hard-coded at 4; exposed so the documented setup can be reproduced.")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n-generate", type=int, default=4)
    ap.add_argument("--prompt", default="a photo of sks person")
    ap.add_argument("--class-prompt", default="a photo of a person")
    ap.add_argument("--n-class-images", type=int, default=8)
    ap.add_argument("--prior-loss-weight", type=float, default=1.0)
    ap.add_argument("--reference-dir", default=None,
                    help="Directory whose images build the clean reference embedding that generations are "
                         "scored against. Defaults to --identity-dir, which is correct only for the "
                         "unprotected condition; for a protected run this MUST point at the clean identity "
                         "set, otherwise the 'clean' reference is built from the protected images.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Replicate index. 0 reproduces the runs already on disk exactly; any other "
                         "value shifts both the LoRA training seed and the generation seeds, giving an "
                         "independent draw of the same cell.")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "results", "lora_calibration.json"))
    args = ap.parse_args()

    pipe = load_sd_pipeline()
    face_app = load_face_app()
    fixed_timesteps = get_fixed_timesteps(pipe, 50)
    text_emb_empty = get_empty_text_embedding(pipe)

    clean_images = load_identity_set(args.identity_dir)
    if not clean_images:
        raise RuntimeError(f"No images found in {args.identity_dir}")
    ref_dir = args.reference_dir or args.identity_dir
    ref_images = clean_images if ref_dir == args.identity_dir else load_identity_set(ref_dir)
    if not ref_images:
        raise RuntimeError(f"No images found in {ref_dir}")
    emb_clean = set_level_reference_embedding(face_app, ref_images)
    if emb_clean is None:
        raise RuntimeError(f"No face detected in any image under {ref_dir}")
    print(f"Loaded identity set: {len(clean_images)} images from {args.identity_dir}")
    print(f"Reference embedding: {len(ref_images)} images from {ref_dir}"
          + ("  [SAME AS TRAINING SET]" if ref_dir == args.identity_dir else "  [FIXED CLEAN REFERENCE]"))

    base_unet = pipe.unet  # pristine reference -- reattach fresh LoRA per condition, don't rely on peft unload()

    print(f"Generating {args.n_class_images} class images from the base model for prior preservation...")
    class_images = generate_class_images(pipe, args.class_prompt, args.n_class_images)

    results = {}
    for cond in args.conditions:
        print(f"\n=== condition: {cond} ===")
        if cond == "clean":
            train_imgs = clean_images
        elif cond.startswith("diffpure_t"):
            t_star = int(cond.replace("diffpure_t", ""))
            train_imgs = [
                diffpure_pass(pipe, img, t_star, seed=i, fixed_timesteps=fixed_timesteps, text_emb=text_emb_empty)
                for i, img in enumerate(clean_images)
            ]
        else:
            raise ValueError(f"unknown condition {cond}")

        # SET-level proxy: prototype embedding of the purified set vs. clean prototype
        proxy_emb = set_level_reference_embedding(face_app, train_imgs)
        proxy_sim = cosine_sim(emb_clean, proxy_emb) if proxy_emb is not None else 0.0
        print(f"  proxy (purified SET ArcFace sim to clean): {proxy_sim:.4f}")

        pipe.unet = base_unet
        pipe = add_lora(pipe, rank=args.rank)
        losses = train_lora_instance(pipe, train_imgs, args.prompt, args.steps, args.lr, seed=args.seed,
                                      class_images=class_images, class_prompt=args.class_prompt,
                                      prior_loss_weight=args.prior_loss_weight)

        real_sims = generate_and_score(pipe, args.prompt, face_app, emb_clean,
                                        args.n_generate, seed_offset=1000 + 1000 * args.seed)
        real_mean = sum(real_sims) / len(real_sims)
        print(f"  REAL attack success (mean ArcFace sim of {args.n_generate} generations): {real_mean:.4f}")
        print(f"  individual: {[round(s, 4) for s in real_sims]}")

        results[cond] = {
            "proxy_sim": proxy_sim,
            "real_sims": real_sims,
            "real_mean": real_mean,
            "final_train_loss": losses[-1],
            "seed": args.seed,
        }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Summary: proxy vs real ===")
    for cond, r in results.items():
        print(f"  {cond:20s} proxy={r['proxy_sim']:.4f}  real_mean={r['real_mean']:.4f}")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
