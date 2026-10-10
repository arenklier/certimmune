"""
Pilot experiment: test Assumption A1 (Lipschitz-in-sigma) from
docs/certification_framework.md Section 3.

Tests whether h_a(sigma) = E_eps[ sim(purify(x+eps, sigma), x) ] is smooth
(Lipschitz) in sigma -- the purification strength -- for a FIXED purifier
architecture a = DiffPure (Nie et al., ICML 2022), reusing the already-cached
SD 1.5 UNet+scheduler as the purifier backbone (this is the exact structural
assumption in certification_framework.md Section 1: P_theta(x) = D_theta(x+eps)).

Uses the cheap proxy from Section 5: a frozen ArcFace embedding distance,
NOT a full DreamBooth fine-tune -- this pilot only checks the mathematical
regularity property, not end-to-end attack success. sigma is parameterized
here as the DDIM start timestep t* in [0, 1000) (standard DiffPure convention).

NOTE: uses a synthetic SD-generated placeholder "identity" image so the pilot
needs no dataset download. Swap --identity-image for a real CelebA-HQ/
VGGFace2 photo before the real experiment.
"""

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler

DEVICE = "cuda"
MODEL_ID = "runwayml/stable-diffusion-v1-5"
INSIGHTFACE_ROOT = "/mnt/data/certimmune/cache/insightface"  # NOT $HOME -- root disk is full


def load_sd_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, safety_checker=None
    ).to(DEVICE)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_face_app():
    from insightface.app import FaceAnalysis

    os.makedirs(INSIGHTFACE_ROOT, exist_ok=True)
    app = FaceAnalysis(
        name="buffalo_l",
        root=INSIGHTFACE_ROOT,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def make_placeholder_identity(pipe, seed, path):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    img = pipe(
        "a passport photo of a person, full face visible, forehead to shoulders, "
        "looking straight at camera, plain white background, even studio lighting, "
        "sharp focus, high quality photograph",
        negative_prompt="extreme closeup, macro, cropped, partial face, blurry, artistic, painting",
        num_inference_steps=30,
        guidance_scale=7.5,
        generator=gen,
    ).images[0]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)
    return img


@torch.no_grad()
def encode_to_latent(pipe, image: Image.Image):
    image = image.resize((512, 512))
    arr = np.array(image).astype(np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE, dtype=torch.float16)
    latent = pipe.vae.encode(tensor).latent_dist.sample() * pipe.vae.config.scaling_factor
    return latent


@torch.no_grad()
def decode_from_latent(pipe, latent):
    latent = latent / pipe.vae.config.scaling_factor
    image = pipe.vae.decode(latent).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    arr = (image[0].permute(1, 2, 0).float().cpu().numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr)


@torch.no_grad()
def diffpure(pipe, image, t_star, seed, fixed_timesteps, text_emb):
    """DiffPure: add Gaussian noise up to timestep t_star, then reverse-denoise
    back to 0 using the SAME fixed timestep grid for every t_star (this keeps
    solver discretization granularity constant across the sigma sweep -- if we
    let step-count scale with t_star instead, discretization artifacts would
    be confounded with the smoothness property we're actually trying to measure).
    """
    scheduler = pipe.scheduler
    latent = encode_to_latent(pipe, image)

    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    noise = torch.randn(latent.shape, generator=gen, device=DEVICE, dtype=latent.dtype)
    t_star_tensor = torch.tensor([t_star], device=DEVICE)
    lat = scheduler.add_noise(latent, noise, t_star_tensor)

    reverse_timesteps = fixed_timesteps[fixed_timesteps <= t_star]
    if len(reverse_timesteps) == 0:
        reverse_timesteps = fixed_timesteps[-1:]

    for t in reverse_timesteps:
        noise_pred = pipe.unet(lat, t, encoder_hidden_states=text_emb).sample
        lat = scheduler.step(noise_pred, t, lat).prev_sample

    return decode_from_latent(pipe, lat)


def arcface_embed(face_app, image: Image.Image):
    arr = np.array(image.convert("RGB"))[:, :, ::-1]  # RGB -> BGR for insightface
    faces = face_app.get(arr)
    if not faces:
        return None
    # largest detected face by bbox area (robust if purification introduces artifacts)
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    return faces[0].normed_embedding


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=12, help="Monte-Carlo noise samples per t*")
    ap.add_argument(
        "--t-grid", type=int, nargs="+", default=[50, 100, 150, 200, 250, 300, 350, 400, 500, 600]
    )
    ap.add_argument("--fixed-steps", type=int, default=50, help="reverse-solver grid size, held constant across the sweep")
    ap.add_argument("--identity-image", type=str, default=None, help="path to a real identity photo; omit to generate a placeholder")
    ap.add_argument("--identity-seed", type=int, default=0)
    ap.add_argument("--out", default="results/a1_lipschitz_pilot.json")
    args = ap.parse_args()

    pipe = load_sd_pipeline()
    face_app = load_face_app()

    if args.identity_image:
        identity_img = Image.open(args.identity_image)
        emb_clean = arcface_embed(face_app, identity_img)
        if emb_clean is None:
            raise RuntimeError("No face detected in identity image -- regenerate/replace it.")
    else:
        identity_img, emb_clean = None, None
        for seed in range(args.identity_seed, args.identity_seed + 10):
            candidate = make_placeholder_identity(pipe, seed, "data/pilot_identity.png")
            emb = arcface_embed(face_app, candidate)
            if emb is not None:
                identity_img, emb_clean = candidate, emb
                print(f"NOTE: using SD-generated placeholder identity (seed={seed}, "
                      f"data/pilot_identity.png) -- swap --identity-image for a real photo "
                      f"before the full experiment.")
                break
        if identity_img is None:
            raise RuntimeError("No face detected across 10 placeholder seeds -- check prompt/detector.")

    scheduler = pipe.scheduler
    scheduler.set_timesteps(args.fixed_steps)
    fixed_timesteps = scheduler.timesteps.clone()

    text_input = pipe.tokenizer(
        [""], padding="max_length", max_length=pipe.tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    ).to(DEVICE)
    with torch.no_grad():
        text_emb = pipe.text_encoder(text_input.input_ids)[0]

    results = {}
    for t_star in args.t_grid:
        sims = []
        for seed in range(args.n_seeds):
            purified = diffpure(pipe, identity_img, t_star, seed, fixed_timesteps, text_emb)
            emb = arcface_embed(face_app, purified)
            if emb is not None:
                sims.append(cosine_sim(emb_clean, emb))
        results[t_star] = {
            "mean_sim": float(np.mean(sims)) if sims else None,
            "std_sim": float(np.std(sims)) if sims else None,
            "n_valid": len(sims),
        }
        r = results[t_star]
        print(f"t*={t_star:4d}  mean_sim={r['mean_sim']:.4f}  std={r['std_sim']:.4f}  "
              f"n={r['n_valid']}/{args.n_seeds}")

    ts = sorted(results.keys())
    slopes = []
    for i in range(1, len(ts)):
        h1, h0 = results[ts[i]]["mean_sim"], results[ts[i - 1]]["mean_sim"]
        if h1 is not None and h0 is not None:
            slopes.append(abs(h1 - h0) / (ts[i] - ts[i - 1]))

    summary = {
        "purifier": "DiffPure (SD1.5 UNet backbone)",
        "t_grid": ts,
        "n_seeds_per_point": args.n_seeds,
        "fixed_solver_steps": args.fixed_steps,
        "results": results,
        "empirical_local_lipschitz_max_slope": max(slopes) if slopes else None,
        "empirical_local_lipschitz_mean_slope": float(np.mean(slopes)) if slopes else None,
    }

    if slopes:
        print(f"\nEmpirical local Lipschitz constant (max adjacent-slope): {max(slopes):.6f}")
        print(f"Empirical local Lipschitz constant (mean adjacent-slope): {np.mean(slopes):.6f}")
        print("-> Compare against README/framework's assumption; a wildly non-monotone or")
        print("   spiky h(t*) curve (slope >> mean at isolated points) would challenge A1.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
