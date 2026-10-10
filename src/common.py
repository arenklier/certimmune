"""Shared utilities for CertImmune pilot experiments: SD1.5 pipeline loading,
DiffPure single-pass purification, and ArcFace identity-similarity embedding.
See docs/certification_framework.md Sec 1 for the P_theta(x) = D_theta(x+eps)
structural assumption this code implements.
"""

import os

import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler

DEVICE = "cuda"
MODEL_ID = "runwayml/stable-diffusion-v1-5"
INSIGHTFACE_ROOT = "/mnt/data/certimmune/cache/insightface"  # NOT $HOME -- root disk is tight


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


def get_empty_text_embedding(pipe):
    text_input = pipe.tokenizer(
        [""], padding="max_length", max_length=pipe.tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    ).to(DEVICE)
    with torch.no_grad():
        return pipe.text_encoder(text_input.input_ids)[0]


def get_fixed_timesteps(pipe, num_steps=50):
    """Held constant across every sigma/round sweep in a run -- see
    certification_framework.md Sec 3a on why varying solver granularity with
    the strength parameter would confound the smoothness measurement."""
    pipe.scheduler.set_timesteps(num_steps)
    return pipe.scheduler.timesteps.clone()


@torch.no_grad()
def encode_to_latent(pipe, image: Image.Image):
    w, h = image.size
    w8, h8 = (w // 8) * 8, (h // 8) * 8
    if (w, h) != (w8, h8):
        image = image.resize((w8, h8))
    arr = np.array(image).astype(np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE, dtype=torch.float16)
    return pipe.vae.encode(tensor).latent_dist.sample() * pipe.vae.config.scaling_factor


@torch.no_grad()
def decode_from_latent(pipe, latent):
    latent = latent / pipe.vae.config.scaling_factor
    image = pipe.vae.decode(latent).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    arr = (image[0].permute(1, 2, 0).float().cpu().numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr)


@torch.no_grad()
def diffpure_latent_pass(pipe, latent, t_star: int, seed: int, fixed_timesteps, text_emb):
    """Core DiffPure step operating directly on a latent tensor (no encode/decode):
    add Gaussian noise up to timestep t_star, reverse-denoise back to 0 along
    `fixed_timesteps`. This is the primitive gridpure.py's latent-blend variant
    needs -- it purifies each patch's latent WITHOUT decoding it, so the 4
    patches can be blended into one latent and decoded exactly once per round
    (see gridpure.py docstring for why decoding-then-blending caused seams)."""
    scheduler = pipe.scheduler
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
    return lat


@torch.no_grad()
def diffpure_pass(pipe, image: Image.Image, t_star: int, seed: int, fixed_timesteps, text_emb):
    """Single DiffPure pass (Nie et al. 2022): encode -> diffpure_latent_pass -> decode.
    Works on any divisible-by-8 image size. Architecture 'diffpure' (whole-image)
    in certification_framework.md Sec 3b -- unchanged by the gridpure latent-blend fix."""
    latent = encode_to_latent(pipe, image)
    lat = diffpure_latent_pass(pipe, latent, t_star, seed, fixed_timesteps, text_emb)
    return decode_from_latent(pipe, lat)


def arcface_embed(face_app, image: Image.Image):
    arr = np.array(image.convert("RGB"))[:, :, ::-1]  # RGB -> BGR
    faces = face_app.get(arr)
    if not faces:
        return None
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    return faces[0].normed_embedding


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
