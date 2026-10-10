"""A THIRD purifier architecture for Theta (certification_framework.md Sec 3b).

Motivation (the reviewer question this answers): Sec. VII's certificates cover
exactly two architectures, DiffPure and GrIDPure, and BOTH of them denoise with
Stable Diffusion 1.5's U-Net -- the same network our delta is optimized through.
A reviewer is entitled to ask whether the certificate says anything about a
purifier the perturbation has never seen.

PixArt-alpha (Chen et al., ICLR 2024) is the cleanest available control:

  * it denoises in the SAME latent space -- PixArt-alpha ships the Stable
    Diffusion VAE (4 channels, 8x downsample, scaling_factor 0.18215), so the
    encode/decode halves of the chain are shared with DiffPure;
  * its denoiser is a DiT TRANSFORMER, not a U-Net -- architecturally disjoint
    from anything the delta was optimized against;
  * it is a noise-then-reverse-diffusion process, so P_theta(x) = D_theta(x+eps)
    holds structurally and the existing grid+Lipschitz+Clopper-Pearson machinery
    applies with NO change (we import certify() from certify.py rather than
    re-deriving it, so the math is literally the same code).

That combination isolates one variable: the denoiser. If the certificate
survives here, the protection lives in the shared latent geometry rather than in
U-Net-specific gradients, which is the mechanism-sharing hypothesis's own
prediction. If it does not, that is a falsification we report.
"""

import numpy as np
import torch
from diffusers import DDIMScheduler, PixArtAlphaPipeline
from PIL import Image

DEVICE = "cuda"
PIXART_ID = "PixArt-alpha/PixArt-XL-2-512x512"


def load_pixart_pipeline():
    """Loads PixArt-alpha 512 and swaps in DDIM, matching common.load_sd_pipeline's
    protocol exactly so the two architectures differ only in the denoiser."""
    pipe = PixArtAlphaPipeline.from_pretrained(PIXART_ID, torch_dtype=torch.float16).to(DEVICE)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe


@torch.no_grad()
def get_empty_t5_embedding(pipe):
    """Empty-prompt T5 embedding + its attention mask. Computed once; the caller
    is then free to drop the text encoder (it is 4.3 B params and never used
    again -- purification is unconditional)."""
    emb, mask, _, _ = pipe.encode_prompt(
        "", do_classifier_free_guidance=False, device=DEVICE, clean_caption=False
    )
    return emb.to(torch.float16), mask


def get_fixed_timesteps(pipe, num_steps=50):
    """Held constant across the whole sigma sweep, exactly as common.get_fixed_timesteps
    does for DiffPure -- varying solver granularity with t* would confound the
    Lipschitz measurement."""
    pipe.scheduler.set_timesteps(num_steps)
    return pipe.scheduler.timesteps.clone()


@torch.no_grad()
def encode_to_latent(pipe, image: Image.Image):
    w, h = image.size
    w8, h8 = (w // 8) * 8, (h // 8) * 8
    if (w, h) != (w8, h8):
        image = image.resize((w8, h8))
    arr = np.array(image.convert("RGB")).astype(np.float32) / 127.5 - 1.0
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
def pixart_pass(pipe, image: Image.Image, t_star: int, seed: int, fixed_timesteps, text_emb, text_mask):
    """One PixArt purification pass: encode -> add noise up to t_star -> reverse
    -denoise along fixed_timesteps with the DiT -> decode. Mirrors
    common.diffpure_pass line for line; only the denoiser call differs."""
    scheduler = pipe.scheduler
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    latent = encode_to_latent(pipe, image)
    noise = torch.randn(latent.shape, generator=gen, device=DEVICE, dtype=latent.dtype)
    lat = scheduler.add_noise(latent, noise, torch.tensor([t_star], device=DEVICE))

    reverse_timesteps = fixed_timesteps[fixed_timesteps <= t_star]
    if len(reverse_timesteps) == 0:
        reverse_timesteps = fixed_timesteps[-1:]

    latent_channels = pipe.transformer.config.in_channels
    added = {"resolution": None, "aspect_ratio": None}
    for t in reverse_timesteps:
        out = pipe.transformer(
            lat,
            encoder_hidden_states=text_emb,
            encoder_attention_mask=text_mask,
            timestep=t.reshape(1).to(DEVICE),
            added_cond_kwargs=added,
        )[0]
        # PixArt predicts (noise, learned_sigma); the scheduler only wants the noise half.
        if pipe.transformer.config.out_channels // 2 == latent_channels:
            out = out.chunk(2, dim=1)[0]
        lat = scheduler.step(out, t, lat).prev_sample
    return decode_from_latent(pipe, lat)
