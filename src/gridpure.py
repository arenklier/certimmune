"""GrIDPure: grid/patch-based iterative purification -- a second, structurally
distinct purifier architecture for the A1 pilot (certification_framework.md
Sec 3b, the discrete union-bound axis over architectures a in A).

v2 (2026-07-22): LATENT-SPACE blending. v1 purified each of the 4 patches
independently ALL THE WAY to decoded pixels, then feather-blended the decoded
images -- this produced a visible seam/mesh artifact (documented in
certification_framework.md "Kok-Neden Teshisi": each patch's own VAE decode
introduces slightly inconsistent color/texture at the boundary, and it
compounds every round). v2 instead purifies each patch's LATENT (never
decoded), blends the 4 purified latents into one, and decodes ONCE per round
-- a single coherent decode instead of 4 independent ones stitched after the
fact. Also faster: 1 encode + 1 decode per round instead of 4 encodes + 4
decodes, since patches are sliced directly out of one whole-image latent
rather than each being cropped-then-re-encoded in pixel space.
"""

import numpy as np
import torch
from PIL import Image

from common import DEVICE, encode_to_latent, decode_from_latent, diffpure_latent_pass


def _feather_mask(size, ramp_width, sides):
    """Ramp weight to 0 only on the given sides ('top'/'bottom'/'left'/'right')
    -- i.e. only on edges that actually overlap a neighboring patch. Ramping on
    a canvas OUTER edge (no neighbor there) starves that region of weight and
    shows up as a vignette/border artifact -- a real bug in the first version
    of this fix, caught by inspecting the output image before committing to a
    full rerun."""
    m = np.ones((size, size), dtype=np.float32)
    ramp_width = max(1, ramp_width)
    ramp = np.linspace(0.0, 1.0, ramp_width, dtype=np.float32)
    if "top" in sides:
        m[:ramp_width, :] *= ramp[:, None]
    if "bottom" in sides:
        m[-ramp_width:, :] *= ramp[::-1][:, None]
    if "left" in sides:
        m[:, :ramp_width] *= ramp[None, :]
    if "right" in sides:
        m[:, -ramp_width:] *= ramp[::-1][None, :]
    return m


def gridpure(pipe, image, rounds, t_star_round, seed, fixed_timesteps, text_emb,
             patch_size=320, overlap=128, img_size=512):
    """rounds = R, the sigma-equivalent (and inherently discrete -- see
    certification_framework.md) strength axis for this architecture. Each
    round: encode the whole image once, purify each of 4 overlapping patch
    REGIONS OF THE LATENT independently, feather-blend the purified latents
    (not decoded pixels), decode once, feed forward into the next round.
    """
    vae_scale = 8  # SD1.5 VAE spatial downsample factor
    patch_lat = patch_size // vae_scale
    overlap_lat = overlap // vae_scale
    img_lat = img_size // vae_scale

    positions_lat = [
        (0, 0),
        (img_lat - patch_lat, 0),
        (0, img_lat - patch_lat),
        (img_lat - patch_lat, img_lat - patch_lat),
    ]
    # each patch only ramps down on the sides that actually overlap a neighbor;
    # canvas-outer-facing sides keep full weight (no vignette)
    sides_per_patch = [
        {"right", "bottom"},  # top-left
        {"left", "bottom"},   # top-right
        {"right", "top"},     # bottom-left
        {"left", "top"},      # bottom-right
    ]
    masks = [
        torch.from_numpy(_feather_mask(patch_lat, overlap_lat // 2, sides)).to(DEVICE, dtype=torch.float16)[None, None]
        for sides in sides_per_patch
    ]

    current = image.resize((img_size, img_size))
    for r in range(rounds):
        full_latent = encode_to_latent(pipe, current)  # 1,4,img_lat,img_lat
        acc = torch.zeros_like(full_latent)
        wsum = torch.zeros((1, 1, img_lat, img_lat), device=DEVICE, dtype=full_latent.dtype)

        for i, (lx, ly) in enumerate(positions_lat):
            patch_latent = full_latent[:, :, ly:ly + patch_lat, lx:lx + patch_lat].clone()
            patch_seed = seed * 1000 + r * 10 + i
            purified_patch = diffpure_latent_pass(
                pipe, patch_latent, t_star_round, patch_seed, fixed_timesteps, text_emb
            )
            mask = masks[i]
            acc[:, :, ly:ly + patch_lat, lx:lx + patch_lat] += purified_patch * mask
            wsum[:, :, ly:ly + patch_lat, lx:lx + patch_lat] += mask

        blended_latent = acc / wsum.clamp_min(1e-6)
        current = decode_from_latent(pipe, blended_latent)

    return current
