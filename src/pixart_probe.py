"""Feasibility probe before spending GPU hours on a full third-architecture sweep.

Two questions, in this order, because the second is meaningless without the first:

  Q1. Is PixArt purification a THREAT at all? A purifier only matters if it strips
      the perturbation while LEAVING THE IDENTITY INTACT. If purifying the clean
      image already drops ArcFace similarity below tau, the purifier destroys the
      person rather than recovering them, both arms read zero, and the certificate
      would be vacuously "good" for the wrong reason.
  Q2. Given Q1 passes, does our EOT delta still suppress the similarity on a
      denoiser it was never optimized through?
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_face_app, arcface_embed, cosine_sim
from pixart_purifier import load_pixart_pipeline, get_empty_t5_embedding, get_fixed_timesteps, pixart_pass

R = "/mnt/data/certimmune/results"
D = "/mnt/data/certimmune/data/celebahq_sample"
GRID = [100, 200, 300, 400]
SEEDS = [0, 1, 2]

pipe = load_pixart_pipeline()
text_emb, text_mask = get_empty_t5_embedding(pipe)
fixed_timesteps = get_fixed_timesteps(pipe, 50)
face_app = load_face_app()

clean = Image.open(os.path.join(D, "identity_00.png")).convert("RGB")
emb_clean = arcface_embed(face_app, clean)
assert emb_clean is not None

ARMS = [("unprotected", clean),
        ("eot-delta", Image.open(os.path.join(R, "delta_immunized_eot.png")).convert("RGB"))]

print("no-purification sanity:")
for name, img in ARMS:
    e = arcface_embed(face_app, img)
    print("  %-12s sim-to-clean %.4f" % (name, cosine_sim(emb_clean, e) if e is not None else float('nan')))

print()
print("%-12s %6s %8s %8s %8s" % ("arm", "t*", "mean", "min", "max"))
for name, img in ARMS:
    for t in GRID:
        sims = []
        for s in SEEDS:
            out = pixart_pass(pipe, img, t, s, fixed_timesteps, text_emb, text_mask)
            e = arcface_embed(face_app, out)
            sims.append(cosine_sim(emb_clean, e) if e is not None else 0.0)
            if name == "unprotected" and t == 200 and s == 0:
                out.save(os.path.join(R, "pixart_probe_unprotected_t200.png"))
        sims = np.array(sims)
        print("%-12s %6d %8.4f %8.4f %8.4f" % (name, t, sims.mean(), sims.min(), sims.max()))
