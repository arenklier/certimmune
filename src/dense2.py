"""Dense sweep, second pass: put the resolution where the assumption can actually break.

The first pass swept three arms over the whole of Theta at n=10, but only one of
them (AntiPure) had a transition inside the range, and it sat at the very bottom,
t* in [100,160]. The arm that matters most was missing: the Glaze analog falls
off in the MIDDLE of Theta (30,30,28,21,5,2,0 on the seven-point grid, i.e.
t* in [200,300]), which is exactly where a grid-interpolation error would do the
most damage. This pass sweeps each arm across its own transition region, at
n=30 rather than n=10 so a one-count wiggle is no longer indistinguishable from
a real bump.
"""
import json
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (load_sd_pipeline, load_face_app, get_empty_text_embedding, get_fixed_timesteps,
                    diffpure_pass, arcface_embed, cosine_sim)

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "..", "results")
DATA = os.path.join(BASE, "..", "data", "celebahq_sample")
N, TAU = 30, 0.5

ARMS = [("glaze", "glaze_baseline_immunized.png", range(200, 311, 10)),
        ("antipure", "antipure_baseline_immunized.png", range(100, 171, 10))]

pipe = load_sd_pipeline()
face_app = load_face_app()
fixed_timesteps = get_fixed_timesteps(pipe, 50)
text_emb = get_empty_text_embedding(pipe)
emb_clean = arcface_embed(face_app, Image.open(os.path.join(DATA, "identity_00.png")).convert("RGB"))

out = {"n_seeds": N, "tau": TAU, "arms": {}}
for label, fname, grid in ARMS:
    grid = list(grid)
    img = Image.open(os.path.join(RES, fname)).convert("RGB")
    res = {}
    for t in grid:
        k = sum(1 for s in range(N)
                if (lambda e: (cosine_sim(emb_clean, e) if e is not None else 0.0))(
                    arcface_embed(face_app, diffpure_pass(pipe, img, t, s, fixed_timesteps, text_emb))) >= TAU)
        res[t] = {"k": k, "n": N}
    out["arms"][label] = {"grid": grid, "results": res}
    print("%-10s %s" % (label, " ".join("%d" % res[t]["k"] for t in grid)), flush=True)
    with open(os.path.join(RES, "dense_sweep2.json"), "w") as f:
        json.dump(out, f, indent=2)
print("DENSE2 DONE")
