"""Does h(t*) stay monotone BETWEEN the certification grid points?

The whole job of the grid+Lipschitz term is to control the interior of Theta,
so checking monotonicity at the same seven points the certificate already uses
proves nothing about the interior -- a reviewer would rightly say we swapped one
unverifiable assumption for another. This sweeps t* on a 10-step grid (31 points
instead of 7) at a smaller n, which trades confidence per point for resolution
in t*: the question here is the SHAPE of h, not a tight bound on it.

Only arms whose h actually varies are worth sweeping. A protected arm that is
k=0 at all seven points is monotone trivially and carries no information, so the
arms swept are the unprotected control and the two partial-protection baselines
that fall off inside [100, 200].
"""
import argparse
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

ap = argparse.ArgumentParser()
ap.add_argument("--n-seeds", type=int, default=10)
ap.add_argument("--step", type=int, default=10)
ap.add_argument("--tau", type=float, default=0.5)
args = ap.parse_args()

GRID = list(range(100, 401, args.step))
ARMS = [("unprotected", os.path.join(DATA, "identity_00.png")),
        ("antipure", os.path.join(RES, "antipure_baseline_immunized.png")),
        ("ours_wcmatch17", os.path.join(RES, "delta_computematched17_immunized.png"))]

pipe = load_sd_pipeline()
face_app = load_face_app()
fixed_timesteps = get_fixed_timesteps(pipe, 50)
text_emb = get_empty_text_embedding(pipe)
clean = Image.open(os.path.join(DATA, "identity_00.png")).convert("RGB")
emb_clean = arcface_embed(face_app, clean)

out = {"t_grid": GRID, "n_seeds": args.n_seeds, "tau": args.tau, "arms": {}}
for label, path in ARMS:
    img = Image.open(path).convert("RGB")
    res = {}
    for t in GRID:
        k = 0
        for s in range(args.n_seeds):
            e = arcface_embed(face_app, diffpure_pass(pipe, img, t, s, fixed_timesteps, text_emb))
            if (cosine_sim(emb_clean, e) if e is not None else 0.0) >= args.tau:
                k += 1
        res[t] = {"k": k, "n": args.n_seeds}
    out["arms"][label] = res
    print("%-16s %s" % (label, " ".join("%d" % res[t]["k"] for t in GRID)), flush=True)
    with open(os.path.join(RES, "dense_sweep.json"), "w") as f:
        json.dump(out, f, indent=2)
print("DENSE DONE")
