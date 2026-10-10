"""Certification against the THIRD purifier architecture (PixArt-alpha DiT).

Everything here is deliberately a copy of the DiffPure path with exactly one
substitution -- pixart_pass for diffpure_pass. The grid, tau, alpha_mc, the
Clopper-Pearson bound, the empirical Lipschitz term and the Bonferroni option
are not re-derived: certify() is IMPORTED from certify.py, so the third column
of the comparison table is produced by literally the same arithmetic as the
first two. A difference in the resulting alpha can therefore only come from the
denoiser.

Two run modes:
  --arms  label=protected.png ...   one clean reference, many protected images
                                    (the cross-method table)
  --cohort 01,02,04,...             per-identity clean/protected pairs
                                    (the cohort replication)

Each arm writes its own JSON and is skipped if that JSON already exists, so an
interrupted run resumes without recomputation.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_face_app, arcface_embed, cosine_sim
from certify import certify
from pixart_purifier import (load_pixart_pipeline, get_empty_t5_embedding, get_fixed_timesteps,
                             pixart_pass)

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "..", "results")
DATA = os.path.join(BASE, "..", "data", "celebahq_sample")


def sweep(pipe, face_app, img, emb_clean, tau, t_grid, n_seeds, fixed_timesteps, text_emb, text_mask):
    """Per grid point: successes k out of n, plus the raw similarity summary.
    The raw mean is not used by the certificate; it is recorded because a k=0
    point tells you nothing about HOW far below tau the arm sits, and the
    unprotected arm's raw value is what says whether the purifier is still
    useful to an attacker at that strength."""
    out = {}
    for t_star in t_grid:
        k, sims = 0, []
        for seed in range(n_seeds):
            purified = pixart_pass(pipe, img, t_star, seed, fixed_timesteps, text_emb, text_mask)
            emb = arcface_embed(face_app, purified)
            sim = cosine_sim(emb_clean, emb) if emb is not None else 0.0
            sims.append(sim)
            if sim >= tau:
                k += 1
        out[t_star] = {"k": k, "n": n_seeds, "phat": k / n_seeds,
                       "sim_mean": float(np.mean(sims)), "sim_max": float(np.max(sims)),
                       "no_face_rate": float(np.mean([s == 0.0 for s in sims]))}
    return out


def run_arm(pipe, face_app, fixed_timesteps, text_emb, text_mask, args,
            tag, clean_path, prot_path):
    out_path = os.path.join(RES, "certificate_pixart_%s.json" % tag)
    if os.path.exists(out_path) and not args.force:
        print("  skip %-22s (already done)" % tag)
        return json.load(open(out_path))

    clean_img = Image.open(clean_path).convert("RGB")
    emb_clean = arcface_embed(face_app, clean_img)
    if emb_clean is None:
        raise RuntimeError("no face in %s" % clean_path)

    t0 = time.time()
    arms = {}
    if args.with_unprotected:
        arms["unprotected"] = sweep(pipe, face_app, clean_img, emb_clean, args.tau, args.t_grid,
                                    args.n_seeds, fixed_timesteps, text_emb, text_mask)
    if prot_path is not None:
        arms["immunized"] = sweep(pipe, face_app, Image.open(prot_path).convert("RGB"), emb_clean,
                                  args.tau, args.t_grid, args.n_seeds, fixed_timesteps, text_emb, text_mask)

    out = {"tag": tag, "architecture": "pixart_alpha_512_dit", "clean": clean_path,
           "protected": prot_path, "tau": args.tau, "t_grid": args.t_grid,
           "n_seeds": args.n_seeds, "alpha_mc": args.alpha_mc,
           "seconds": round(time.time() - t0, 1)}
    for name, res in arms.items():
        out[name] = {"results": res,
                     "marginal": certify(res, args.t_grid, args.alpha_mc, simultaneous=False),
                     "simultaneous": certify(res, args.t_grid, args.alpha_mc, simultaneous=True)}
        print("  %-22s %-12s alpha_marg=%.4f  alpha_sim=%.4f   k/n: %s" % (
            tag, name, out[name]["marginal"]["alpha_final"], out[name]["simultaneous"]["alpha_final"],
            " ".join("%d" % res[t]["k"] for t in args.t_grid)))
        print("  %-22s %-12s sim_mean:  %s" % (
            "", "", " ".join("%.2f" % res[t]["sim_mean"] for t in args.t_grid)))
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print("  -> %s  (%.0f s)" % (os.path.basename(out_path), out["seconds"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=[],
                    help="label=results-relative-filename, one clean reference for all")
    ap.add_argument("--clean", default=os.path.join(DATA, "identity_00.png"))
    ap.add_argument("--cohort", default="", help="comma-separated identity ids, e.g. 01,02,04")
    ap.add_argument("--cohort-protected", default="immunized_identity_%s_eps8.png")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--t-grid", type=int, nargs="+", default=[100, 150, 200, 250, 300, 350, 400])
    ap.add_argument("--n-seeds", type=int, default=30)
    ap.add_argument("--alpha-mc", type=float, default=0.05)
    ap.add_argument("--with-unprotected", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    pipe = load_pixart_pipeline()
    text_emb, text_mask = get_empty_t5_embedding(pipe)
    fixed_timesteps = get_fixed_timesteps(pipe, 50)
    face_app = load_face_app()
    print("grid=%s  n=%d  tau=%.2f" % (args.t_grid, args.n_seeds, args.tau), flush=True)

    for spec in args.arms:
        label, fname = spec.split("=", 1)
        run_arm(pipe, face_app, fixed_timesteps, text_emb, text_mask, args,
                label, args.clean, os.path.join(RES, fname) if fname else None)

    for ident in [i for i in args.cohort.split(",") if i]:
        run_arm(pipe, face_app, fixed_timesteps, text_emb, text_mask, args,
                "identity_%s" % ident,
                os.path.join(DATA, "identity_%s.png" % ident),
                os.path.join(RES, args.cohort_protected % ident))


if __name__ == "__main__":
    main()
