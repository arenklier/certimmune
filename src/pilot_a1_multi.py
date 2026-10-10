"""Expanded A1 pilot: real CelebA-HQ identities x two purifier architectures
(DiffPure, GrIDPure) -- see docs/certification_framework.md Sec 3a/3b.

Extends src/pilot_a1_lipschitz.py (single synthetic identity, DiffPure only)
along both axes flagged there as needed before A1 evidence generalizes:
real photos, and a second, structurally distinct architecture for the
discrete union-bound side of the certification.
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
from PIL import Image

from common import (
    load_sd_pipeline, load_face_app, get_empty_text_embedding, get_fixed_timesteps,
    diffpure_pass, arcface_embed, cosine_sim,
)
from gridpure import gridpure


def slopes_of(xs, ys):
    out = []
    for i in range(1, len(xs)):
        if ys[i] is not None and ys[i - 1] is not None:
            out.append(abs(ys[i] - ys[i - 1]) / (xs[i] - xs[i - 1]))
    return out


def _score(face_app, emb_clean, purified):
    """No detectable face => treat as similarity 0, NOT a dropped sample.
    Silently discarding non-detections is survivor bias: it would average
    only the 'lucky' high-purification samples and hide exactly the floor
    effect (purification destroying the image enough to break the attacker's
    own downstream pipeline) that matters for the certification story."""
    emb = arcface_embed(face_app, purified)
    if emb is None:
        return 0.0, False
    return cosine_sim(emb_clean, emb), True


def run_diffpure_sweep(pipe, face_app, identity_img, emb_clean, t_grid, n_seeds,
                        fixed_timesteps, text_emb):
    results = {}
    for t_star in t_grid:
        sims, n_detected = [], 0
        for seed in range(n_seeds):
            purified = diffpure_pass(pipe, identity_img, t_star, seed, fixed_timesteps, text_emb)
            s, detected = _score(face_app, emb_clean, purified)
            sims.append(s)
            n_detected += int(detected)
        results[t_star] = {"mean_sim": float(np.mean(sims)), "std_sim": float(np.std(sims)),
                            "n_detected": n_detected, "n_seeds": n_seeds}
    return results


def run_gridpure_sweep(pipe, face_app, identity_img, emb_clean, r_grid, n_seeds,
                        t_star_round, fixed_timesteps, text_emb):
    results = {}
    for rounds in r_grid:
        sims, n_detected = [], 0
        for seed in range(n_seeds):
            purified = gridpure(pipe, identity_img, rounds, t_star_round, seed,
                                 fixed_timesteps, text_emb)
            s, detected = _score(face_app, emb_clean, purified)
            sims.append(s)
            n_detected += int(detected)
        results[rounds] = {"mean_sim": float(np.mean(sims)), "std_sim": float(np.std(sims)),
                            "n_detected": n_detected, "n_seeds": n_seeds}
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--identities-glob",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "celebahq_sample", "*.png"),
    )
    ap.add_argument("--diffpure-t-grid", type=int, nargs="+",
                     default=[50, 100, 150, 200, 250, 300, 400, 500])
    ap.add_argument("--diffpure-n-seeds", type=int, default=8)
    ap.add_argument("--gridpure-r-grid", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--gridpure-t-star-round", type=int, default=150)
    ap.add_argument("--gridpure-n-seeds", type=int, default=6)
    ap.add_argument("--fixed-steps", type=int, default=50)
    ap.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "a1_multi_pilot.json"),
    )
    args = ap.parse_args()

    identity_paths = sorted(glob.glob(args.identities_glob))
    if not identity_paths:
        raise RuntimeError(f"No identity images found at {args.identities_glob}")
    print(f"Found {len(identity_paths)} identities: {identity_paths}")

    pipe = load_sd_pipeline()
    face_app = load_face_app()
    fixed_timesteps = get_fixed_timesteps(pipe, args.fixed_steps)
    text_emb = get_empty_text_embedding(pipe)

    per_identity = {}
    for path in identity_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        img = Image.open(path).convert("RGB")
        emb_clean = arcface_embed(face_app, img)
        if emb_clean is None:
            print(f"WARNING: no face detected in {path}, skipping")
            continue

        print(f"\n=== {name}: DiffPure sweep ===")
        dp = run_diffpure_sweep(pipe, face_app, img, emb_clean, args.diffpure_t_grid,
                                 args.diffpure_n_seeds, fixed_timesteps, text_emb)
        for t, r in dp.items():
            print(f"  t*={t:4d}  mean_sim={r['mean_sim']:.4f}  std={r['std_sim']:.4f}  "
                  f"detected={r['n_detected']}/{r['n_seeds']}")

        print(f"=== {name}: GrIDPure sweep ===")
        gp = run_gridpure_sweep(pipe, face_app, img, emb_clean, args.gridpure_r_grid,
                                 args.gridpure_n_seeds, args.gridpure_t_star_round,
                                 fixed_timesteps, text_emb)
        for r_, res in gp.items():
            print(f"  R={r_}  mean_sim={res['mean_sim']:.4f}  std={res['std_sim']:.4f}  "
                  f"detected={res['n_detected']}/{res['n_seeds']}")

        per_identity[name] = {"diffpure": dp, "gridpure": gp}

    def aggregate(arch):
        keys = sorted({k for d in per_identity.values() for k in d[arch].keys()})
        agg = {}
        for k in keys:
            vals = [d[arch][k]["mean_sim"] for d in per_identity.values()
                     if d[arch][k]["mean_sim"] is not None]
            agg[k] = float(np.mean(vals)) if vals else None
        return keys, agg

    summary = {"identities": list(per_identity.keys()), "per_identity": per_identity}
    for arch in ("diffpure", "gridpure"):
        keys, agg = aggregate(arch)
        vals = [agg[k] for k in keys]
        sl = slopes_of(keys, vals)
        summary[arch] = {
            "strength_grid": keys,
            "mean_across_identities": agg,
            "empirical_max_slope": max(sl) if sl else None,
            "empirical_mean_slope": float(np.mean(sl)) if sl else None,
        }
        print(f"\n[{arch}] aggregated across {len(per_identity)} identities:")
        for k in keys:
            print(f"  {k}: {agg[k]:.4f}" if agg[k] is not None else f"  {k}: None")
        if sl:
            print(f"  max adjacent-slope={max(sl):.6f}  mean adjacent-slope={np.mean(sl):.6f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
