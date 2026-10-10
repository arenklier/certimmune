"""Re-check GrIDPure at n_seeds=30 (up from 6) to distinguish true A1
(Lipschitz-in-R) non-smoothness from Monte-Carlo sampling noise -- the open
question flagged in docs/certification_framework.md "Pilot Genislemesi".

At n=6 with observed std ~0.2-0.34, the standard error of the mean was
~0.08-0.14, the same order of magnitude as the adjacent-point differences
being measured -- so the earlier "jumps" could not be told apart from noise.
This script adds an explicit z-score per adjacent pair:
  z = |h(R_i) - h(R_{i-1})| / sqrt(SEM_i^2 + SEM_{i-1}^2)
z >~ 2 => likely a real jump (not explained by sampling noise alone)
z <~ 1 => consistent with pure Monte-Carlo noise
"""

import argparse
import glob
import json
import os

import numpy as np
from PIL import Image

from common import (
    load_sd_pipeline, load_face_app, get_empty_text_embedding, get_fixed_timesteps,
    arcface_embed, cosine_sim,
)
from gridpure import gridpure


def score(face_app, emb_clean, purified):
    emb = arcface_embed(face_app, purified)
    if emb is None:
        return 0.0, False
    return cosine_sim(emb_clean, emb), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--identities-glob",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "celebahq_sample", "*.png"),
    )
    ap.add_argument("--r-grid", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--t-star-round", type=int, default=150)
    ap.add_argument("--n-seeds", type=int, default=30)
    ap.add_argument("--fixed-steps", type=int, default=50)
    ap.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "gridpure_recheck.json"),
    )
    args = ap.parse_args()

    identity_paths = sorted(glob.glob(args.identities_glob))
    print(f"Found {len(identity_paths)} identities, n_seeds={args.n_seeds}, "
          f"R-grid={args.r_grid}")

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
            print(f"WARNING: no face in {path}, skipping")
            continue

        print(f"\n=== {name} ===")
        results = {}
        for rounds in args.r_grid:
            sims, n_detected = [], 0
            for seed in range(args.n_seeds):
                purified = gridpure(pipe, img, rounds, args.t_star_round, seed,
                                     fixed_timesteps, text_emb)
                s, detected = score(face_app, emb_clean, purified)
                sims.append(s)
                n_detected += int(detected)
            mean = float(np.mean(sims))
            std = float(np.std(sims))
            sem = std / np.sqrt(args.n_seeds)
            results[rounds] = {"mean_sim": mean, "std_sim": std, "sem": sem,
                                "n_detected": n_detected, "n_seeds": args.n_seeds}
            print(f"  R={rounds}  mean={mean:.4f}  std={std:.4f}  sem={sem:.4f}  "
                  f"detected={n_detected}/{args.n_seeds}")
        per_identity[name] = results

    # aggregate + per-adjacent-pair z-score (is the observed drop bigger than sampling noise?)
    r_grid = args.r_grid
    agg_mean, agg_sem = {}, {}
    for r in r_grid:
        means = [per_identity[k][r]["mean_sim"] for k in per_identity]
        sems = [per_identity[k][r]["sem"] for k in per_identity]
        agg_mean[r] = float(np.mean(means))
        # combined SEM across identities (mean of per-identity SEMs, conservative)
        agg_sem[r] = float(np.mean(sems))

    print("\n=== Aggregated across identities, with z-score per adjacent pair ===")
    z_scores = []
    for i in range(1, len(r_grid)):
        r0, r1 = r_grid[i - 1], r_grid[i]
        diff = abs(agg_mean[r1] - agg_mean[r0])
        combined_sem = np.sqrt(agg_sem[r0] ** 2 + agg_sem[r1] ** 2)
        z = diff / combined_sem if combined_sem > 0 else float("inf")
        z_scores.append(z)
        verdict = "REAL JUMP (z>=2)" if z >= 2 else ("ambiguous (1<=z<2)" if z >= 1 else "consistent with noise (z<1)")
        print(f"  R={r0}->{r1}: h={agg_mean[r0]:.4f}->{agg_mean[r1]:.4f}  diff={diff:.4f}  "
              f"combined_sem={combined_sem:.4f}  z={z:.2f}  [{verdict}]")

    summary = {
        "n_seeds": args.n_seeds, "r_grid": r_grid,
        "per_identity": per_identity,
        "aggregated_mean": agg_mean, "aggregated_sem": agg_sem,
        "adjacent_z_scores": z_scores,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
