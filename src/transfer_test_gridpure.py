"""Black-box transfer test: does the delta optimized AGAINST DiffPure (v0.2,
mean-only EOT -- the best-performing variant so far, see certification_
framework.md "v0.3 ... BEKLENMEDIK OLUMSUZ SONUC") also protect against
GrIDPure, a structurally different, UNSEEN-during-optimization purifier?

This is exactly the open gap AntiPure (ICCV 2025) itself flags: their
perturbation is white-box-only with "limited black-box transfer" left as
future work. A positive transfer result here would be a genuine, sharp
differentiator for this project's contribution.
"""

import argparse
import json
import os

import numpy as np
from PIL import Image

from common import load_sd_pipeline, load_face_app, get_empty_text_embedding, get_fixed_timesteps, \
    arcface_embed, cosine_sim
from gridpure import gridpure


def sweep_gridpure(pipe, face_app, img, emb_clean, r_grid, n_seeds, t_star_round, fixed_timesteps, text_emb):
    results = {}
    for rounds in r_grid:
        sims, n_detected = [], 0
        for seed in range(n_seeds):
            purified = gridpure(pipe, img, rounds, t_star_round, seed, fixed_timesteps, text_emb)
            emb = arcface_embed(face_app, purified)
            if emb is not None:
                sims.append(cosine_sim(emb_clean, emb))
                n_detected += 1
            else:
                sims.append(0.0)
        results[rounds] = {"mean_sim": float(np.mean(sims)), "n_detected": n_detected, "n_seeds": n_seeds}
    return results


def main():
    ap = argparse.ArgumentParser()
    base = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--clean", default=os.path.join(base, "..", "data", "celebahq_sample", "identity_00.png"))
    ap.add_argument("--immunized", default=os.path.join(base, "..", "results", "delta_immunized_eot.png"))
    ap.add_argument("--r-grid", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--t-star-round", type=int, default=150)
    ap.add_argument("--n-seeds", type=int, default=15)
    ap.add_argument("--out", default=os.path.join(base, "..", "results", "transfer_gridpure.json"))
    args = ap.parse_args()

    pipe = load_sd_pipeline()
    face_app = load_face_app()
    fixed_timesteps = get_fixed_timesteps(pipe, 50)
    text_emb = get_empty_text_embedding(pipe)

    clean_img = Image.open(args.clean).convert("RGB")
    immunized_img = Image.open(args.immunized).convert("RGB")
    emb_clean = arcface_embed(face_app, clean_img)
    if emb_clean is None:
        raise RuntimeError("No face detected in clean reference image")

    print(f"=== Unprotected identity through GrIDPure (R={args.r_grid}, n={args.n_seeds}) ===")
    results_unprotected = sweep_gridpure(pipe, face_app, clean_img, emb_clean, args.r_grid, args.n_seeds,
                                          args.t_star_round, fixed_timesteps, text_emb)
    for r, res in results_unprotected.items():
        print(f"  R={r}  mean_sim={res['mean_sim']:.4f}  detected={res['n_detected']}/{res['n_seeds']}")

    print(f"\n=== DiffPure-optimized delta (v0.2 EOT-mean) through GrIDPure (BLACK-BOX TRANSFER) ===")
    results_immunized = sweep_gridpure(pipe, face_app, immunized_img, emb_clean, args.r_grid, args.n_seeds,
                                        args.t_star_round, fixed_timesteps, text_emb)
    for r, res in results_immunized.items():
        print(f"  R={r}  mean_sim={res['mean_sim']:.4f}  detected={res['n_detected']}/{res['n_seeds']}")

    print(f"\n{'R':>4} {'unprotected':>12} {'immunized (transfer)':>22} {'relative drop':>14}")
    summary = {}
    for r in args.r_grid:
        u = results_unprotected[r]["mean_sim"]
        i = results_immunized[r]["mean_sim"]
        rel_drop = (u - i) / u if u != 0 else float("nan")
        print(f"{r:>4} {u:>12.4f} {i:>22.4f} {rel_drop:>13.1%}")
        summary[r] = {"unprotected": u, "immunized": i, "relative_drop": rel_drop}

    with open(args.out, "w") as f:
        json.dump({"unprotected": results_unprotected, "immunized": results_immunized, "summary": summary}, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
