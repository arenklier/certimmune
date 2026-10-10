"""GrIDPure certification (certification_framework.md Sec 3b): since R (round
count) is INHERENTLY DISCRETE, no grid+Lipschitz-interpolation is needed --
just a plain union bound (max Clopper-Pearson upper bound over R, i.e. no
grid_residual term at all). This is simpler AND exact (not an approximation),
unlike DiffPure's continuous-sigma case. See certification_framework.md
"GrIDPure Recheck" for why this simplification is valid for this architecture.
"""

import argparse
import json
import os

import numpy as np
from PIL import Image

from common import load_sd_pipeline, load_face_app, get_empty_text_embedding, get_fixed_timesteps, \
    arcface_embed, cosine_sim
from gridpure import gridpure
from certify import clopper_pearson_upper


def sweep_indicator_gridpure(pipe, face_app, img, emb_clean, tau, r_grid, n_seeds, t_star_round,
                              fixed_timesteps, text_emb):
    results = {}
    for rounds in r_grid:
        k = 0
        for seed in range(n_seeds):
            purified = gridpure(pipe, img, rounds, t_star_round, seed, fixed_timesteps, text_emb)
            emb = arcface_embed(face_app, purified)
            sim = cosine_sim(emb_clean, emb) if emb is not None else 0.0
            if sim >= tau:
                k += 1
        results[rounds] = {"k": k, "n": n_seeds, "phat": k / n_seeds}
    return results


def certify_discrete(results, r_grid, alpha_mc, simultaneous=False):
    """No grid_residual term -- R is discrete and r_grid enumerates it exactly,
    so union bound over the K points IS the full Theta, not an approximation."""
    K = len(r_grid)
    alpha_mc_eff = alpha_mc / K if simultaneous else alpha_mc
    cp_upper = {r: clopper_pearson_upper(results[r]["k"], results[r]["n"], alpha_mc_eff) for r in r_grid}
    alpha_final = max(cp_upper.values())
    return {"cp_upper": cp_upper, "alpha_final": alpha_final, "alpha_mc_used": alpha_mc_eff}


def main():
    ap = argparse.ArgumentParser()
    base = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base, "..", "results")
    ap.add_argument("--clean", default=os.path.join(base, "..", "data", "celebahq_sample", "identity_00.png"))
    ap.add_argument("--immunized", default=os.path.join(results_dir, "delta_immunized_eot.png"))
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--r-grid", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--t-star-round", type=int, default=150)
    ap.add_argument("--n-seeds", type=int, default=30)
    ap.add_argument("--alpha-mc", type=float, default=0.05)
    ap.add_argument("--out", default=os.path.join(results_dir, "certificate_gridpure.json"))
    args = ap.parse_args()

    pipe = load_sd_pipeline()
    face_app = load_face_app()
    fixed_timesteps = get_fixed_timesteps(pipe, 50)
    text_emb = get_empty_text_embedding(pipe)

    clean_img = Image.open(args.clean).convert("RGB")
    emb_clean = arcface_embed(face_app, clean_img)
    if emb_clean is None:
        raise RuntimeError("No face detected in clean reference image")

    print(f"=== Certifying UNPROTECTED baseline against GrIDPure (tau={args.tau}, R={args.r_grid}, "
          f"n={args.n_seeds}) ===")
    results_unprotected = sweep_indicator_gridpure(pipe, face_app, clean_img, emb_clean, args.tau, args.r_grid,
                                                    args.n_seeds, args.t_star_round, fixed_timesteps, text_emb)
    for r, res in results_unprotected.items():
        print(f"  R={r}  k/n={res['k']}/{res['n']}  phat={res['phat']:.3f}")

    print(f"\n=== Certifying IMMUNIZED (v0.2 EOT delta, DiffPure-optimized) against GrIDPure "
          f"(BLACK-BOX) ===")
    immunized_img = Image.open(args.immunized).convert("RGB")
    results_immunized = sweep_indicator_gridpure(pipe, face_app, immunized_img, emb_clean, args.tau, args.r_grid,
                                                  args.n_seeds, args.t_star_round, fixed_timesteps, text_emb)
    for r, res in results_immunized.items():
        print(f"  R={r}  k/n={res['k']}/{res['n']}  phat={res['phat']:.3f}")

    cert_u_marginal = certify_discrete(results_unprotected, args.r_grid, args.alpha_mc, simultaneous=False)
    cert_u_simul = certify_discrete(results_unprotected, args.r_grid, args.alpha_mc, simultaneous=True)
    cert_i_marginal = certify_discrete(results_immunized, args.r_grid, args.alpha_mc, simultaneous=False)
    cert_i_simul = certify_discrete(results_immunized, args.r_grid, args.alpha_mc, simultaneous=True)

    print(f"\n--- unprotected: alpha_marginal={cert_u_marginal['alpha_final']:.4f}  "
          f"alpha_simultaneous={cert_u_simul['alpha_final']:.4f} ---")
    print(f"--- immunized (black-box transfer): alpha_marginal={cert_i_marginal['alpha_final']:.4f}  "
          f"alpha_simultaneous={cert_i_simul['alpha_final']:.4f} ---")

    with open(args.out, "w") as f:
        json.dump({
            "tau": args.tau, "r_grid": args.r_grid, "n_seeds": args.n_seeds,
            "unprotected": {"results": results_unprotected, "marginal": cert_u_marginal, "simultaneous": cert_u_simul},
            "immunized": {"results": results_immunized, "marginal": cert_i_marginal, "simultaneous": cert_i_simul},
        }, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
