"""Consolidated one-shot experiment: optimize a v0.2 EOT-mean delta for a
given identity + epsilon, then run the certification pipeline (certify.py's
grid+Lipschitz+Clopper-Pearson math) on it. Built to be invoked repeatedly
across identities and epsilon budgets for the scale-up pass (multi-identity
certification, epsilon-alpha Pareto frontier) -- see README #16/certification_
framework.md.
"""

import argparse
import json
import os

import numpy as np
from PIL import Image

from common import load_sd_pipeline, load_face_app, get_empty_text_embedding, get_fixed_timesteps, \
    diffpure_pass, arcface_embed, cosine_sim
from delta_optimize import pil_to_pixel_tensor, load_facenet, optimize_delta_eot
from certify import sweep_indicator, certify


def main():
    ap = argparse.ArgumentParser()
    base = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base, "..", "results")
    ap.add_argument("--identity", required=True)
    ap.add_argument("--tag", required=True, help="short name for output files, e.g. identity_01_eps8")
    ap.add_argument("--epsilon", type=float, default=8 / 255)
    ap.add_argument("--theta-min", type=int, default=100)
    ap.add_argument("--theta-max", type=int, default=400)
    ap.add_argument("--k-theta", type=int, default=3)
    ap.add_argument("--opt-steps", type=int, default=25)
    ap.add_argument("--opt-step-size", type=float, default=1 / 255)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--t-grid", type=int, nargs="+", default=[100, 150, 200, 250, 300, 350, 400])
    ap.add_argument("--n-seeds", type=int, default=30)
    ap.add_argument("--alpha-mc", type=float, default=0.05)
    args = ap.parse_args()

    pipe = load_sd_pipeline()
    face_app = load_face_app()
    facenet = load_facenet()
    fixed_timesteps = get_fixed_timesteps(pipe, 50)
    text_emb = get_empty_text_embedding(pipe)

    clean_img = Image.open(args.identity).convert("RGB")
    emb_clean = arcface_embed(face_app, clean_img)
    if emb_clean is None:
        raise RuntimeError(f"No face detected in {args.identity}")
    clean_pixel = pil_to_pixel_tensor(clean_img)

    print(f"=== [{args.tag}] Optimizing v0.2 EOT-mean delta "
          f"(epsilon={args.epsilon:.4f}, Theta=[{args.theta_min},{args.theta_max}]) ===")
    delta, _ = optimize_delta_eot(pipe, facenet, clean_pixel, args.theta_min, args.theta_max, args.k_theta,
                                   fixed_timesteps, text_emb, args.epsilon, args.opt_steps, args.opt_step_size,
                                   var_weight=0.0)
    adv_pixel = (clean_pixel + delta).clamp(0, 1)
    immunized_img = Image.fromarray((adv_pixel[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))
    immunized_path = os.path.join(results_dir, f"immunized_{args.tag}.png")
    immunized_img.save(immunized_path)
    import torch
    torch.cuda.empty_cache()

    print(f"\n=== [{args.tag}] Certifying (tau={args.tau}, Theta={args.t_grid}, n={args.n_seeds}) ===")
    print("  unprotected...")
    results_unprotected = sweep_indicator(pipe, face_app, clean_img, emb_clean, args.tau, args.t_grid,
                                           args.n_seeds, fixed_timesteps, text_emb)
    print("  immunized...")
    results_immunized = sweep_indicator(pipe, face_app, immunized_img, emb_clean, args.tau, args.t_grid,
                                         args.n_seeds, fixed_timesteps, text_emb)

    cert_u_marginal = certify(results_unprotected, args.t_grid, args.alpha_mc, simultaneous=False)
    cert_u_simul = certify(results_unprotected, args.t_grid, args.alpha_mc, simultaneous=True)
    cert_i_marginal = certify(results_immunized, args.t_grid, args.alpha_mc, simultaneous=False)
    cert_i_simul = certify(results_immunized, args.t_grid, args.alpha_mc, simultaneous=True)

    print(f"\n--- [{args.tag}] unprotected: alpha_marginal={cert_u_marginal['alpha_final']:.4f}  "
          f"alpha_simultaneous={cert_u_simul['alpha_final']:.4f} ---")
    print(f"--- [{args.tag}] immunized:   alpha_marginal={cert_i_marginal['alpha_final']:.4f}  "
          f"alpha_simultaneous={cert_i_simul['alpha_final']:.4f} ---")

    out = {
        "tag": args.tag, "identity": args.identity, "epsilon": args.epsilon,
        "theta": [args.theta_min, args.theta_max], "tau": args.tau, "t_grid": args.t_grid,
        "n_seeds": args.n_seeds,
        "unprotected": {"results": results_unprotected, "marginal": cert_u_marginal, "simultaneous": cert_u_simul},
        "immunized": {"results": results_immunized, "marginal": cert_i_marginal, "simultaneous": cert_i_simul},
    }
    out_path = os.path.join(results_dir, f"certificate_{args.tag}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
