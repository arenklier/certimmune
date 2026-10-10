"""Test the "Mekanizma-Paylasimi Hipotezi" (mechanism-sharing hypothesis) from
certification_framework.md: does black-box transfer to GrIDPure track how
much representational mechanism each baseline's attack shares with DiffPure?

Prediction: Glaze (CLIP) and Anti-DreamBooth (facenet) -- zero mechanism
overlap with DiffPure's VAE/UNet -- should transfer WEAKLY to GrIDPure.
PhotoGuard (VAE latent) -- partial overlap -- should transfer MODERATELY.
Our EOT delta (full purification chain) -- already confirmed to transfer
strongly (see certification_framework.md "Black-Box Transfer Testi").
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
        sims = []
        for seed in range(n_seeds):
            purified = gridpure(pipe, img, rounds, t_star_round, seed, fixed_timesteps, text_emb)
            emb = arcface_embed(face_app, purified)
            sims.append(cosine_sim(emb_clean, emb) if emb is not None else 0.0)
        results[rounds] = float(np.mean(sims))
    return results


def main():
    ap = argparse.ArgumentParser()
    base = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base, "..", "results")
    ap.add_argument("--clean", default=os.path.join(base, "..", "data", "celebahq_sample", "identity_00.png"))
    ap.add_argument("--methods", nargs="+", default=["glaze", "photoguard", "antidreambooth", "eot"])
    ap.add_argument("--r-grid", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--t-star-round", type=int, default=150)
    ap.add_argument("--n-seeds", type=int, default=15)
    ap.add_argument("--out", default=os.path.join(results_dir, "transfer_all_baselines.json"))
    args = ap.parse_args()

    image_paths = {
        "glaze": os.path.join(results_dir, "glaze_baseline_immunized.png"),
        "photoguard": os.path.join(results_dir, "photoguard_baseline_immunized.png"),
        "antidreambooth": os.path.join(results_dir, "antidreambooth_baseline_immunized.png"),
        "eot": os.path.join(results_dir, "delta_immunized_eot.png"),
        "antipure": os.path.join(results_dir, "antipure_baseline_immunized.png"),
    }

    pipe = load_sd_pipeline()
    face_app = load_face_app()
    fixed_timesteps = get_fixed_timesteps(pipe, 50)
    text_emb = get_empty_text_embedding(pipe)

    clean_img = Image.open(args.clean).convert("RGB")
    emb_clean = arcface_embed(face_app, clean_img)
    if emb_clean is None:
        raise RuntimeError("No face detected in clean reference image")

    print(f"=== Unprotected identity through GrIDPure (R={args.r_grid}, n={args.n_seeds}) ===")
    unprotected = sweep_gridpure(pipe, face_app, clean_img, emb_clean, args.r_grid, args.n_seeds,
                                 args.t_star_round, fixed_timesteps, text_emb)
    for r, v in unprotected.items():
        print(f"  R={r}  mean_sim={v:.4f}")

    all_results = {"unprotected": unprotected}
    for method in args.methods:
        path = image_paths[method]
        if not os.path.exists(path):
            print(f"\nSKIP {method}: {path} not found")
            continue
        print(f"\n=== {method}: through GrIDPure (BLACK-BOX TRANSFER) ===")
        img = Image.open(path).convert("RGB")
        res = sweep_gridpure(pipe, face_app, img, emb_clean, args.r_grid, args.n_seeds,
                              args.t_star_round, fixed_timesteps, text_emb)
        for r, v in res.items():
            print(f"  R={r}  mean_sim={v:.4f}")
        all_results[method] = res

    # summary table: worst-case (max over R) relative drop per method
    print(f"\n{'method':>18} {'worst-case sim':>15} {'relative drop':>15}")
    worst_unprotected = max(unprotected.values())
    print(f"{'unprotected':>18} {worst_unprotected:>15.4f} {'--':>15}")
    summary = {"unprotected_worst_case": worst_unprotected}
    for method in args.methods:
        if method not in all_results:
            continue
        worst = max(all_results[method].values())
        rel_drop = (worst_unprotected - worst) / worst_unprotected
        print(f"{method:>18} {worst:>15.4f} {rel_drop:>14.1%}")
        summary[method] = {"worst_case": worst, "relative_drop": rel_drop}

    with open(args.out, "w") as f:
        json.dump({"per_round": all_results, "summary": summary}, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
