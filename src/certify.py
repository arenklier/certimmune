"""The certification pipeline itself (certification_framework.md Sec 3):
grid + Lipschitz-in-sigma + Clopper-Pearson + uncertainty combination,
producing an actual (epsilon, tau, Theta, alpha)-style certificate for our
v0.2 EOT-mean delta, with the unprotected baseline as a contrasting worked
example.

SCOPE (honest, load-bearing decision): this certifies on the CHEAP PROXY
(purified-image ArcFace similarity to the clean identity), NOT the full
real-attack-success chain (fine-tune + generate + measure). Running n x K
Monte-Carlo samples of the REAL chain (each ~10-15min of LoRA fine-tuning)
would need hundreds of fine-tune runs -- infeasible. This is exactly why
Sec 5 designed a two-tier proxy+calibration-spot-check structure: the
calibration study (already run, see "Kalibrasyon Pipeline Denemeleri" /
"n=20 Sonucu" / "TAM ZINCIR Dogrulamasi") establishes that the proxy tracks
real success in DIRECTION (protection always reduces real success) though
not tightly in MAGNITUDE (variable across identities) -- so a proxy-level
certificate here is an honest, correctly-scoped deliverable, not a
substitute for a real-attack-level one.

Math (Sec 3a/3c):
  h_a(sigma) := E_eps[ 1[ S(purify(x,sigma,eps)) >= tau ] ]   (Sec 2)
  grid t* in Theta at K points, n Monte-Carlo seeds each
  Clopper-Pearson upper bound at each point (one-sided, confidence 1-alpha_MC)
  empirical Lipschitz-in-sigma L_sigma from adjacent-point slopes
  alpha_final = max_k CP_upper(t_k) + L_sigma_hat * |Theta| / (2(K-1))
"""

import argparse
import json
import os

import numpy as np
from PIL import Image
from scipy.stats import beta as beta_dist

from common import load_sd_pipeline, load_face_app, get_empty_text_embedding, get_fixed_timesteps, \
    diffpure_pass, arcface_embed, cosine_sim


def clopper_pearson_upper(k, n, alpha_mc):
    """One-sided upper confidence bound at level 1-alpha_mc for a Binomial(n,p)
    with k observed successes. Standard exact (Clopper-Pearson) formula."""
    if k == n:
        return 1.0
    return float(beta_dist.ppf(1 - alpha_mc, k + 1, n - k))


def sweep_indicator(pipe, face_app, img, emb_clean, tau, t_grid, n_seeds, fixed_timesteps, text_emb):
    """Returns, per grid point: (successes k, trials n, point-estimate proportion)."""
    results = {}
    for t_star in t_grid:
        k = 0
        for seed in range(n_seeds):
            purified = diffpure_pass(pipe, img, t_star, seed, fixed_timesteps, text_emb)
            emb = arcface_embed(face_app, purified)
            sim = cosine_sim(emb_clean, emb) if emb is not None else 0.0
            if sim >= tau:
                k += 1
        results[t_star] = {"k": k, "n": n_seeds, "phat": k / n_seeds}
    return results


def certify(results, t_grid, alpha_mc, simultaneous=False):
    """Sec 3a/3c: grid+Lipschitz+Clopper-Pearson+(optional Bonferroni).
    simultaneous=True claims coverage for ALL K grid points AT ONCE (needs
    Bonferroni: alpha_mc/K per point); simultaneous=False (default) claims
    only MARGINAL coverage per-point, no correction needed -- Sec 3c requires
    stating explicitly which is being claimed, so this is not a silent default."""
    K = len(t_grid)
    alpha_mc_eff = alpha_mc / K if simultaneous else alpha_mc

    cp_upper = {}
    for t in t_grid:
        r = results[t]
        cp_upper[t] = clopper_pearson_upper(r["k"], r["n"], alpha_mc_eff)

    # empirical Lipschitz-in-sigma: max adjacent-slope of the POINT ESTIMATES (phat), a practical
    # (not yet fully rigorous -- see docs) stand-in for the true L_sigma this pilot needs to validate
    sorted_t = sorted(t_grid)
    slopes = []
    for i in range(1, len(sorted_t)):
        dt = sorted_t[i] - sorted_t[i - 1]
        dphat = abs(results[sorted_t[i]]["phat"] - results[sorted_t[i - 1]]["phat"])
        slopes.append(dphat / dt)
    L_sigma_hat = max(slopes) if slopes else 0.0

    theta_width = sorted_t[-1] - sorted_t[0]
    grid_residual = L_sigma_hat * theta_width / (2 * (K - 1)) if K > 1 else 0.0

    max_cp_upper = max(cp_upper.values())
    alpha_final = max_cp_upper + grid_residual

    return {
        "cp_upper": cp_upper,
        "L_sigma_hat": L_sigma_hat,
        "grid_residual": grid_residual,
        "max_cp_upper": max_cp_upper,
        "alpha_final": alpha_final,
        "alpha_mc_used": alpha_mc_eff,
    }


def main():
    ap = argparse.ArgumentParser()
    base = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base, "..", "results")
    ap.add_argument("--clean", default=os.path.join(base, "..", "data", "celebahq_sample", "identity_00.png"))
    ap.add_argument("--immunized", default=os.path.join(results_dir, "delta_immunized_eot.png"))
    ap.add_argument("--tau", type=float, default=0.5,
                     help="PROXY-level threshold (purified-image ArcFace sim) -- see script docstring for scope")
    ap.add_argument("--t-grid", type=int, nargs="+", default=[100, 150, 200, 250, 300, 350, 400])
    ap.add_argument("--n-seeds", type=int, default=30)
    ap.add_argument("--alpha-mc", type=float, default=0.05)
    ap.add_argument("--bonferroni", action="store_true", help="claim simultaneous (not just marginal) coverage")
    ap.add_argument("--skip-unprotected", action="store_true",
                     help="skip the unprotected sweep (it is deterministic and identical for every "
                          "method on the same identity, so one run per identity suffices); the JSON "
                          "then records unprotected=null and the arm must be read from the companion "
                          "certificate that did measure it")
    ap.add_argument("--out", default=os.path.join(results_dir, "certificate.json"))
    args = ap.parse_args()

    pipe = load_sd_pipeline()
    face_app = load_face_app()
    fixed_timesteps = get_fixed_timesteps(pipe, 50)
    text_emb = get_empty_text_embedding(pipe)

    clean_img = Image.open(args.clean).convert("RGB")
    emb_clean = arcface_embed(face_app, clean_img)
    if emb_clean is None:
        raise RuntimeError("No face detected in clean reference image")

    if args.skip_unprotected:
        print("=== SKIPPING unprotected baseline (--skip-unprotected): deterministic, "
              "measured once per identity ===")
        results_unprotected = None
        cert_unprotected = None
    else:
        print(f"=== Certifying UNPROTECTED baseline (tau={args.tau}, Theta={args.t_grid}, n={args.n_seeds}) ===")
        results_unprotected = sweep_indicator(pipe, face_app, clean_img, emb_clean, args.tau, args.t_grid,
                                               args.n_seeds, fixed_timesteps, text_emb)
        for t, r in results_unprotected.items():
            print(f"  t*={t:4d}  k/n={r['k']}/{r['n']}  phat={r['phat']:.3f}")
        cert_unprotected = certify(results_unprotected, args.t_grid, args.alpha_mc, simultaneous=args.bonferroni)

    print(f"\n=== Certifying IMMUNIZED (v0.2 EOT delta) (tau={args.tau}, Theta={args.t_grid}, n={args.n_seeds}) ===")
    immunized_img = Image.open(args.immunized).convert("RGB")
    results_immunized = sweep_indicator(pipe, face_app, immunized_img, emb_clean, args.tau, args.t_grid,
                                         args.n_seeds, fixed_timesteps, text_emb)
    for t, r in results_immunized.items():
        print(f"  t*={t:4d}  k/n={r['k']}/{r['n']}  phat={r['phat']:.3f}")
    cert_immunized = certify(results_immunized, args.t_grid, args.alpha_mc, simultaneous=args.bonferroni)

    def report(name, cert):
        print(f"\n--- Certificate: {name} ---")
        print(f"  Clopper-Pearson upper bounds per t*: "
              f"{ {t: round(v,3) for t,v in cert['cp_upper'].items()} }")
        print(f"  max CP upper bound (over grid):      {cert['max_cp_upper']:.4f}")
        print(f"  empirical L_sigma_hat:                {cert['L_sigma_hat']:.6f}")
        print(f"  grid-Lipschitz residual term:         {cert['grid_residual']:.4f}")
        print(f"  ==> alpha_final = {cert['alpha_final']:.4f}")
        print(f"  ==> this is a (epsilon=8/255, tau={args.tau}, Theta={args.t_grid}, "
              f"alpha={cert['alpha_final']:.4f})-certified claim")

    if cert_unprotected is not None:
        report("unprotected", cert_unprotected)
    report("immunized (v0.2 EOT delta)", cert_immunized)

    with open(args.out, "w") as f:
        json.dump({
            "tau": args.tau, "t_grid": args.t_grid, "n_seeds": args.n_seeds, "alpha_mc": args.alpha_mc,
            "unprotected": (None if cert_unprotected is None
                             else {"results": results_unprotected, "certificate": cert_unprotected}),
            "immunized": {"results": results_immunized, "certificate": cert_immunized},
        }, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
