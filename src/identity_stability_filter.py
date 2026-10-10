"""Cheap, purification-independent pre-filter to flag ArcFace-detector-fragile
identities BEFORE spending GPU time on a full purification sweep.

Motivation (docs/certification_framework.md "Latent-Uzayi Duzeltmesi Sonucu"):
identity_01/03 stayed fragile under GrIDPure even after fixing the real seam
bug -- pointing to an identity-intrinsic property, not an implementation
artifact. If that fragility is already visible from mild, IDENTITY-PRESERVING
nuisance transforms (small crop, JPEG, blur, brightness, h-flip) on the CLEAN
image alone -- with no purification involved at all -- then this becomes a
cheap (one forward pass per transform) screening step for any future,
larger identity pool, instead of an expensive purification sweep per
candidate.

Validation target: does this cheap score correctly separate our known-fragile
(01, 03) from known-robust (00, 02, 04) identities?
"""

import argparse
import glob
import io
import json
import os

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

from common import load_face_app, arcface_embed, cosine_sim


def nuisance_variants(img: Image.Image):
    """Mild, identity-preserving transforms -- NOT purification. Each should
    leave a human-recognizable version of the same photo."""
    variants = {"original": img}

    w, h = img.size
    crop_frac = 0.95
    cw, ch = int(w * crop_frac), int(h * crop_frac)
    left, top = (w - cw) // 2, (h - ch) // 2
    variants["center_crop_95pct"] = img.crop((left, top, left + cw, top + ch)).resize((w, h))

    variants["hflip"] = img.transpose(Image.FLIP_LEFT_RIGHT)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=50)
    buf.seek(0)
    variants["jpeg_q50"] = Image.open(buf).convert("RGB")

    variants["gaussian_blur_r1"] = img.filter(ImageFilter.GaussianBlur(radius=1.2))

    variants["brightness_minus15pct"] = ImageEnhance.Brightness(img).enhance(0.85)
    variants["brightness_plus15pct"] = ImageEnhance.Brightness(img).enhance(1.15)

    return variants


def score_identity(face_app, img: Image.Image):
    variants = nuisance_variants(img)
    emb_orig = arcface_embed(face_app, variants["original"])
    if emb_orig is None:
        return {"detection_rate": 0.0, "min_sim": None, "mean_sim": None,
                "per_variant": {}, "verdict": "FRAGILE (no face in original)"}

    per_variant, sims, n_detected = {}, [], 0
    for name, v_img in variants.items():
        if name == "original":
            per_variant[name] = {"detected": True, "sim": 1.0}
            n_detected += 1
            continue
        emb = arcface_embed(face_app, v_img)
        if emb is None:
            per_variant[name] = {"detected": False, "sim": None}
        else:
            s = cosine_sim(emb_orig, emb)
            per_variant[name] = {"detected": True, "sim": s}
            sims.append(s)
            n_detected += 1

    detection_rate = n_detected / len(variants)
    min_sim = float(min(sims)) if sims else None
    mean_sim = float(np.mean(sims)) if sims else None

    is_stable = detection_rate == 1.0 and min_sim is not None and min_sim >= 0.85
    verdict = "STABLE" if is_stable else "FRAGILE"
    return {"detection_rate": detection_rate, "min_sim": min_sim, "mean_sim": mean_sim,
            "per_variant": per_variant, "verdict": verdict}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--identities-glob",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "celebahq_sample", "*.png"),
    )
    ap.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "identity_stability.json"),
    )
    args = ap.parse_args()

    face_app = load_face_app()
    paths = sorted(glob.glob(args.identities_glob))
    print(f"Screening {len(paths)} identities with mild nuisance transforms "
          f"(NOT purification -- pure detector self-consistency check)\n")

    results = {}
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        img = Image.open(path).convert("RGB")
        r = score_identity(face_app, img)
        results[name] = r
        print(f"{name}: detection_rate={r['detection_rate']:.2f}  "
              f"min_sim={r['min_sim']}  mean_sim={r['mean_sim']}  -> {r['verdict']}")
        for vname, vres in r["per_variant"].items():
            if vname == "original":
                continue
            flag = "" if vres["detected"] else "  <-- FACE LOST"
            sim_str = f"{vres['sim']:.4f}" if vres["sim"] is not None else "None"
            print(f"    {vname:24s} detected={vres['detected']}  sim={sim_str}{flag}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
