"""Stress-test the EOT and PhotoGuard perturbations against an off-the-shelf
super-resolution round-trip -- the specific attack Hoenig et al. (ICLR 2025)
show breaks protective perturbations that were never optimized against it.

This is deliberately NOT run through certify.py: an upscale-then-downscale
pipeline is not of the form D_theta(x+eps) this paper's threat model covers
(Sec. III), so nothing here is certified against. It is the same kind of raw
ArcFace-similarity measurement Table II already reports for the baselines,
just run through one more preprocessing step before scoring, to measure
whether protection outside the certified family A survives in practice.

Protocol: for each identity, the SD x4 upscaler (stabilityai/stable-diffusion-
x4-upscaler, the "off-the-shelf" tool the attack paper itself uses) is run on
the 512x512 protected image, then the 2048x2048 result is downsampled back to
512x512 (bicubic) so ArcFace scores it on the same scale as every other
number in this paper. Clean images go through the identical round-trip as a
control, since a R upscale pipeline can shift identity on its own.
"""
import argparse
import io
import json
import os
import sys
import time

import numpy as np
import torch
from diffusers import StableDiffusionUpscalePipeline
from PIL import Image

sys.path.insert(0, "/mnt/data/certimmune/src")
import common  # noqa: E402

ROOT = "/mnt/data/certimmune"
CLEAN = os.path.join(ROOT, "data/celebahq_sample/identity_%s.png")
R = os.path.join(ROOT, "results")
OUT = os.path.join(R, "upscale_attack.json")

UPSCALER_ID = "stabilityai/stable-diffusion-x4-upscaler"


def load_upscaler():
    pipe = StableDiffusionUpscalePipeline.from_pretrained(
        UPSCALER_ID, torch_dtype=torch.float16
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def upscale_roundtrip(pipe, img, generator):
    # SD x4 upscaler conditions on a short caption; empty prompt matches how
    # an attacker with no knowledge of the image's content would actually use
    # this tool off the shelf.
    low_res = img.convert("RGB").resize((128, 128), Image.BICUBIC)
    with torch.no_grad():
        up = pipe(prompt="", image=low_res, num_inference_steps=20,
                   generator=generator).images[0]
    return up.resize((512, 512), Image.BICUBIC)


def load512(path):
    return Image.open(path).convert("RGB").resize((512, 512), Image.BICUBIC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--identities", default="00,01,02,04,05,06,07,08,09,10,"
                                             "11,12,13,14,15,16,17,18,19,20")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    ids = args.identities.split(",")

    face_app = common.load_face_app()
    pipe = load_upscaler()
    generator = torch.Generator("cuda").manual_seed(args.seed)

    results = {}
    if args.resume and os.path.exists(OUT):
        results = json.load(open(OUT))

    for ident in ids:
        if ident in results:
            print("skip %s (already done)" % ident, flush=True)
            continue
        clean_path = CLEAN % ident
        eot_path = os.path.join(R, "immunized_identity_%s_eps8_n20.png" % ident if ident == "00" else "immunized_identity_%s_eps8.png" % ident)
        pg_path = os.path.join(R, "photoguard_baseline_%s_immunized.png" % ident)
        if not (os.path.exists(clean_path) and os.path.exists(eot_path)):
            print("SKIP %s: missing source image(s)" % ident, flush=True)
            continue

        t0 = time.time()
        clean = load512(clean_path)
        clean_emb = common.arcface_embed(face_app, clean)
        if clean_emb is None:
            print("SKIP %s: no face detected in clean image" % ident, flush=True)
            continue

        row = {}
        for tag, path in (("eot", eot_path), ("photoguard", pg_path)):
            if not os.path.exists(path):
                continue
            protected = load512(path)
            pre_emb = common.arcface_embed(face_app, protected)
            pre_sim = common.cosine_sim(clean_emb, pre_emb) if pre_emb is not None else None

            attacked = upscale_roundtrip(pipe, protected, generator)
            post_emb = common.arcface_embed(face_app, attacked)
            post_sim = common.cosine_sim(clean_emb, post_emb) if post_emb is not None else None

            row[tag] = {"pre_attack_sim": pre_sim, "post_attack_sim": post_sim,
                        "face_detected_post": post_emb is not None}
            print("  %s/%s: pre=%s post=%s" % (ident, tag, pre_sim, post_sim), flush=True)

        # control: does the round-trip alone move a CLEAN image's similarity to itself?
        clean_rt = upscale_roundtrip(pipe, clean, generator)
        clean_rt_emb = common.arcface_embed(face_app, clean_rt)
        row["clean_control"] = {
            "post_attack_sim": common.cosine_sim(clean_emb, clean_rt_emb) if clean_rt_emb is not None else None
        }

        results[ident] = row
        json.dump(results, open(OUT, "w"), indent=1)
        print("identity_%s done in %.1fs (%d/%d)" % (ident, time.time() - t0, len(results), len(ids)), flush=True)

    print("\nwrote %s (%d identities)" % (OUT, len(results)))


if __name__ == "__main__":
    main()
