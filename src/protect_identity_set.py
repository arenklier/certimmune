"""Apply the v0.2 EOT-mean delta optimization (best-performing variant, see
certification_framework.md "v0.3 ... BEKLENMEDIK OLUMSUZ SONUC") to EACH
image in a multi-image identity set, producing a protected copy of the set.

This closes the loop between delta_optimize.py (so far tested only via raw
ArcFace-on-purified-image) and lora_calibration.py (the real fine-tune +
generate + measure pipeline) -- the first end-to-end test of whether delta
survives the FULL threat-model chain: purify -> DreamBooth fine-tune ->
generate -> measure identity recovery.
"""

import argparse
import glob
import os

import numpy as np
from PIL import Image

from common import load_sd_pipeline, get_fixed_timesteps
from delta_optimize import load_facenet, pil_to_pixel_tensor, optimize_delta_eot


def main():
    ap = argparse.ArgumentParser()
    base = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--identity-dir", default=os.path.join(base, "..", "data", "metacloak_identities", "identity_3"))
    ap.add_argument("--out-dir", default=os.path.join(base, "..", "data", "metacloak_identities", "identity_3_protected"))
    ap.add_argument("--theta-min", type=int, default=100)
    ap.add_argument("--theta-max", type=int, default=400)
    ap.add_argument("--k-theta", type=int, default=3)
    ap.add_argument("--epsilon", type=float, default=8 / 255)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--step-size", type=float, default=1 / 255)
    args = ap.parse_args()

    pipe = load_sd_pipeline()
    facenet = load_facenet()
    fixed_timesteps = get_fixed_timesteps(pipe, 50)

    text_input = pipe.tokenizer([""], padding="max_length", max_length=pipe.tokenizer.model_max_length,
                                 truncation=True, return_tensors="pt").to("cuda")
    import torch
    with torch.no_grad():
        text_emb = pipe.text_encoder(text_input.input_ids)[0]

    paths = sorted(glob.glob(os.path.join(args.identity_dir, "*.png")))
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Protecting {len(paths)} images from {args.identity_dir} -> {args.out_dir}")

    for path in paths:
        name = os.path.basename(path)
        img = Image.open(path).convert("RGB")
        clean_pixel = pil_to_pixel_tensor(img)

        print(f"\n=== {name} ===")
        delta, _ = optimize_delta_eot(pipe, facenet, clean_pixel, args.theta_min, args.theta_max,
                                       args.k_theta, fixed_timesteps, text_emb, args.epsilon,
                                       args.steps, args.step_size, var_weight=0.0)
        adv = (clean_pixel + delta).clamp(0, 1)
        out_arr = (adv[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        Image.fromarray(out_arr).save(os.path.join(args.out_dir, name))
        torch.cuda.empty_cache()

    print(f"\nDone. Protected set saved to {args.out_dir}")


if __name__ == "__main__":
    main()
