"""Download the PhilSad/celeba-hq-1.5k parquet (146MB -- a small subset mirror,
NOT the full ~90GB CelebA-HQ) and extract N sample images as PNGs, keeping
only faces ArcFace can actually detect (real photos, so this should mostly
just work, unlike the SD-generated placeholder in the single-identity pilot).
"""

import argparse
import io
import os
import sys

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from common import load_face_app, arcface_embed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "celebahq_sample"),
    )
    args = ap.parse_args()

    print("Downloading PhilSad/celeba-hq-1.5k parquet (~146MB)...")
    path = hf_hub_download(
        repo_id="PhilSad/celeba-hq-1.5k",
        filename="data/train-00000-of-00001-b0da07636178f57e.parquet",
        repo_type="dataset",
    )
    table = pq.read_table(path)
    print("Columns:", table.column_names, "rows:", table.num_rows)

    image_col = "image" if "image" in table.column_names else table.column_names[0]
    images = table.column(image_col).to_pylist()

    face_app = load_face_app()
    os.makedirs(args.out_dir, exist_ok=True)

    saved = 0
    rng = np.random.default_rng(0)
    order = rng.permutation(len(images))
    for idx in order:
        raw = images[idx]
        b = raw["bytes"] if isinstance(raw, dict) else raw
        try:
            img = Image.open(io.BytesIO(b)).convert("RGB")
        except Exception:
            continue
        emb = arcface_embed(face_app, img)
        if emb is None:
            continue
        out_path = os.path.join(args.out_dir, f"identity_{saved:02d}.png")
        img.save(out_path)
        print(f"saved {out_path}  (source row {idx}, size {img.size})")
        saved += 1
        if saved >= args.n:
            break

    print(f"\nDone: {saved}/{args.n} face-verified real identity images saved to {args.out_dir}")


if __name__ == "__main__":
    main()
