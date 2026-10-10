"""Extract a few multi-image identities from yixin/metacloak_celeba_vggface2
(the dataset used by the MetaCloak paper -- DreamBooth-protection research,
directly in our subfield). Each identity has 4 images at 512x512 -- the
standard DreamBooth training-set size, fixing the single-image limitation
that stalled src/lora_calibration.py (see certification_framework.md
"Kalibrasyon Pipeline Denemeleri").
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
    ap.add_argument("--n-identities", type=int, default=3)
    ap.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "metacloak_identities"),
    )
    args = ap.parse_args()

    print("Downloading metacloak celeba_gen parquet (~6MB, 46 identities x 4 images)...")
    path = hf_hub_download(
        repo_id="yixin/metacloak_celeba_vggface2",
        filename="data/celeba_gen-00000-of-00001-181dce553293078a.parquet",
        repo_type="dataset",
    )
    table = pq.read_table(path)
    names = table.column("instance_name").to_pylist()
    images = table.column("image").to_pylist()

    face_app = load_face_app()

    order = sorted(set(names), key=int)
    saved_identities = 0
    for inst in order:
        idxs = [i for i, n in enumerate(names) if n == inst]
        out_dir = os.path.join(args.out_dir, f"identity_{inst}")
        os.makedirs(out_dir, exist_ok=True)
        n_detected, saved_paths = 0, []
        for j, idx in enumerate(idxs):
            raw = images[idx]
            b = raw["bytes"] if isinstance(raw, dict) else raw
            img = Image.open(io.BytesIO(b)).convert("RGB")
            emb = arcface_embed(face_app, img)
            if emb is not None:
                n_detected += 1
            out_path = os.path.join(out_dir, f"img_{j}.png")
            img.save(out_path)
            saved_paths.append(out_path)

        print(f"identity_{inst}: {len(idxs)} images, {n_detected} face-detected -> {out_dir}")
        if n_detected == len(idxs):
            saved_identities += 1
        if saved_identities >= args.n_identities:
            break

    print(f"\nDone: extracted identities under {args.out_dir}")


if __name__ == "__main__":
    main()
