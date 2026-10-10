"""Extend the CelebA-HQ identity pool without touching the 21 already extracted.

extract_celebahq_sample.py walks a seed-0 permutation and writes identity_00
upward, so re-running it with a larger --n would rewrite the existing files.
They are the inputs to every certificate in the paper, so instead this script
replays the same permutation, checks that the first 21 it would produce are
byte-identical to the ones on disk, and only then writes the new ones.

If the check fails, nothing is written: a mismatch would mean the dataset
revision or the detector changed under us, which is a finding in itself and not
something to paper over by overwriting.
"""
import argparse
import hashlib
import io
import os
import sys

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

sys.path.insert(0, "/mnt/data/certimmune/src")
from common import load_face_app, arcface_embed

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=60, help="total identities wanted")
ap.add_argument("--out-dir", default="/mnt/data/certimmune/data/celebahq_sample")
args = ap.parse_args()


def sha(b):
    return hashlib.sha256(b).hexdigest()[:16]


path = hf_hub_download(repo_id="PhilSad/celeba-hq-1.5k",
                       filename="data/train-00000-of-00001-b0da07636178f57e.parquet",
                       repo_type="dataset")
table = pq.read_table(path)
col = "image" if "image" in table.column_names else table.column_names[0]
images = table.column(col).to_pylist()
face_app = load_face_app()

rng = np.random.default_rng(0)
order = rng.permutation(len(images))

existing = sorted(f for f in os.listdir(args.out_dir)
                  if f.startswith("identity_") and f.endswith(".png") and "_" not in f[9:])
n_existing = len(existing)
print("already on disk: %d identities" % n_existing)

pending, saved, checked, mismatches = [], 0, 0, []
for idx in order:
    raw = images[idx]
    b = raw["bytes"] if isinstance(raw, dict) else raw
    try:
        img = Image.open(io.BytesIO(b)).convert("RGB")
    except Exception:
        continue
    if arcface_embed(face_app, img) is None:
        continue
    name = "identity_%02d.png" % saved
    out = os.path.join(args.out_dir, name)
    if saved < n_existing:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        on_disk = open(out, "rb").read() if os.path.exists(out) else b""
        same = sha(buf.getvalue()) == sha(on_disk)
        checked += 1
        if not same:
            mismatches.append(name)
    else:
        pending.append((out, img, idx))
    saved += 1
    if saved >= args.n:
        break

print("reproduced and compared the first %d: %d mismatch(es) %s"
      % (checked, len(mismatches), mismatches[:5]))
if mismatches:
    print("REFUSING to write: the extraction no longer reproduces what the paper used.")
    sys.exit(1)

for out, img, idx in pending:
    img.save(out)
print("wrote %d new identities (%s .. %s)"
      % (len(pending), os.path.basename(pending[0][0]) if pending else "-",
         os.path.basename(pending[-1][0]) if pending else "-"))
