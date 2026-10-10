"""Per-step wall-clock for all three optimizers, on an exclusive GPU.

Sec. IX currently records an asymmetry: the paper's headline wall-clock numbers
were measured with the GPU to ourselves, but fast-EOT's ~1.1 s/step came from a
shared card and is therefore only a conservative upper bound. That is fine for
"early evidence" and not fine for a headline claim, so this re-measures all
three under the same conditions, in one process, with the same pipeline
instance, batch size 1 and synchronize-bracketed timing -- exactly the protocol
timing_confound.py used for the first two.

Run this with nothing else on the card.
"""
import json
import os
import sys
import time

import torch
from PIL import Image

sys.path.insert(0, "/mnt/data/certimmune/src")
from common import DEVICE, load_sd_pipeline, get_fixed_timesteps
from delta_optimize import load_facenet, pil_to_pixel_tensor, optimize_delta_eot
from delta_optimize_fast import optimize_delta_eot_fast
from photoguard_baseline import optimize_photoguard

BASE = "/mnt/data/certimmune"
IDENT = os.path.join(BASE, "data", "celebahq_sample", "identity_00.png")

pipe = load_sd_pipeline()
facenet = load_facenet()
fixed_timesteps = get_fixed_timesteps(pipe, 50)
ti = pipe.tokenizer([""], padding="max_length", max_length=pipe.tokenizer.model_max_length,
                    truncation=True, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    text_emb = pipe.text_encoder(ti.input_ids)[0]
clean_pixel = pil_to_pixel_tensor(Image.open(IDENT).convert("RGB"))


def timed(label, fn, steps, warmup=2):
    """A couple of untimed warm-up steps first: the first CUDA launch of a new
    graph shape pays kernel autotuning that the steady state does not."""
    fn(warmup)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    t0 = time.time()
    fn(steps)
    torch.cuda.synchronize()
    dt = time.time() - t0
    print("%-18s %3d steps in %7.2fs -> %.4f s/step" % (label, steps, dt, dt / steps))
    return dt / steps


pg = timed("PhotoGuard", lambda n: optimize_photoguard(pipe, clean_pixel, 8 / 255, n, 0.01, seed=0), 50)
eot = timed("EOT (full)", lambda n: optimize_delta_eot(
    pipe, facenet, clean_pixel, 100, 400, 3, fixed_timesteps, text_emb, 8 / 255, n, 1 / 255,
    var_weight=0.0, seed=0), 15)
fast = timed("fast-EOT", lambda n: optimize_delta_eot_fast(
    pipe, facenet, clean_pixel, 100, 400, 3, text_emb, 8 / 255, n, 1 / 255, seed=0), 50)

budget = pg * 1000
out = {
    "protocol": "exclusive GPU, single process, one shared fp16 SD1.5 pipeline, batch 1, "
                "synchronize-bracketed, 2 warm-up steps discarded",
    "photoguard_per_step_s": pg,
    "eot_per_step_s": eot,
    "fasteot_per_step_s": fast,
    "eot_over_photoguard": eot / pg,
    "fasteot_over_photoguard": fast / pg,
    "eot_over_fasteot": eot / fast,
    "photoguard_1000step_budget_s": budget,
    "matched_steps_eot": budget / eot,
    "matched_steps_fasteot": budget / fast,
}
print()
for k in ("eot_over_photoguard", "fasteot_over_photoguard", "eot_over_fasteot"):
    print("  %-26s %.1fx" % (k, out[k]))
print("  PhotoGuard 1000-step budget  %.1fs" % budget)
print("  matched steps, EOT           %.1f" % out["matched_steps_eot"])
print("  matched steps, fast-EOT      %.1f" % out["matched_steps_fasteot"])

with open(os.path.join(BASE, "results", "timing_exclusive.json"), "w") as f:
    json.dump(out, f, indent=2)
print("saved results/timing_exclusive.json")
