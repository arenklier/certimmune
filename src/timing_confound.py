"""Measures real per-step wall-clock cost for PhotoGuard's optimize_photoguard()
vs. our optimize_delta_eot(), to convert the qualitative "roughly two orders of
magnitude" per-step-cost claim (draft_v1.md Limitations #3) into an actual
FLOP/wall-clock-matched comparison. Short, fixed step counts -- timing only,
perturbation quality from these runs is not evaluated or used.
"""
import os
import sys
import time

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DEVICE, load_sd_pipeline, get_fixed_timesteps
from delta_optimize import load_facenet, pil_to_pixel_tensor, optimize_delta_eot
from photoguard_baseline import optimize_photoguard

base = os.path.dirname(os.path.abspath(__file__))
identity_path = os.path.join(base, "..", "data", "celebahq_sample", "identity_00.png")

pipe = load_sd_pipeline()
facenet = load_facenet()
fixed_timesteps = get_fixed_timesteps(pipe, 50)
text_input = pipe.tokenizer([""], padding="max_length", max_length=pipe.tokenizer.model_max_length,
                             truncation=True, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    text_emb = pipe.text_encoder(text_input.input_ids)[0]

clean_img = Image.open(identity_path).convert("RGB")
clean_pixel = pil_to_pixel_tensor(clean_img)

PG_STEPS = 50
OURS_STEPS = 15

torch.cuda.synchronize()
t0 = time.time()
_ = optimize_photoguard(pipe, clean_pixel, 8 / 255, PG_STEPS, 0.01, seed=0)
torch.cuda.synchronize()
t1 = time.time()
pg_total = t1 - t0
pg_per_step = pg_total / PG_STEPS
print(f"PhotoGuard: {PG_STEPS} steps in {pg_total:.2f}s -> {pg_per_step:.4f}s/step")

torch.cuda.empty_cache()
torch.cuda.synchronize()
t0 = time.time()
_ = optimize_delta_eot(pipe, facenet, clean_pixel, 100, 400, 3, fixed_timesteps, text_emb,
                        8 / 255, OURS_STEPS, 1 / 255, var_weight=0.0, seed=0)
torch.cuda.synchronize()
t1 = time.time()
ours_total = t1 - t0
ours_per_step = ours_total / OURS_STEPS
print(f"Ours (EOT): {OURS_STEPS} steps in {ours_total:.2f}s -> {ours_per_step:.4f}s/step")

ratio = ours_per_step / pg_per_step
print(f"\nPer-step cost ratio (ours / PhotoGuard): {ratio:.2f}x")

pg_1000_total = pg_per_step * 1000
print(f"PhotoGuard's 1000-step total wall-clock (extrapolated): {pg_1000_total:.1f}s ({pg_1000_total/60:.2f} min)")

n_matched = pg_1000_total / ours_per_step
print(f"Compute-matched step count for OURS at PhotoGuard's 1000-step wall-clock budget: {n_matched:.1f} steps")

import json
with open(os.path.join(base, "..", "results", "timing_confound.json"), "w") as f:
    json.dump({
        "photoguard_steps": PG_STEPS, "photoguard_total_s": pg_total, "photoguard_per_step_s": pg_per_step,
        "ours_steps": OURS_STEPS, "ours_total_s": ours_total, "ours_per_step_s": ours_per_step,
        "per_step_cost_ratio_ours_over_pg": ratio,
        "photoguard_1000step_total_s_extrapolated": pg_1000_total,
        "compute_matched_steps_for_ours": n_matched,
    }, f, indent=2)
print("\nSaved to ../results/timing_confound.json")
