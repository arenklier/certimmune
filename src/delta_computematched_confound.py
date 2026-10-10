"""TRUE compute/wall-clock-matched comparison (2026-07-24), following
timing_confound.py's measurement: PhotoGuard's 1000-step recipe costs ~55.1s
wall-clock (0.0551 s/step); our EOT-mean-only method costs ~3.3076 s/step, so
a wall-clock-matched budget for ours is ~16.6 steps, not the 30 originally
used (which already exceeds PhotoGuard's full budget by ~1.8x) nor the 100
tested for the step-count-only confound (which uses ~6x PhotoGuard's budget).
This runs the actual compute-matched step count and certifies it, for the
one true apples-to-apples number.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DEVICE, load_sd_pipeline, load_face_app, get_fixed_timesteps
from delta_optimize import load_facenet, pil_to_pixel_tensor, optimize_delta_eot, evaluate_grid
from PIL import Image

STEPS = 17  # wall-clock-matched to PhotoGuard's 1000-step budget (~55.1s / 3.3076s per step)
EPSILON = 8 / 255
THETA_MIN, THETA_MAX, K_THETA = 100, 400, 3
STEP_SIZE = 1 / 255
EVAL_GRID = [100, 150, 200, 250, 300, 350, 400]
EVAL_SEEDS = 5

base = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(base, "..", "results")
identity_path = os.path.join(base, "..", "data", "celebahq_sample", "identity_00.png")

pipe = load_sd_pipeline()
face_app = load_face_app()
facenet = load_facenet()
fixed_timesteps = get_fixed_timesteps(pipe, 50)

text_input = pipe.tokenizer([""], padding="max_length", max_length=pipe.tokenizer.model_max_length,
                             truncation=True, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    text_emb = pipe.text_encoder(text_input.input_ids)[0]

clean_img = Image.open(identity_path).convert("RGB")
clean_pixel = pil_to_pixel_tensor(clean_img)

print(f"=== Compute-matched confound test: v0.2 EOT-mean-only delta, steps={STEPS} "
      f"(wall-clock-matched to PhotoGuard's 1000-step budget) ===")
delta, history = optimize_delta_eot(pipe, facenet, clean_pixel, THETA_MIN, THETA_MAX, K_THETA,
                                     fixed_timesteps, text_emb, EPSILON, STEPS, STEP_SIZE,
                                     var_weight=0.0, seed=0)
adv_pixel = (clean_pixel + delta).clamp(0, 1)
torch.cuda.empty_cache()

print(f"\n=== Evaluating across the FULL Theta grid {EVAL_GRID} (n_seeds={EVAL_SEEDS}) ===")
grid_unprotected = evaluate_grid(pipe, face_app, clean_img, clean_pixel, EVAL_GRID,
                                  fixed_timesteps, text_emb, EVAL_SEEDS)
grid_matched = evaluate_grid(pipe, face_app, clean_img, adv_pixel, EVAL_GRID,
                              fixed_timesteps, text_emb, EVAL_SEEDS)

print(f"\n{'t*':>6} {'unprotected':>12} {'ours-17step':>12}")
for t in EVAL_GRID:
    print(f"{t:>6} {grid_unprotected[t]:>12.4f} {grid_matched[t]:>12.4f}")

worst_unprotected = max(grid_unprotected.values())
worst_matched = max(grid_matched.values())
print(f"\nworst-case: unprotected={worst_unprotected:.4f}  ours-17step(compute-matched)={worst_matched:.4f}  "
      f"photoguard-1000step(reference)=0.3178")

os.makedirs(results_dir, exist_ok=True)
img = Image.fromarray((adv_pixel[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))
img.save(os.path.join(results_dir, "delta_computematched17_immunized.png"))
with open(os.path.join(results_dir, "delta_computematched_confound.json"), "w") as f:
    json.dump({
        "steps": STEPS,
        "rationale": "wall-clock-matched to PhotoGuard's 1000-step budget per timing_confound.json",
        "unprotected": grid_unprotected,
        "ours_matched": grid_matched,
        "worst_case": {"unprotected": worst_unprotected, "ours_matched": worst_matched,
                        "photoguard_1000step_reference": 0.3178},
    }, f, indent=2)
print(f"\nSaved to {results_dir}")
