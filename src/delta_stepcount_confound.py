"""Step-count confound test (2026-07-24): does our v0.2 EOT-mean-only method's
raw-magnitude gap versus PhotoGuard (0.346 @ 30 steps vs. PhotoGuard's 0.318 @
1000 steps) close if we give our method more optimization steps? PhotoGuard's
per-step cost is a single VAE-encoder forward/backward; ours backprops through
k_theta=3 branches of a partial reverse-diffusion chain per step, so we cannot
reach 1000 comparably cheaply -- this tests 100 steps (~3.3x the original
budget) as a feasible, informative data point, not a full compute-match.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DEVICE, load_sd_pipeline, load_face_app, get_fixed_timesteps
from delta_optimize import (load_facenet, pil_to_pixel_tensor, optimize_delta_eot,
                             evaluate_grid)
from PIL import Image

STEPS = 100
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

print(f"=== Step-count confound test: v0.2 EOT-mean-only delta, steps={STEPS} "
      f"(vs. original 30) ===")
delta, history = optimize_delta_eot(pipe, facenet, clean_pixel, THETA_MIN, THETA_MAX, K_THETA,
                                     fixed_timesteps, text_emb, EPSILON, STEPS, STEP_SIZE,
                                     var_weight=0.0, seed=0)
adv_pixel = (clean_pixel + delta).clamp(0, 1)
torch.cuda.empty_cache()

print(f"\n=== Evaluating across the FULL Theta grid {EVAL_GRID} (n_seeds={EVAL_SEEDS}) ===")
grid_unprotected = evaluate_grid(pipe, face_app, clean_img, clean_pixel, EVAL_GRID,
                                  fixed_timesteps, text_emb, EVAL_SEEDS)
grid_100steps = evaluate_grid(pipe, face_app, clean_img, adv_pixel, EVAL_GRID,
                               fixed_timesteps, text_emb, EVAL_SEEDS)

grid_30steps_prior = {100: 0.3461, 150: 0.2823, 200: 0.2544, 250: 0.2143,
                       300: 0.1909, 350: 0.1774, 400: 0.1246}
photoguard_1000steps_prior = None
pg_path = os.path.join(results_dir, "photoguard_baseline_grid.json")
if os.path.exists(pg_path):
    with open(pg_path) as f:
        photoguard_1000steps_prior = json.load(f)["photoguard"]

print(f"\n{'t*':>6} {'unprotected':>12} {'ours-30step':>12} {'ours-100step':>13} {'photoguard-1000':>16}")
for t in EVAL_GRID:
    pg = photoguard_1000steps_prior.get(str(t)) if photoguard_1000steps_prior else None
    pg_str = f"{pg:.4f}" if pg is not None else "n/a"
    print(f"{t:>6} {grid_unprotected[t]:>12.4f} {grid_30steps_prior[t]:>12.4f} "
          f"{grid_100steps[t]:>13.4f} {pg_str:>16}")

worst_unprotected = max(grid_unprotected.values())
worst_30 = max(grid_30steps_prior.values())
worst_100 = max(grid_100steps.values())
worst_pg = max(photoguard_1000steps_prior.values()) if photoguard_1000steps_prior else None
print(f"\nworst-case: unprotected={worst_unprotected:.4f}  ours-30step={worst_30:.4f}  "
      f"ours-100step={worst_100:.4f}  photoguard-1000step={worst_pg}")

os.makedirs(results_dir, exist_ok=True)
img = Image.fromarray((adv_pixel[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))
img.save(os.path.join(results_dir, "delta_stepcount100_immunized.png"))
with open(os.path.join(results_dir, "delta_stepcount_confound.json"), "w") as f:
    json.dump({
        "steps": STEPS,
        "unprotected": grid_unprotected,
        "ours_30step_prior": grid_30steps_prior,
        "ours_100step": grid_100steps,
        "photoguard_1000step_prior": photoguard_1000steps_prior,
        "worst_case": {"unprotected": worst_unprotected, "ours_30step": worst_30,
                        "ours_100step": worst_100, "photoguard_1000step": worst_pg},
        "history_tail": history[-5:],
    }, f, indent=2)
print(f"\nSaved to {results_dir}")
