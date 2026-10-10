"""delta optimization (certification_framework.md Sec 4): build an immunizing
perturbation delta, ||delta||_inf <= epsilon, that pushes the identity
embedding of a PURIFIED image away from the clean identity -- i.e., make
DiffPure's own reverse-denoising process reconstruct something that no
longer matches the person, backpropagating gradients through the whole
encode -> add-noise -> UNet-denoise-loop -> decode chain.

Differentiability note: insightface's ArcFace (onnxruntime) is NOT part of
the torch autograd graph, so it CANNOT be used inside this optimization loop
-- it stays the certification/evaluation metric only. facenet-pytorch's
InceptionResnetV1 (VGGFace2-pretrained, pure PyTorch) is the differentiable
identity-similarity SURROGATE used here during optimization, matching how
prior protective-perturbation work (Anti-DreamBooth family) builds its loss.

v0.1: single fixed theta (t_star) -- first sanity check that gradients
through the purification pipeline actually push the identity embedding away
at all. RESULT: worked (ArcFace 0.625 -> 0.360 at the trained-on t*=200).

v0.2 (2026-07-22): adds EOT-over-Theta + theta-variance penalty (Sec 4's
actual objective). Each step samples k random t* from a range Theta=[t_min,
t_max] instead of one fixed t*, and penalizes variance across those samples:
  loss = mean_theta[ sim(purified, clean) ] + var_weight * var_theta[ sim ]
Motivation: v0.1's delta was optimized against ONE t* -- nothing guarantees
it protects at OTHER t* an adaptive attacker might pick. The key comparison
this version runs: evaluate BOTH the v0.1 (single-theta) delta and the new
EOT delta across a full GRID spanning Theta, to see whether EOT actually
buys more uniform protection across the range (the whole point of building
a certificate over sup_theta rather than one empirical stress-test point).
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
from PIL import Image

from common import DEVICE, load_sd_pipeline, load_face_app, get_fixed_timesteps, arcface_embed, cosine_sim


def load_facenet():
    return InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)


def facenet_embed(model, img_01):
    """img_01: (1,3,H,W) tensor in [0,1], differentiable end-to-end."""
    x = F.interpolate(img_01, size=(160, 160), mode="bilinear", align_corners=False)
    x = (x * 255.0 - 127.5) / 128.0  # facenet-pytorch's fixed_image_standardization, in [0,1]-input terms
    return model(x)


def pil_to_pixel_tensor(img: Image.Image, size=512):
    img = img.resize((size, size))
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)


def pixel_to_vae_input(x_01):
    return (x_01 * 2.0 - 1.0).to(torch.float16)


def differentiable_diffpure(pipe, x_01, t_star, fixed_timesteps, text_emb, noise):
    """Same math as common.diffpure_pass, but WITHOUT @torch.no_grad() so
    gradients flow from the output pixels all the way back to x_01 (and thus
    to delta, if x_01 = clean + delta upstream)."""
    scheduler = pipe.scheduler
    vae_in = pixel_to_vae_input(x_01)
    latent = pipe.vae.encode(vae_in).latent_dist.mean * pipe.vae.config.scaling_factor  # .mean, not .sample(): deterministic, keeps graph simpler

    t_star_tensor = torch.tensor([t_star], device=DEVICE)
    lat = scheduler.add_noise(latent, noise, t_star_tensor)

    reverse_timesteps = fixed_timesteps[fixed_timesteps <= t_star]
    if len(reverse_timesteps) == 0:
        reverse_timesteps = fixed_timesteps[-1:]
    for t in reverse_timesteps:
        noise_pred = pipe.unet(lat, t, encoder_hidden_states=text_emb).sample
        lat = scheduler.step(noise_pred, t, lat).prev_sample

    lat = lat / pipe.vae.config.scaling_factor
    out = pipe.vae.decode(lat).sample
    out_01 = (out.float() / 2 + 0.5).clamp(0, 1)
    return out_01


def optimize_delta(pipe, facenet, clean_pixel, t_star, fixed_timesteps, text_emb,
                    epsilon, steps, step_size, seed=0):
    delta = torch.zeros_like(clean_pixel, requires_grad=True)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)

    with torch.no_grad():
        clean_emb = facenet_embed(facenet, clean_pixel).detach()

    history = []
    for step in range(steps):
        x_adv = (clean_pixel + delta).clamp(0, 1)
        noise = torch.randn(1, 4, 64, 64, generator=gen, device=DEVICE, dtype=torch.float16)

        purified = differentiable_diffpure(pipe, x_adv, t_star, fixed_timesteps, text_emb, noise)
        purified_emb = facenet_embed(facenet, purified)

        sim = F.cosine_similarity(purified_emb, clean_emb).mean()  # minimize this
        sim.backward()

        with torch.no_grad():
            delta -= step_size * delta.grad.sign()
            delta.clamp_(-epsilon, epsilon)
        delta.grad = None

        history.append(sim.item())
        if step % 5 == 0 or step == steps - 1:
            print(f"    step {step:3d}: facenet cosine-sim(purified, clean) = {sim.item():.4f}")

    return delta.detach(), history


def optimize_delta_eot(pipe, facenet, clean_pixel, theta_min, theta_max, k_theta, fixed_timesteps, text_emb,
                        epsilon, steps, step_size, var_weight, seed=0):
    """v0.2: Sec 4's EOT-over-Theta objective. MEMORY-SAFE variant: each of
    the k sampled thetas gets its OWN forward+backward pass, freed
    immediately (PyTorch's default graph-free-after-backward), instead of
    stacking all k branches' graphs before a single backward -- the naive
    version OOM'd at k=3 (42.7GB, backprop-through-~20-UNet-steps per branch
    held x3 simultaneously). d(mean_sim)/d(delta) = (1/k) sum_i d(sim_i)/d(delta),
    so backpropping (sim_i/k_theta) per branch and letting PyTorch accumulate
    into delta.grad across the k calls is exactly equivalent to backpropping
    the mean in one shot -- just spread over k cheap, sequential graphs.

    KNOWN SIMPLIFICATION (flagged): this only backprops the MEAN term. The
    variance term's true gradient needs all k branches jointly (Var's
    gradient depends on every branch's value via the shared mean), which
    would need either holding all graphs at once (the OOM'ing approach) or a
    two-pass recompute -- deferred. var_sim below is a MONITORING statistic
    only (computed from detached values), not yet an optimized term."""
    delta = torch.zeros_like(clean_pixel, requires_grad=True)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)

    with torch.no_grad():
        clean_emb = facenet_embed(facenet, clean_pixel).detach()

    history = []
    for step in range(steps):
        thetas = torch.randint(theta_min, theta_max + 1, (k_theta,), device=DEVICE, generator=gen).tolist()
        sim_values = []
        for t_star in thetas:
            x_adv = (clean_pixel + delta).clamp(0, 1)
            noise = torch.randn(1, 4, 64, 64, generator=gen, device=DEVICE, dtype=torch.float16)
            purified = differentiable_diffpure(pipe, x_adv, t_star, fixed_timesteps, text_emb, noise)
            purified_emb = facenet_embed(facenet, purified)
            sim = F.cosine_similarity(purified_emb, clean_emb).mean()
            (sim / k_theta).backward()
            sim_values.append(sim.item())

        mean_sim = float(np.mean(sim_values))
        var_sim = float(np.var(sim_values))  # monitoring only, see docstring

        with torch.no_grad():
            delta -= step_size * delta.grad.sign()
            delta.clamp_(-epsilon, epsilon)
        delta.grad = None

        history.append({"mean_sim": mean_sim, "var_sim": var_sim, "thetas": thetas})
        if step % 5 == 0 or step == steps - 1:
            print(f"    step {step:3d}: mean_sim={mean_sim:.4f}  var={var_sim:.5f} (monitor-only)  "
                  f"thetas={thetas}")

    return delta.detach(), history


def optimize_delta_eot_full(pipe, facenet, clean_pixel, theta_min, theta_max, k_theta, fixed_timesteps, text_emb,
                             epsilon, steps, step_size, var_weight, seed=0):
    """v0.3: the FULL Sec 4 objective mean_theta[sim] + var_weight*var_theta[sim],
    with the variance term's TRUE gradient (not just monitored, as in v0.2).

    Derivation: Var = (1/k) sum_i sim_i^2 - mean^2, so
        d(Var)/d(delta) = (2/k) sum_i (sim_i - mean) * d(sim_i)/d(delta)
    Combined with d(mean)/d(delta) = (1/k) sum_i d(sim_i)/d(delta), the total
    objective's gradient is a per-branch WEIGHTED sum:
        d(mean + var_weight*Var)/d(delta) = (1/k) sum_i w_i * d(sim_i)/d(delta),
        w_i = 1 + 2*var_weight*(sim_i - mean)
    Two-pass-per-step trick keeps this memory-safe (one branch's graph alive
    at a time, same footprint as v0.2's mean-only version):
      pass 1 (no_grad): recompute the k branches to get raw sim_i values and
        thus mean/weights (cheap, no graph).
      pass 2 (grad): recompute the SAME k branches (same noise seeds) with
        grad enabled, each immediately backpropping (w_i * sim_i) and freeing
        its own graph before the next branch -- exact gradient, ~2x forward
        compute per step vs v0.2, same peak memory.
    """
    delta = torch.zeros_like(clean_pixel, requires_grad=True)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)

    with torch.no_grad():
        clean_emb = facenet_embed(facenet, clean_pixel).detach()

    history = []
    for step in range(steps):
        thetas = torch.randint(theta_min, theta_max + 1, (k_theta,), device=DEVICE, generator=gen).tolist()
        branch_seeds = [seed * 100_000 + step * 1000 + i for i in range(k_theta)]

        # --- pass 1: no-grad, get raw sim values to compute mean/weights ---
        sim_values = []
        with torch.no_grad():
            x_adv = (clean_pixel + delta).clamp(0, 1)
            for t_star, bseed in zip(thetas, branch_seeds):
                g = torch.Generator(device=DEVICE).manual_seed(bseed)
                noise = torch.randn(1, 4, 64, 64, generator=g, device=DEVICE, dtype=torch.float16)
                purified = differentiable_diffpure(pipe, x_adv, t_star, fixed_timesteps, text_emb, noise)
                emb = facenet_embed(facenet, purified)
                sim_values.append(F.cosine_similarity(emb, clean_emb).mean().item())

        mean_sim = float(np.mean(sim_values))
        var_sim = float(np.var(sim_values))
        weights = [(1.0 + 2.0 * var_weight * (s - mean_sim)) / k_theta for s in sim_values]

        # --- pass 2: with-grad, weighted backward per branch, freed immediately ---
        # x_adv MUST be recomputed fresh per branch: it depends on delta (a leaf with
        # requires_grad=True), so if computed once and reused across branches, the first
        # branch's .backward() frees x_adv's own tiny graph too, breaking the second branch's
        # backward ("Trying to backward through the graph a second time").
        for t_star, bseed, w in zip(thetas, branch_seeds, weights):
            x_adv = (clean_pixel + delta).clamp(0, 1)
            g = torch.Generator(device=DEVICE).manual_seed(bseed)
            noise = torch.randn(1, 4, 64, 64, generator=g, device=DEVICE, dtype=torch.float16)
            purified = differentiable_diffpure(pipe, x_adv, t_star, fixed_timesteps, text_emb, noise)
            emb = facenet_embed(facenet, purified)
            sim = F.cosine_similarity(emb, clean_emb).mean()
            (w * sim).backward()

        with torch.no_grad():
            delta -= step_size * delta.grad.sign()
            delta.clamp_(-epsilon, epsilon)
        delta.grad = None

        history.append({"mean_sim": mean_sim, "var_sim": var_sim, "thetas": thetas})
        if step % 5 == 0 or step == steps - 1:
            print(f"    step {step:3d}: mean_sim={mean_sim:.4f}  var={var_sim:.5f} (OPTIMIZED)  "
                  f"thetas={thetas}")

    return delta.detach(), history


def evaluate_arcface(pipe, face_app, clean_pixel_np_img, adv_pixel, t_star, fixed_timesteps, text_emb, n_seeds=8):
    """Real (ArcFace, non-differentiable) evaluation: does the immunized image
    actually reduce post-purification identity similarity vs. the unprotected
    baseline, averaged over purification noise seeds?"""
    def to_pil(t):
        arr = (t[0].permute(1, 2, 0).detach().cpu().numpy() * 255).round().astype(np.uint8)
        return Image.fromarray(arr)

    emb_clean = arcface_embed(face_app, clean_pixel_np_img)

    @torch.no_grad()
    def sweep(pixel_tensor):
        sims = []
        for seed in range(n_seeds):
            gen = torch.Generator(device=DEVICE).manual_seed(1000 + seed)
            noise = torch.randn(1, 4, 64, 64, generator=gen, device=DEVICE, dtype=torch.float16)
            purified = differentiable_diffpure(pipe, pixel_tensor, t_star, fixed_timesteps, text_emb, noise)
            emb = arcface_embed(face_app, to_pil(purified))
            sims.append(cosine_sim(emb_clean, emb) if emb is not None else 0.0)
        return sims

    return sweep


def evaluate_grid(pipe, face_app, clean_img, pixel_tensor, t_grid, fixed_timesteps, text_emb, n_seeds):
    """Full-Theta-grid evaluation: mean ArcFace similarity at EACH t* in
    t_grid, not just the single point a delta happened to be optimized at.
    This is the actual test of whether EOT buys robustness across Theta."""
    results = {}
    for t_star in t_grid:
        sweep_fn = evaluate_arcface(pipe, face_app, clean_img, pixel_tensor, t_star, fixed_timesteps,
                                     text_emb, n_seeds)
        sims = sweep_fn(pixel_tensor)
        results[t_star] = float(np.mean(sims))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--identity", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "celebahq_sample", "identity_00.png"))
    ap.add_argument("--t-star", type=int, default=200, help="v0.1 single-theta baseline")
    ap.add_argument("--theta-min", type=int, default=100)
    ap.add_argument("--theta-max", type=int, default=400)
    ap.add_argument("--k-theta", type=int, default=3, help="EOT samples per step")
    ap.add_argument("--var-weight", type=float, default=0.5)
    ap.add_argument("--epsilon", type=float, default=8 / 255)
    ap.add_argument("--steps-single", type=int, default=20)
    ap.add_argument("--steps-eot", type=int, default=30)
    ap.add_argument("--step-size", type=float, default=1 / 255)
    ap.add_argument("--eval-grid", type=int, nargs="+", default=[100, 150, 200, 250, 300, 350, 400])
    ap.add_argument("--eval-seeds", type=int, default=5)
    ap.add_argument("--out-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "results"))
    args = ap.parse_args()

    pipe = load_sd_pipeline()
    face_app = load_face_app()
    facenet = load_facenet()
    fixed_timesteps = get_fixed_timesteps(pipe, 50)

    text_input = pipe.tokenizer([""], padding="max_length", max_length=pipe.tokenizer.model_max_length,
                                 truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        text_emb = pipe.text_encoder(text_input.input_ids)[0]

    clean_img = Image.open(args.identity).convert("RGB")
    clean_pixel = pil_to_pixel_tensor(clean_img)

    print(f"=== v0.1: single-theta delta (t*={args.t_star}) ===")
    delta_single, _ = optimize_delta(pipe, facenet, clean_pixel, args.t_star, fixed_timesteps,
                                      text_emb, args.epsilon, args.steps_single, args.step_size)
    adv_single = (clean_pixel + delta_single).clamp(0, 1)
    torch.cuda.empty_cache()

    print(f"\n=== v0.3: EOT-over-Theta delta, FULL objective (mean + var_weight*Var, real gradient) "
          f"(Theta=[{args.theta_min},{args.theta_max}], k={args.k_theta}, var_weight={args.var_weight}) ===")
    delta_full, _ = optimize_delta_eot_full(pipe, facenet, clean_pixel, args.theta_min, args.theta_max,
                                             args.k_theta, fixed_timesteps, text_emb, args.epsilon,
                                             args.steps_eot, args.step_size, args.var_weight)
    adv_full = (clean_pixel + delta_full).clamp(0, 1)
    torch.cuda.empty_cache()

    print(f"\n=== Evaluating across the FULL Theta grid {args.eval_grid} "
          f"(n_seeds={args.eval_seeds} per point) ===")
    print("  unprotected...")
    grid_unprotected = evaluate_grid(pipe, face_app, clean_img, clean_pixel, args.eval_grid,
                                      fixed_timesteps, text_emb, args.eval_seeds)
    print("  v0.1 single-theta delta...")
    grid_single = evaluate_grid(pipe, face_app, clean_img, adv_single, args.eval_grid,
                                 fixed_timesteps, text_emb, args.eval_seeds)
    print("  v0.3 EOT delta (full mean+var objective)...")
    grid_full = evaluate_grid(pipe, face_app, clean_img, adv_full, args.eval_grid,
                               fixed_timesteps, text_emb, args.eval_seeds)

    # v0.2 (mean-only EOT) numbers from the prior run, same identity/grid/seeds -- reference for the writeup,
    # not recomputed here to save compute (already validated: docs/certification_framework.md "v0.2")
    grid_eot_meanonly_prior = {100: 0.3461, 150: 0.2823, 200: 0.2544, 250: 0.2143,
                                300: 0.1909, 350: 0.1774, 400: 0.1246}

    print(f"\n{'t*':>6} {'unprotected':>12} {'single-theta':>13} {'EOT-mean(v0.2)':>15} {'EOT-full(v0.3)':>15}")
    for t in args.eval_grid:
        prior = grid_eot_meanonly_prior.get(t)
        prior_str = f"{prior:.4f}" if prior is not None else "n/a"
        print(f"{t:>6} {grid_unprotected[t]:>12.4f} {grid_single[t]:>13.4f} {prior_str:>15} {grid_full[t]:>15.4f}")

    worst_unprotected = max(grid_unprotected.values())
    worst_single = max(grid_single.values())
    worst_full = max(grid_full.values())
    worst_meanonly = max(grid_eot_meanonly_prior.values())
    print(f"\nworst-case (max) over Theta: unprotected={worst_unprotected:.4f}  single-theta={worst_single:.4f}  "
          f"EOT-mean(v0.2)={worst_meanonly:.4f}  EOT-full(v0.3)={worst_full:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    for name, adv in [("single", adv_single), ("eot_full", adv_full)]:
        img = Image.fromarray((adv[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))
        img.save(os.path.join(args.out_dir, f"delta_immunized_{name}.png"))
    import json
    with open(os.path.join(args.out_dir, "delta_eot_full_comparison.json"), "w") as f:
        json.dump({"unprotected": grid_unprotected, "single_theta": grid_single,
                   "eot_mean_only_v0_2_prior": grid_eot_meanonly_prior, "eot_full_v0_3": grid_full,
                   "worst_case": {"unprotected": worst_unprotected, "single_theta": worst_single,
                                  "eot_mean_only": worst_meanonly, "eot_full": worst_full}},
                  f, indent=2)
    print(f"\nSaved images + comparison JSON to {args.out_dir}")


if __name__ == "__main__":
    main()
