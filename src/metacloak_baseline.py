"""MetaCloak baseline (Liu et al., CVPR 2024) for the certified comparison.

Run through the OFFICIAL implementation (third_party/MetaCloak, unmodified
source), not a port and not an analog: the repo is public, so the standard we
held PhotoGuard and Anti-DreamBooth to applies here too. This driver only
prepares the inputs the official script expects, invokes it, and checks the
artifact it produces.

Deviations from the released default config, all deliberate:
  - r = 8 rather than 11, so MetaCloak carries the SAME 8/255 budget every
    other method in this comparison carries. step_size = r/10 = 0.8 follows
    their own rule for deriving it.
  - generation backbone is SD1.5 rather than their default SD2.1-base. Every
    other perturbation here is optimized against SD1.5, which is also the
    purifiers' backbone; leaving MetaCloak on SD2.1 would confound its method
    with a backbone mismatch no other baseline carries, and the released
    script supports SD15 explicitly.
  - the instance set is a single image (set_A = set_B = set_C = the one
    CelebA-HQ photograph), the same single-image degenerate case this paper
    already runs Anti-DreamBooth in. MetaCloak is designed for multi-image
    sets; this is a scope limit to state, not a silent choice.

Everything else is the released configuration: total_trail_num=4,
interval=200, advance_steps=2, unroll_steps=1, total_train_steps=1000,
defense_pgd_step_num=6, defense_sample_num=1, attack_pgd_radius=0,
attack_pgd_step_num=3, transform_gau + gau_kernel_size=7 + transform_hflip
(their 'gau' robust training mode, which is the configuration their own
scripts default to), lr=5e-7, prior preservation with 200 class images,
resolution 512, center crop, train_text_encoder, fp16.

Both this driver and the official load_data() bring the 1024x1024 CelebA-HQ
source down to 512x512 with PIL's default resize, which is also what
delta_optimize.pil_to_pixel_tensor does for every other method, so the clean
reference the budget is measured against is bit-identical across methods.
"""

import argparse
import os
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = "/mnt/data/certimmune"
MC_ROOT = os.path.join(ROOT, "third_party", "MetaCloak")
SD15 = ("/mnt/data/hf_cache/hub/models--runwayml--stable-diffusion-v1-5"
        "/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14")


def clean_512(path):
    """The 512x512 clean reference, produced exactly as every other method in
    this comparison produces it."""
    return np.array(Image.open(path).convert("RGB").resize((512, 512))).astype(np.int16)


def prepare_instance(identity_png, work_root, name):
    """MetaCloak expects dataset/<set>/<instance>/{set_A,set_B,set_C} plus a
    class.txt. With one photograph the three sets coincide."""
    inst = os.path.join(work_root, "dataset", "CelebA-HQ-cert", name)
    img = Image.open(identity_png).convert("RGB").resize((512, 512))
    for sub in ("set_A", "set_B", "set_C"):
        d = os.path.join(inst, sub)
        os.makedirs(d, exist_ok=True)
        img.save(os.path.join(d, f"{name}.png"))
    with open(os.path.join(inst, "class.txt"), "w") as f:
        f.write("person\n")
    return inst


def build_cmd(inst_dir, out_dir, class_dir, radius, seed):
    step_size = radius / 10.0  # their own rule: step_size = r/10
    return [
        sys.executable, "attacks/algs/metacloak.py",
        "--instance_name", os.path.basename(inst_dir),
        "--dataset_name", "CelebA-HQ-cert",
        "--total_train_steps", "1000",
        "--wandb_entity_name", "disabled",
        "--seed", str(seed),
        "--interval", "200",
        "--advance_steps", "2",
        "--unroll_steps", "1",
        "--total_trail_num", "4",
        "--exp_name", "certimmune",
        "--exp_hyper", f"r-{radius}-SD15",
        f"--pretrained_model_name_or_path={SD15}",
        f"--instance_data_dir_for_train={os.path.join(inst_dir, 'set_A')}",
        f"--instance_data_dir_for_adversarial={os.path.join(inst_dir, 'set_B')}",
        "--instance_prompt=a photo of sks person",
        f"--class_data_dir={class_dir}",
        "--num_class_images=200",
        "--class_prompt=a photo of person",
        f"--output_dir={out_dir}",
        "--center_crop",
        "--with_prior_preservation",
        "--prior_loss_weight=1.0",
        "--resolution=512",
        "--train_text_encoder",
        "--train_batch_size=1",
        "--checkpointing_iterations=10",
        "--learning_rate=5e-7",
        f"--defense_pgd_radius={radius}",
        f"--defense_pgd_step_size={step_size}",
        "--defense_pgd_step_num=6",
        "--defense_sample_num=1",
        "--defense_pgd_ascending",
        "--attack_pgd_radius=0",
        "--attack_pgd_step_size=1",
        "--attack_pgd_step_num=3",
        "--mixed_precision=fp16",
        # their 'gau' robust mode, which their own released scripts default to
        "--transform_gau",
        "--gau_kernel_size", "7",
        "--transform_hflip",
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--identity", required=True, help="path to the clean identity PNG")
    ap.add_argument("--radius", type=int, default=8, help="l_inf budget in 0-255 units")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--work-root", default=os.path.join(ROOT, "metacloak_work"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results"))
    ap.add_argument("--skip-if-exists", action="store_true")
    args = ap.parse_args()

    name = os.path.splitext(os.path.basename(args.identity))[0]
    final_png = os.path.join(args.out_dir, f"metacloak_baseline_immunized_{name}.png")
    if args.skip_if_exists and os.path.exists(final_png):
        print(f"SKIP {name}: {final_png} already present")
        return 0

    inst_dir = prepare_instance(args.identity, args.work_root, name)
    out_dir = os.path.join(args.work_root, "gen_output", name)
    # class images are identical across identities, so generate them once and share
    class_dir = os.path.join(args.work_root, "prior-data", "SD15", "class-person")
    os.makedirs(class_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = MC_ROOT
    env["WANDB_MODE"] = "disabled"
    env["HF_HOME"] = "/mnt/data/hf_cache"
    env["HF_HUB_OFFLINE"] = "1"

    cmd = build_cmd(inst_dir, out_dir, class_dir, args.radius, args.seed)
    print(f"=== MetaCloak (official impl) on {name}, r={args.radius}/255, SD1.5 ===")
    print(" ".join(cmd))
    rc = subprocess.call(cmd, cwd=os.path.join(MC_ROOT, "robust_facecloak"), env=env)
    if rc != 0:
        print(f"FAIL {name}: official script exited {rc}")
        return rc

    produced = os.path.join(out_dir, "noise-ckpt", "final")
    cands = [f for f in os.listdir(produced)] if os.path.isdir(produced) else []
    if not cands:
        print(f"FAIL {name}: no image under {produced}")
        return 2
    adv_path = os.path.join(produced, sorted(cands)[0])

    adv = np.array(Image.open(adv_path).convert("RGB")).astype(np.int16)
    ref = clean_512(args.identity)
    if adv.shape != ref.shape:
        print(f"FAIL {name}: shape {adv.shape} != clean {ref.shape}")
        return 3
    linf = int(np.abs(adv - ref).max())
    print(f"budget check: max per-channel |adv-clean| = {linf}/255 (allowed {args.radius})")
    if linf > args.radius:
        print(f"FAIL {name}: budget exceeded ({linf} > {args.radius})")
        return 4

    shutil.copyfile(adv_path, final_png)
    print(f"OK {name}: {final_png}  (linf={linf}/255)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
