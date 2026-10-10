"""MetaCloak in its native multi-image setting.

metacloak_baseline.py runs the official implementation on a single CelebA-HQ
photograph, which is the degenerate case this paper already runs
Anti-DreamBooth in. MetaCloak, though, meta-learns a perturbation across
DreamBooth surrogates trained on an *instance set*; with one image that
surrogate training is off-design, and a reviewer would rightly ask whether the
weak certificate is a property of the method or of how we ran it.

This driver answers that: same official code, same budget, same backbone, but
the four-image MetaCloak-dataset instance sets the method was built for. The
certificate is then computed on the set's first image, which is exactly how
the supplement already proxy-certifies the MetaCloak identities.

Everything else matches metacloak_baseline.py, including r=8 and SD1.5.
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
    return np.array(Image.open(path).convert("RGB").resize((512, 512))).astype(np.int16)


def prepare_instance(src_dir, work_root, name):
    """set_A/set_B/set_C all carry the full instance set: MetaCloak trains its
    surrogates on set_A and perturbs set_B, and with a four-image identity
    both are the identity's own four photographs."""
    imgs = sorted(f for f in os.listdir(src_dir)
                  if os.path.splitext(f)[1].lower() in (".png", ".jpg", ".jpeg"))
    if not imgs:
        raise SystemExit(f"no images in {src_dir}")
    inst = os.path.join(work_root, "dataset", "MetaCloak-multi", name)
    for sub in ("set_A", "set_B", "set_C"):
        d = os.path.join(inst, sub)
        os.makedirs(d, exist_ok=True)
        for fn in imgs:
            Image.open(os.path.join(src_dir, fn)).convert("RGB").resize((512, 512)).save(
                os.path.join(d, fn))
    with open(os.path.join(inst, "class.txt"), "w") as f:
        f.write("person\n")
    return inst, imgs


def build_cmd(inst_dir, out_dir, class_dir, radius, seed):
    step_size = radius / 10.0
    return [
        sys.executable, "attacks/algs/metacloak.py",
        "--instance_name", os.path.basename(inst_dir),
        "--dataset_name", "MetaCloak-multi",
        "--total_train_steps", "1000",
        "--wandb_entity_name", "disabled",
        "--seed", str(seed),
        "--interval", "200",
        "--advance_steps", "2",
        "--unroll_steps", "1",
        "--total_trail_num", "4",
        "--exp_name", "certimmune-multi",
        "--exp_hyper", f"r-{radius}-SD15-multi",
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
        "--transform_gau",
        "--gau_kernel_size", "7",
        "--transform_hflip",
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance-dir", required=True,
                    help="directory holding the identity's images (e.g. .../identity_0)")
    ap.add_argument("--radius", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--work-root", default=os.path.join(ROOT, "metacloak_work"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results"))
    ap.add_argument("--skip-if-exists", action="store_true")
    args = ap.parse_args()

    name = os.path.basename(os.path.normpath(args.instance_dir))
    final_png = os.path.join(args.out_dir, f"metacloak_multi_immunized_{name}.png")
    if args.skip_if_exists and os.path.exists(final_png):
        print(f"SKIP {name}: {final_png} already present")
        return 0

    inst_dir, imgs = prepare_instance(args.instance_dir, args.work_root, name)
    print(f"=== MetaCloak (official impl, MULTI-IMAGE) on {name}: "
          f"{len(imgs)} images, r={args.radius}/255, SD1.5 ===")

    out_dir = os.path.join(args.work_root, "gen_output_multi", name)
    class_dir = os.path.join(args.work_root, "prior-data", "SD15", "class-person")
    for d in (out_dir, class_dir, args.out_dir):
        os.makedirs(d, exist_ok=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = MC_ROOT
    env["WANDB_MODE"] = "disabled"
    env["HF_HOME"] = "/mnt/data/hf_cache"
    env["HF_HUB_OFFLINE"] = "1"

    cmd = build_cmd(inst_dir, out_dir, class_dir, args.radius, args.seed)
    print(" ".join(cmd))
    rc = subprocess.call(cmd, cwd=os.path.join(MC_ROOT, "robust_facecloak"), env=env)
    if rc != 0:
        print(f"FAIL {name}: official script exited {rc}")
        return rc

    produced = os.path.join(out_dir, "noise-ckpt", "final")
    if not os.path.isdir(produced):
        print(f"FAIL {name}: no output under {produced}")
        return 2
    out_imgs = sorted(os.listdir(produced))
    if not out_imgs:
        print(f"FAIL {name}: output directory empty")
        return 2

    worst = 0
    for fn in out_imgs:
        src_name = fn[len("noisy_"):] if fn.startswith("noisy_") else fn
        src = os.path.join(args.instance_dir, src_name)
        if not os.path.exists(src):
            print(f"  WARN: no clean source for {fn}")
            continue
        adv = np.array(Image.open(os.path.join(produced, fn)).convert("RGB")).astype(np.int16)
        ref = clean_512(src)
        if adv.shape != ref.shape:
            print(f"FAIL {name}: shape mismatch on {fn}")
            return 3
        linf = int(np.abs(adv - ref).max())
        worst = max(worst, linf)
        print(f"  budget check {fn}: linf={linf}/255")
    if worst > args.radius:
        print(f"FAIL {name}: budget exceeded ({worst} > {args.radius})")
        return 4

    # certify on the set's first image, as the supplement already does
    first = sorted(out_imgs)[0]
    shutil.copyfile(os.path.join(produced, first), final_png)
    for fn in out_imgs:
        shutil.copyfile(os.path.join(produced, fn),
                        os.path.join(args.out_dir, f"metacloak_multi_{name}_{fn}"))
    print(f"OK {name}: {final_png}  (worst linf={worst}/255, {len(out_imgs)} images saved)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
