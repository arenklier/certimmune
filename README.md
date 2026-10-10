# Certifying Image Immunization Against Diffusion-Based Personalization Under Adaptive-Strength Purification

Code and raw results accompanying the paper. This repository exists to let a
reader inspect the certification implementation and the numbers behind every
reported figure without waiting on a request; the development history follows
on acceptance (see **After acceptance** below).

## What is here

- `src/` -- all source code: the certification procedure (`certify.py`), the
  optimizers (`delta_optimize.py`, `delta_optimize_fast.py`), the baseline
  ports (`photoguard_baseline.py`, `antipure_baseline.py`,
  `metacloak_baseline.py`, ...), the purifiers
  (`gridpure.py`, `pixart_purifier.py`, ...), and the evaluation scripts
  (`imperceptibility.py`, `imperceptibility_paired.py`,
  `validate_surrogate_transfer.py`, ...).
- `results/` -- every certificate, delta, and log the paper's numbers come
  from: 989 JSON files (certificates and calibration data), 308 PNG files
  (immunized images and figure sources), 91 log files. This includes the
  MetaCloak baseline certificates, the 120-certificate $n=100$ three-seed
  fast-EOT/PhotoGuard comparison, the $n=60$ cohort and the upscaling
  round-trip stress test.
- `ARTIFACT_INDEX.md` -- maps each number reported in the paper to the exact
  file it comes from. Start here.

## What is not here, and why

- **The two public datasets** (CelebA-HQ; the identity set released with
  MetaCloak) are not redistributed here; they are third-party data under their
  own licenses. The paper's Ethics section names both and the code expects
  them under `data/` with the layout `certify.py --help` documents.
- **Third-party baseline code** (Anti-DreamBooth, IMPRESS's official
  repository) is not vendored here for the same reason. The paper describes
  exactly which commit/release of each was used; `src/` contains this
  project's own ports and analogs, not copies of those repositories.
- **The Python environment** (PyTorch, diffusers, insightface, lpips,
  scikit-image, ...) is not bundled; nothing here is unusual for a
  diffusion + face-embedding pipeline.

## Reproducing a result

Every script in `src/` has its paths set for the machine this project ran on
(`/mnt/data/certimmune/...`, mostly as a `BASE`/`ROOT` constant near the top
of the file, occasionally inlined). Point those at your own layout and the
scripts run as they did here; nothing in the certification logic itself
depends on that layout. `certify.py --help` and `delta_optimize_fast.py --help`
document the arguments each script takes.

The certification math itself is short and self-contained: the Clopper-Pearson
bound, the grid-Lipschitz residual, and the discrete union bound are all in
`certify.py`, independent of the diffusion pipeline around them, and are the
first place to look to check the core claim rather than reproduce a multi-GPU-hour
sweep end to end.

## After acceptance

The release on acceptance adds the full development history: a research log
documenting every design decision, correction, and retracted claim along the
way (several are referenced directly in the paper's Scope and Limitations
section). That log is held back only until the paper is settled, not because
anything in it is withheld -- nothing here has been edited to look better than
the process that produced it.

## Licence

Source code in `src/` is MIT-licensed (`LICENSE`); the contents of `results/`
are CC BY 4.0 (`LICENSE-DATA`). The two public datasets the code expects under
`data/` are not redistributed here and remain under their own terms.
