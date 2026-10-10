"""Emit the job queue for the two reviewer-driven experiments.

p0  MetaCloak generation on identity_00, run alone. It also generates the 200
    shared prior-preservation class images, which twenty racing workers would
    corrupt, and it is the measurement that tells us what MetaCloak actually
    costs per identity before committing the other nineteen.
p1  MetaCloak generation on the remaining nineteen identities.
p2  n=100 re-certification of the fast-EOT/PhotoGuard paired comparison at
    optimization seed 0. Seed 0 is completed across the whole cohort before
    any second seed starts, so stopping at any point still leaves a complete,
    reportable unit.
p3  Certifying the MetaCloak perturbations: the twenty-identity DiffPure
    cohort at n=20 (the scale-up's own sample size), plus identity_00 on all
    three architectures at the sample sizes Table VI uses.
p4  The remaining two optimization seeds of the n=100 comparison.

The unprotected arm is a deterministic function of the clean image, the grid
and the seed range -- it does not depend on which method's perturbation is
being certified, nor on that perturbation's optimization seed. So it is
measured exactly once per identity, on the fast-EOT seed-0 job, and every
other job for that identity runs with --skip-unprotected. That is a little
under half the total draws across p2 and p4.

identity_00's n=100 certificates for both methods already exist from the
earlier round, so they are not re-queued.
"""

import os

ROOT = "/mnt/data/certimmune"
RES = f"{ROOT}/results"
DATA = f"{ROOT}/data/celebahq_sample"
PY = f"{ROOT}/venv/bin/python"
PY_MC = f"{ROOT}/venv_metacloak/bin/python"

COHORT = ["00", "01", "02", "04", "05", "06", "07", "08", "09", "10",
          "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"]

METHODS = {
    "fasteot80": ("delta_immunized_fasteot80_%s%s.png",
                  "certificate_fasteot80_%s%s_n100.json"),
    "photoguard": ("photoguard_baseline_%s%s_immunized.png",
                   "certificate_photoguard_%s%s_n100.json"),
}
SEED_SUFFIX = {0: "", 1: "_d1", 2: "_d2"}


def mc_gen(ident):
    return (f"mcgen_{ident}",
            f"{PY_MC} metacloak_baseline.py "
            f"--identity {DATA}/identity_{ident}.png --skip-if-exists")


def b100(method, ident, seed, measure_unprotected):
    img_tpl, cert_tpl = METHODS[method]
    sfx = SEED_SUFFIX[seed]
    skip = "" if measure_unprotected else " --skip-unprotected"
    return (f"b100_{method}_{ident}_s{seed}",
            f"{PY} certify.py --clean {DATA}/identity_{ident}.png "
            f"--immunized {RES}/{img_tpl % (ident, sfx)} --n-seeds 100 --bonferroni{skip} "
            f"--out {RES}/{cert_tpl % (ident, sfx)}")


def mc_cert_diffpure(ident, n):
    tag = "" if n == 20 else f"_n{n}"
    return (f"mccert_{ident}_n{n}",
            f"{PY} certify.py --clean {DATA}/identity_{ident}.png "
            f"--immunized {RES}/metacloak_baseline_immunized_identity_{ident}.png "
            f"--n-seeds {n} --out {RES}/certificate_metacloak_{ident}{tag}.json")


def main():
    phases = {}
    phases["p0"] = [mc_gen("00")]
    phases["p1"] = [mc_gen(i) for i in COHORT if i != "00"]

    # fast-EOT seed 0 is the one job per identity that measures the unprotected arm
    p2 = []
    for ident in COHORT:
        if ident == "00":
            continue
        p2.append(b100("fasteot80", ident, 0, measure_unprotected=True))
        p2.append(b100("photoguard", ident, 0, measure_unprotected=False))
    phases["p2"] = p2

    # Everything that needs only identity_00's artifact runs first, as p3a. It
    # is about an hour of work and it answers the question the whole addition
    # turns on -- whether MetaCloak certifies at all -- instead of letting the
    # other nineteen generations spend two days before we find out.
    p3a = [mc_cert_diffpure("00", 20), mc_cert_diffpure("00", 30)]
    for n in (20, 30):
        p3a.append((f"mcgrid_00_n{n}",
                    f"{PY} certify_gridpure.py --clean {DATA}/identity_00.png "
                    f"--immunized {RES}/metacloak_baseline_immunized_identity_00.png "
                    f"--n-seeds {n} --out {RES}/certificate_gridpure_metacloak_n{n}.json"))
    p3a.append(("mcpixart_00_n30",
                f"{PY} certify_pixart.py --n-seeds 30 "
                f"--arms metacloak=metacloak_baseline_immunized_identity_00.png"))
    phases["p3a"] = p3a

    phases["p3"] = [mc_cert_diffpure(i, 20) for i in COHORT if i != "00"]

    # MetaCloak in the multi-image setting it was designed for, on one identity
    # first. Single-image is the degenerate case, and if the four-image result
    # differs sharply then the nineteen single-image generations queued behind
    # it are measuring the wrong thing -- so this runs before them, not after.
    MULTI_ID = "identity_19"
    MCDATA = f"{ROOT}/data/metacloak_identities/{MULTI_ID}"
    SINGLE = f"{ROOT}/data/metacloak_single/{MULTI_ID}_img0.png"
    phases["p1m"] = [
        (f"mcmulti_{MULTI_ID}",
         f"{PY_MC} metacloak_multi.py --instance-dir {MCDATA} --skip-if-exists"),
        (f"mcmulti_cert_{MULTI_ID}_n20",
         f"{PY} certify.py --clean {MCDATA}/img_0.png "
         f"--immunized {RES}/metacloak_multi_immunized_{MULTI_ID}.png "
         f"--n-seeds 20 --out {RES}/certificate_metacloak_multi_{MULTI_ID}.json"),
        # Same identity, one image: isolates the image-count factor from the
        # identity factor, which the multi-image comparison alone confounds
        # (it changed both at once).
        (f"mcsingle_{MULTI_ID}",
         f"mkdir -p {ROOT}/data/metacloak_single && "
         f"cp -n {MCDATA}/img_0.png {SINGLE}; "
         f"{PY_MC} metacloak_baseline.py --identity {SINGLE} --skip-if-exists"),
        (f"mcsingle_cert_{MULTI_ID}_n20",
         f"{PY} certify.py --clean {SINGLE} "
         f"--immunized {RES}/metacloak_baseline_immunized_{MULTI_ID}_img0.png "
         f"--n-seeds 20 --out {RES}/certificate_metacloak_single_{MULTI_ID}.json"),
    ]

    phases["p4"] = [b100(m, i, s, measure_unprotected=False)
                    for s in (1, 2) for i in COHORT for m in METHODS]

    qdir = f"{ROOT}/queue"
    os.makedirs(qdir, exist_ok=True)
    total = 0
    for phase in sorted(phases):
        jobs = phases[phase]
        with open(os.path.join(qdir, f"jobs_{phase}.txt"), "w") as f:
            for jid, cmd in jobs:
                f.write(f"{jid}\t{cmd}\n")
        n_skip = sum(1 for _, c in jobs if "--skip-unprotected" in c)
        print(f"jobs_{phase}.txt: {len(jobs)} jobs ({n_skip} skip-unprotected)")
        total += len(jobs)
    print(f"total: {total} jobs")


if __name__ == "__main__":
    main()
