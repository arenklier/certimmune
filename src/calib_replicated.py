"""The calibration term, recomputed from three independent replicates per row.

Sec. VII-G bounds the proxy-vs-real verdict-disagreement rate from k disagreeing
rows out of n=34, where each row's real verdict came from a SINGLE run. A
four-repeat probe put the run-to-run spread of real_mean at sd 0.039 at this
configuration, and one of the two disagreements was decided by a single
generation out of twenty -- so the verdict itself, not just the number behind
it, was inside the noise. Every row has now been run three times (seeds 0, 1, 2;
seed 0 being the run already reported).

Three readings, in increasing order of what they let you claim:
  per-seed   -- the term you would have reported had you run that seed. Its
                spread is the thing the single-run version could not show.
  majority   -- each row's verdict is the majority of its three, which is what
                replication is actually for.
  any-seed   -- a row counts as disagreeing if it ever disagrees. Deliberately
                pessimistic; the honest upper end.
"""
import json
import os

import numpy as np
from scipy.stats import beta as beta_dist

R = "/mnt/data/certimmune/results"
IDS = [3, 17, 38, 9, 8, 37, 2, 15, 19, 4, 13, 14, 18, 21, 23, 28, 31]
SEEDS = [0, 1, 2]
TAU, REAL_THR, ALPHA_MC = 0.5, 0.2, 0.05


def cp_upper(k, n):
    return 1.0 if k == n else float(beta_dist.ppf(1 - ALPHA_MC, k + 1, n - k))


def path(i, arm, seed):
    stem = "lora_calib_identity_%d_%s_r4" % (i, arm)
    return os.path.join(R, stem + (".json" if seed == 0 else "_s%d.json" % seed))


rows = []
for i in IDS:
    for arm in ("unprotected", "protected"):
        cell = {"id": i, "arm": arm, "rate": {}, "proxy": {}}
        for s in SEEDS:
            d = json.load(open(path(i, arm, s)))
            c = d.get("results", d)["diffpure_t200"]
            sims = np.asarray(c["real_sims"], dtype=float)
            cell["rate"][s] = float(np.mean(sims >= REAL_THR))
            cell["proxy"][s] = c["proxy_sim"]
        rows.append(cell)
print("rows: %d   replicates each: %d" % (len(rows), len(SEEDS)))

pdrift = max(max(r["proxy"].values()) - min(r["proxy"].values()) for r in rows)
print("largest proxy drift across replicates: %.5f  (the proxy does not depend on the run seed)" % pdrift)


def disagrees(r, s):
    """proxy says PROTECTED iff proxy < tau; real says ATTACKED iff any generation matches."""
    return (r["proxy"][s] < TAU) != (r["rate"][s] == 0.0)


print()
per_seed = []
for s in SEEDS:
    bad = [r for r in rows if disagrees(r, s)]
    per_seed.append(cp_upper(len(bad), len(rows)))
    print("seed %d : k=%d  term=%.3f   %s" % (
        s, len(bad), per_seed[-1],
        ", ".join("id_%d/%s(%.0f%%)" % (b["id"], b["arm"][:4], 100 * b["rate"][s]) for b in bad) or "-"))
print("        per-seed term range: %.3f - %.3f" % (min(per_seed), max(per_seed)))

maj = [r for r in rows if sum(disagrees(r, s) for s in SEEDS) >= 2]
anyd = [r for r in rows if any(disagrees(r, s) for s in SEEDS)]
print()
print("majority-of-3 : k=%d  term=%.3f   %s" % (
    len(maj), cp_upper(len(maj), len(rows)),
    ", ".join("id_%d/%s" % (b["id"], b["arm"][:4]) for b in maj) or "-"))
print("any-seed      : k=%d  term=%.3f   %s" % (
    len(anyd), cp_upper(len(anyd), len(rows)),
    ", ".join("id_%d/%s" % (b["id"], b["arm"][:4]) for b in anyd) or "-"))

print()
print("rows whose verdict is not unanimous across the three replicates:")
flips = [r for r in rows if 0 < sum(disagrees(r, s) for s in SEEDS) < len(SEEDS)]
for r in flips:
    print("  id_%-3d %-12s match rates %s   proxy %.3f" % (
        r["id"], r["arm"], ["%.0f%%" % (100 * r["rate"][s]) for s in SEEDS], r["proxy"][0]))
print("  (%d of %d rows)" % (len(flips), len(rows)))

print()
print("headline, folded onto the tightest proxy-level certificate (0.030):")
for name, term in (("single run, seed 0 (as reported)", per_seed[0]),
                   ("majority-of-3", cp_upper(len(maj), len(rows))),
                   ("any-seed (pessimistic)", cp_upper(len(anyd), len(rows)))):
    print("  %-34s alpha <= 0.030 + %.3f = %.3f" % (name, term, 0.030 + term))
