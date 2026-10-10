"""How much does the 0.174 calibration term depend on choices nobody validated?

Sec. VII-G bounds the proxy-vs-real verdict-disagreement rate by applying
Clopper-Pearson to k=2 disagreements out of n=34 rows. A "disagreement" is
defined by the most fragile binarization available -- match rate > 0% versus
= 0% -- and one of the two disagreeing rows is decided by ONE generation out of
twenty. Meanwhile Sec. IX argues, about the unprotected rates, that "a binary
split over them is not a reportable quantity" precisely because they sit near
the threshold. This script asks whether that tension costs anything, using only
data already on disk:

  (a) sensitivity to the binarization: recount at match > 0%, >= 10%, >= 20%;
  (b) sensitivity to the twenty generations themselves: bootstrap the per-row
      generations (B=2000, fixed seed) and report the distribution of the term.

(b) measures only the within-run generation noise. It cannot see run-to-run
fine-tuning variance, which a separate four-repeat probe puts at sd 0.039 on
real_mean -- so the spread reported here is a LOWER bound on the term's true
instability, not an estimate of it.
"""
import json
import os

import numpy as np
from scipy.stats import beta as beta_dist

R = "/mnt/data/certimmune/results"
ORIGINAL = [3, 17, 38, 9, 8, 37, 2, 15, 19]
ADDED = [4, 13, 14, 18, 21, 23, 28, 31]
TAU, REAL_THR, ALPHA_MC, B = 0.5, 0.2, 0.05, 2000


def cp_upper(k, n, a=ALPHA_MC):
    return 1.0 if k == n else float(beta_dist.ppf(1 - a, k + 1, n - k))


rows = []
for group, ids in (("original", ORIGINAL), ("added", ADDED)):
    for i in ids:
        for arm in ("unprotected", "protected"):
            f = os.path.join(R, "lora_calib_identity_%d_%s_r4.json" % (i, arm))
            if not os.path.exists(f):
                print("MISSING", os.path.basename(f))
                continue
            d = json.load(open(f))
            c = d.get("results", d)["diffpure_t200"]
            rows.append({"id": i, "arm": arm, "group": group, "proxy": c["proxy_sim"],
                         "sims": np.asarray(c["real_sims"], dtype=float)})
print("rows reconstructed: %d (expected 34)" % len(rows))


def agrees(row, sims, cutoff):
    """proxy verdict: PROTECTED iff proxy < tau.  real verdict: ATTACKED iff match rate > cutoff.
    The two agree when the proxy calls a row protected exactly if the real chain finds no attack."""
    rate = float(np.mean(sims >= REAL_THR))
    return (row["proxy"] < TAU) == (rate <= cutoff), rate


print()
print("%-10s %6s %8s %8s   %s" % ("binarize", "k", "n", "CP upper", "disagreeing rows"))
for cutoff, label in ((0.0, "> 0%"), (0.10, ">= 10%"), (0.20, ">= 20%")):
    bad = [r for r in rows if not agrees(r, r["sims"], cutoff)[0]]
    print("%-10s %6d %8d %8.3f   %s" % (
        label, len(bad), len(rows), cp_upper(len(bad), len(rows)),
        ", ".join("id_%d/%s(%.0f%%)" % (b["id"], b["arm"][:4], 100 * agrees(b, b["sims"], cutoff)[1])
                  for b in bad) or "-"))

rng = np.random.default_rng(0)
terms = []
for _ in range(B):
    k = sum(1 for r in rows
            if not agrees(r, rng.choice(r["sims"], size=r["sims"].size, replace=True), 0.0)[0])
    terms.append(cp_upper(k, len(rows)))
terms = np.array(terms)
print()
print("bootstrap over the 20 generations per row (B=%d, seed 0), '> 0%%' binarization:" % B)
print("  term: median %.3f   2.5%%-97.5%% [%.3f, %.3f]   min %.3f max %.3f"
      % (np.median(terms), *np.percentile(terms, [2.5, 97.5]), terms.min(), terms.max()))
vals, cnts = np.unique(terms, return_counts=True)
print("  distinct values: %s" % ", ".join("%.3f (%.0f%%)" % (v, 100 * c / B) for v, c in zip(vals, cnts)))
