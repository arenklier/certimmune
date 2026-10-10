"""Does h(theta) actually decrease with purifier strength, everywhere we have measured it?

This decides whether the Lipschitz assumption A1 can be replaced by the strictly
weaker, theoretically motivated assumption that h is non-increasing in strength.
A single large violation kills the idea; a handful of +-1 wiggles at small n is
sampling noise and is reported as such.

Counts, over every stored sweep (DiffPure, GrIDPure and PixArt; protected and
unprotected arms alike):
  * adjacent pairs where k increases with theta, and by how much
  * whether the binomial evidence for an increase is significant at all, using
    a one-sided Fisher exact test on the 2x2 table (k_i, n-k_i ; k_{i+1}, n-k_{i+1})
"""
import glob
import json
import os
from math import comb

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def fisher_greater(a, b, c, d):
    """P(X >= a) for the hypergeometric null, i.e. one-sided evidence that the
    LATER grid point has a higher success rate than the earlier one."""
    n1, n2, t = a + b, c + d, a + c
    tot = 0.0
    denom = comb(n1 + n2, t)
    for x in range(a, min(n1, t) + 1):
        tot += comb(n1, x) * comb(n2, t - x)
    return tot / denom


def sweeps(path):
    d = json.load(open(path))
    grid = d.get("t_grid") or d.get("r_grid") or d.get("grid")
    for arm in ("unprotected", "immunized"):
        if arm in d and isinstance(d[arm], dict) and "results" in d[arm]:
            yield d.get("tag", os.path.basename(path)), arm, grid, d[arm]["results"]


files = sorted(glob.glob(os.path.join(R, "certificate*.json")))
pairs = viol = sig = 0
worst = []
skipped = []
for f in files:
    try:
        got = list(sweeps(f))
    except Exception as e:
        skipped.append((os.path.basename(f), str(e)[:40]))
        continue
    if not got:
        skipped.append((os.path.basename(f), "no sweep"))
    for tag, arm, grid, res in got:
        if not grid:
            continue
        ks = [res[str(t)] for t in grid]
        for i in range(len(ks) - 1):
            pairs += 1
            a, b = ks[i]["k"], ks[i + 1]["k"]
            if b > a:
                viol += 1
                p = fisher_greater(ks[i + 1]["k"], ks[i + 1]["n"] - ks[i + 1]["k"],
                                   ks[i]["k"], ks[i]["n"] - ks[i]["k"])
                if p < 0.05:
                    sig += 1
                worst.append((b - a, p, os.path.basename(f), arm, grid[i], grid[i + 1], a, b, ks[i]["n"]))

print("files scanned            : %d  (%d with no readable sweep)" % (len(files), len(skipped)))
print("adjacent pairs           : %d" % pairs)
print("non-monotone pairs (k up): %d  (%.1f%%)" % (viol, 100.0 * viol / max(pairs, 1)))
print("of those, significant    : %d at one-sided Fisher p<0.05" % sig)
print()
print("largest increases:")
for v in sorted(worst, reverse=True)[:12]:
    print("  +%d  p=%.3f  %-42s %-11s t*%d->%d  k %d->%d (n=%d)" % v)
