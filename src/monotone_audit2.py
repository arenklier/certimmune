"""The honest denominator for the monotonicity audit.

Most protected sweeps are k=0 at every point, so they are monotone trivially and
carry almost no evidence. What matters is the INFORMATIVE pairs -- those where at
least one endpoint is strictly inside (0, n), so the sweep could have gone either
way -- and, separately, how often the drop is large enough to be real.
"""
import glob
import json
import os
from math import comb

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def fisher_greater(a, b, c, d):
    n1, n2, t = a + b, c + d, a + c
    return sum(comb(n1, x) * comb(n2, t - x) for x in range(a, min(n1, t) + 1)) / comb(n1 + n2, t)


def arch_of(name):
    if "pixart" in name:
        return "DiffPure-DiT"
    if "gridpure" in name:
        return "GrIDPure"
    return "DiffPure-SD1.5"


stats = {}
for f in sorted(glob.glob(os.path.join(R, "certificate*.json"))):
    d = json.load(open(f))
    grid = d.get("t_grid") or d.get("r_grid")
    if not grid:
        continue
    a = arch_of(os.path.basename(f))
    for arm in ("unprotected", "immunized"):
        if arm not in d or "results" not in d.get(arm, {}):
            continue
        ks = [d[arm]["results"][str(t)] for t in grid]
        for i in range(len(ks) - 1):
            lo, hi, n = ks[i]["k"], ks[i + 1]["k"], ks[i]["n"]
            informative = (0 < lo < n) or (0 < hi < ks[i + 1]["n"])
            s = stats.setdefault((a, arm), [0, 0, 0, 0, 0])
            s[0] += 1
            if informative:
                s[1] += 1
                if hi > lo:
                    s[2] += 1
                    if fisher_greater(hi, ks[i + 1]["n"] - hi, lo, n - lo) < 0.05:
                        s[3] += 1
                elif hi < lo:
                    s[4] += 1

print("%-16s %-12s %7s %12s %10s %11s %10s" % (
    "architecture", "arm", "pairs", "informative", "k goes up", "significant", "k goes down"))
tot = [0, 0, 0, 0, 0]
for key in sorted(stats):
    s = stats[key]
    print("%-16s %-12s %7d %12d %10d %11d %10d" % (key[0], key[1], s[0], s[1], s[2], s[3], s[4]))
    tot = [t + x for t, x in zip(tot, s)]
print("%-16s %-12s %7d %12d %10d %11d %10d" % ("ALL", "", *tot))
