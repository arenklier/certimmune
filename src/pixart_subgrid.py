"""Confound control for the third architecture, computed from the saved sweeps --
no new GPU work.

The problem: PixArt is a HARSHER purifier than DiffPure over the same grid. Its
unprotected control arm collapses to k=0 from t*=300 upward, i.e. at the top of
Theta it is not recovering the person at all. Every method's certificate is
therefore flattered by grid points where the purifier has simply destroyed the
image, and a bare comparison of the two alpha columns would be reading that
destruction as protection.

The control: re-certify every arm on the ATTACKER-USEFUL sub-grid only, defined
BEFORE looking at any protected arm as the grid points where the UNPROTECTED
control still recovers the identity at least half the time (phat >= 0.5). A
strength outside that set is self-defeating for the attacker: the purified image
no longer matches the person, so fine-tuning on it cannot personalize to them.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from certify import certify

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
ARMS = ["ours", "photoguard", "glaze", "antidreambooth", "antipure", "ours_wcmatch17"]

ctrl = json.load(open(os.path.join(R, "certificate_pixart_ours.json")))
full_grid = ctrl["t_grid"]
useful = [t for t in full_grid if ctrl["unprotected"]["results"][str(t)]["phat"] >= 0.5]
print("full grid   : %s" % full_grid)
print("unprotected : %s" % " ".join("%.2f" % ctrl["unprotected"]["results"][str(t)]["phat"]
                                    for t in full_grid))
print("useful sub-grid (unprotected phat >= 0.5): %s" % useful)
print()


def sub(res, grid, sim):
    r = {t: res[str(t)] for t in grid}
    return certify(r, grid, 0.05, simultaneous=sim)["alpha_final"]


print("%-16s %-12s %18s %18s" % ("arm", "sweep", "alpha full-grid", "alpha useful-grid"))
rows = [("ours", "unprotected", ctrl["unprotected"]["results"])]
for a in ARMS:
    d = json.load(open(os.path.join(R, "certificate_pixart_%s.json" % a)))
    rows.append((a, "immunized", d["immunized"]["results"]))
for name, sweep, res in rows:
    print("%-16s %-12s  %6.3f m / %6.3f s  %6.3f m / %6.3f s" % (
        name, sweep, sub(res, full_grid, False), sub(res, full_grid, True),
        sub(res, useful, False), sub(res, useful, True)))
