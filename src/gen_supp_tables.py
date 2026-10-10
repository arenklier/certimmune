"""Emit the supplement's tables as LaTeX rows, straight from the stored JSON.

Transcribing three tables of counts by hand is exactly how a number ends up
disagreeing with the file it came from, so nothing here is typed twice.
"""
import glob
import json
import os
from math import comb

import numpy as np
from scipy.stats import beta as beta_dist

R = "/mnt/data/certimmune/results"
BS = chr(92)
NL = chr(10)


def cp(k, n, a=0.05):
    return 1.0 if k == n else float(beta_dist.ppf(1 - a, k + 1, n - k))


def row(*cells):
    return " & ".join(str(c) for c in cells) + " " + BS + BS


# ---------------------------------------------------------------- S1 PixArt cohort
print("%%% TABLE S1: PixArt cohort")
for f in sorted(glob.glob(os.path.join(R, "certificate_pixart_identity_*.json"))):
    d = json.load(open(f))
    g = d["t_grid"]
    u, i = d["unprotected"]["results"], d["immunized"]["results"]
    umax = max(u[str(t)]["phat"] for t in g)
    print(row(d["tag"].replace("_", BS + "_"),
              " ".join(str(u[str(t)]["k"]) for t in g),
              " ".join(str(i[str(t)]["k"]) for t in g),
              "%.3f" % d["immunized"]["marginal"]["alpha_final"],
              "yes" if umax >= 0.5 else "\\textbf{no}"))

# ---------------------------------------------------------------- S2 sub-grid control
print()
print("%%% TABLE S2: attacker-useful sub-grid")
ctrl = json.load(open(os.path.join(R, "certificate_pixart_ours.json")))
full = ctrl["t_grid"]
useful = [t for t in full if ctrl["unprotected"]["results"][str(t)]["phat"] >= 0.5]
print("%% useful sub-grid = %s" % useful)
import sys
sys.path.insert(0, "/mnt/data/certimmune/src")
from certify import certify


def sub(res, grid, sim):
    return certify({t: res[str(t)] for t in grid}, grid, 0.05, simultaneous=sim)["alpha_final"]


arms = [("unprotected", ctrl["unprotected"]["results"])]
for a in ["ours", "photoguard", "glaze", "antidreambooth", "antipure", "ours_wcmatch17"]:
    arms.append((a, json.load(open(os.path.join(R, "certificate_pixart_%s.json" % a)))["immunized"]["results"]))
for name, res in arms:
    print(row(name.replace("_", BS + "_"),
              "%.3f / %.3f" % (sub(res, full, False), sub(res, full, True)),
              "%.3f / %.3f" % (sub(res, useful, False), sub(res, useful, True))))

# ---------------------------------------------------------------- S3 replication
print()
print("%%% TABLE S3: replication of the calibration term")
IDS = [3, 17, 38, 9, 8, 37, 2, 15, 19, 4, 13, 14, 18, 21, 23, 28, 31]


def path(i, arm, seed):
    stem = "lora_calib_identity_%d_%s_r4" % (i, arm)
    return os.path.join(R, stem + (".json" if seed == 0 else "_s%d.json" % seed))


ks = {}
for s in (0, 1, 2):
    bad = []
    for i in IDS:
        for arm in ("unprotected", "protected"):
            d = json.load(open(path(i, arm, s)))
            c = d.get("results", d)["diffpure_t200"]
            rate = float(np.mean(np.asarray(c["real_sims"]) >= 0.2))
            if (c["proxy_sim"] < 0.5) != (rate == 0.0):
                bad.append("id%d/%s(%.0f%%)" % (i, arm[:4], 100 * rate))
    ks[s] = bad
    print(row("seed %d" % s, len(bad), "%.3f" % cp(len(bad), 34), ", ".join(bad) or "---"))

print()
print("%%% per-row match rates for the rows whose verdict is not unanimous")
for i, arm in ((3, "unprotected"), (4, "protected"), (13, "protected")):
    rates = []
    for s in (0, 1, 2):
        d = json.load(open(path(i, arm, s)))
        c = d.get("results", d)["diffpure_t200"]
        rates.append("%.0f\\%%" % (100 * float(np.mean(np.asarray(c["real_sims"]) >= 0.2))))
    print(row("identity%s%d / %s" % (BS + "_", i, arm), " / ".join(rates)))

# ---------------------------------------------------------------- S4 low-t full chain
print()
print("%%% TABLE S4: real attack success below Theta")
for arm in ("unprotected", "protected"):
    d = json.load(open(os.path.join(R, "lora_calib_identity_19_%s_lowt.json" % arm)))
    r = d.get("results", d)
    cells = []
    for c in ("clean", "diffpure_t50", "diffpure_t100", "diffpure_t200"):
        cells.append("%.3f" % r[c]["real_mean"])
    print(row(arm, *cells))

# ---------------------------------------------------------------- S5 dense sweeps
print()
print("%%% dense sweeps")
d1 = json.load(open(os.path.join(R, "dense_sweep.json")))
print("%% coarse (n=%d): grid %s" % (d1["n_seeds"], d1["t_grid"][0]) )
for lab, res in d1["arms"].items():
    print("%% %-16s %s" % (lab, " ".join(str(res[str(t)]["k"]) for t in d1["t_grid"])))
d2 = json.load(open(os.path.join(R, "dense_sweep2.json")))
for lab, v in d2["arms"].items():
    print("%% %-16s t*=%d..%d  %s" % (lab, v["grid"][0], v["grid"][-1],
                                      " ".join(str(v["results"][str(t)]["k"]) for t in v["grid"])))
