import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os

RESULTS = r"G:\CertImmune\results"
PAPER = r"G:\CertImmune\paper"
os.makedirs(os.path.join(PAPER, "figures"), exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ---------- Figure 1: imperceptibility example ----------
clean = mpimg.imread(os.path.join(RESULTS, "identity_00_original_check.png"))
immunized = mpimg.imread(os.path.join(RESULTS, "delta_immunized_eot.png"))

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6))
axes[0].imshow(clean)
axes[0].set_title("Clean", fontsize=12)
axes[0].axis("off")
axes[1].imshow(immunized)
axes[1].set_title(r"Immunized ($\epsilon{=}8/255$, EOT-$\delta$)", fontsize=12)
axes[1].axis("off")
plt.tight_layout()
fig.savefig(os.path.join(PAPER, "figures", "fig1_imperceptibility.pdf"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(PAPER, "figures", "fig1_imperceptibility.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("Figure 1 saved.")

# ---------- Figure 2: certified alpha comparison ----------
def load(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)

# DiffPure alpha_final (marginal), from certificate.json-style files: cert["immunized"]["certificate"]["alpha_final"]
def diffpure_alpha(name):
    d = load(name)
    return d["immunized"]["certificate"]["alpha_final"]

def gridpure_alpha(name):
    d = load(name)
    return d["immunized"]["marginal"]["alpha_final"]

methods = ["Unprotected", "Glaze\n(analog)", "Anti-DreamBooth\n(faithful)", "AntiPure\n(analog)",
           "PhotoGuard\n(faithful, 1000 steps)", "Ours\n(compute-matched,\n17 steps)", "Ours\n(original,\n30 steps)"]

diffpure_vals = [
    load("certificate.json")["unprotected"]["certificate"]["alpha_final"],
    diffpure_alpha("certificate_glaze.json"),
    diffpure_alpha("certificate_antidreambooth.json"),
    diffpure_alpha("certificate_antipure.json"),
    diffpure_alpha("certificate_photoguard.json"),
    load("certificate_computematched.json")["immunized"]["certificate"]["alpha_final"],
    diffpure_alpha("certificate.json"),
]

gridpure_vals = [
    load("certificate_gridpure.json")["unprotected"]["marginal"]["alpha_final"],
    gridpure_alpha("certificate_gridpure_glaze.json"),
    gridpure_alpha("certificate_gridpure_antidreambooth.json"),
    gridpure_alpha("certificate_gridpure_antipure.json"),
    gridpure_alpha("certificate_gridpure_photoguard.json"),
    np.nan,  # not run for the compute-matched variant
    gridpure_alpha("certificate_gridpure.json"),
]

x = np.arange(len(methods))
width = 0.35

fig, ax = plt.subplots(figsize=(9.8, 4.6))
b1 = ax.bar(x - width/2, diffpure_vals, width, label="DiffPure $\\alpha$", color="#4C72B0")
gv = [0 if (isinstance(v, float) and np.isnan(v)) else v for v in gridpure_vals]
b2 = ax.bar(x + width/2, gv, width, label="GrIDPure $\\alpha$", color="#DD8452")

# mark the "not run" bar (index of the wall-clock-matched "compute-matched" entry)
notrun_idx = methods.index("Ours\n(compute-matched,\n17 steps)")
ax.text(x[notrun_idx] + width/2, 0.02, "not run", ha="center", va="bottom", fontsize=8, style="italic", color="gray")

ax.axhline(1.0, color="black", linewidth=1, linestyle="--", alpha=0.7)
ax.text(len(methods) - 0.6, 1.02, "vacuous ceiling ($\\alpha=1$)", fontsize=8, va="bottom", ha="right", color="dimgray")

ax.set_ylabel(r"Certified $\alpha$ (marginal, lower = tighter guarantee)")
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=9)
ax.set_ylim(0, 1.4)
ax.legend(loc="upper right", frameon=False)

for i, v in enumerate(diffpure_vals):
    ax.text(x[i] - width/2, v + 0.02, f"{v:.3f}", ha="center", fontsize=7.5)
for i, v in enumerate(gridpure_vals):
    if not (isinstance(v, float) and np.isnan(v)):
        ax.text(x[i] + width/2, v + 0.02, f"{v:.3f}", ha="center", fontsize=7.5)

plt.tight_layout()
fig.savefig(os.path.join(PAPER, "figures", "fig2_certified_comparison.pdf"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(PAPER, "figures", "fig2_certified_comparison.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("Figure 2 saved.")
print("diffpure_vals:", diffpure_vals)
print("gridpure_vals:", gridpure_vals)
