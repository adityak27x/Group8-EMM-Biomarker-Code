#!/usr/bin/env python3
"""
Figure for the randomisation test (Duivesteijn & Knobbe, ICDM 2011 style):
distribution of the best quality phi over ~1000 randomised datasets with the observed best phi.
Overlays the PLAIN null (all biomarker rows shuffled) and the STRATIFIED null (shuffled within diagnosis x site).

Inputs : <stratified>/<experiment>/permutation_null.npy + summary.json   (emm_heavy.py default run)
         <plain>/<experiment>/permutation_null.npy                       (emm_heavy.py ... --perm-strata none)
Usage  : python make_null_figure.py --stratified heavy_noCA199 --plain heavy_plainnull --out fig_null_distributions.png
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ap = argparse.ArgumentParser()
ap.add_argument("--stratified", default="heavy_noCA199")
ap.add_argument("--plain", default="heavy_plainnull")
ap.add_argument("--experiments", nargs="+", default=["E1_clinical", "E4_PDAC_only", "E5_PDAC_with_origin", "E7_benign_only"])
ap.add_argument("--out", default="fig_null_distributions.svg")
a = ap.parse_args()
NAMES = {"E1_clinical": "E1 all patients", "E4_PDAC_only": "E2 PDAC only", "E5_PDAC_with_origin": "E3 PDAC + site",
         "E6_controls_only": "E4 controls only", "E7_benign_only": "E5 benign only"}  # paper labels
GREY, BLUE, RED = "#8E8E8E", "#1976D2", "#C62828"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
n = len(a.experiments)
fig, axs = plt.subplots(1, n, figsize=(3.0 * n + 0.4, 3.5))
axs = np.atleast_1d(axs)
for k, (ax, e) in enumerate(zip(axs, a.experiments)):
    sp, pp = f"{a.stratified}/{e}", f"{a.plain}/{e}"
    obs = json.load(open(f"{sp}/summary.json"))["best_phi"] if os.path.exists(f"{sp}/summary.json") else np.nan
    nulls = {}
    for key, folder in (("plain", pp), ("strat", sp)):
        f = f"{folder}/permutation_null.npy"
        if os.path.exists(f):
            nulls[key] = np.load(f)
        else:
            print("missing", f)
    allv = np.concatenate([v for v in nulls.values()] + [[obs]])
    bins = np.linspace(allv.min() - 0.15, allv.max() + 0.15, 32)
    sub = []
    for key, col in (("plain", GREY), ("strat", BLUE)):
        if key in nulls:
            ax.hist(nulls[key], bins=bins, color=col, alpha=.55, linewidth=0)
            p95 = np.percentile(nulls[key], 95)
            ax.axvline(p95, color=col, ls="--", lw=1.6)
            sub.append(f"{'plain' if key == 'plain' else 'strat.'} {p95:.2f}")
    ax.axvline(obs, color=RED, lw=2.2)
    ax.set_title(f"{chr(65 + k)}  {NAMES.get(e, e)}", fontsize=10, fontweight="bold", pad=15)
    ax.text(0.5, 1.02, "95th pct: " + " · ".join(sub) + f" | obs {obs:.2f}", transform=ax.transAxes, ha="center", va="bottom", fontsize=7)
    ax.set_xlabel("best quality φ per randomised dataset")
    if k == 0:
        ax.set_ylabel("randomised datasets")
h = [Patch(facecolor=GREY, alpha=.55, label="plain null (all rows shuffled)"),
     Patch(facecolor=BLUE, alpha=.55, label="stratified null (within diagnosis × site)"),
     Line2D([0], [0], color="k", ls="--", lw=1.6, label="null 95th percentile"),
     Line2D([0], [0], color=RED, lw=2.2, label="observed best φ")]
fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(a.out, dpi=300, bbox_inches="tight")
print("saved", a.out)
