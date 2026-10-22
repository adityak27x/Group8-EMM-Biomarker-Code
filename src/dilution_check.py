#!/usr/bin/env python3
"""
Dilution / site hypothesis check (needs emm_corr_rcs.py and emm_site_stratified.py in the same folder).

Part A  Range-restriction check: per site x diagnosis cell, how spread is creatinine and how strong are the
        creatinine~biomarker correlations? (Correlations shrink when the spread of one variable is small.)
Part B  Do the site / diagnosis contrasts survive once creatinine is controlled?
        method 'raw'      Spearman between the biomarkers (3 biomarker-biomarker pairs only)
        method 'partial'  partial Spearman controlling for creatinine
        method 'ratio'    Spearman between biomarker/creatinine ratios
Usage: python dilution_check.py --data "Debernardi et al 2020 data.csv" --out results_dilution
"""
import argparse, itertools, os
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import emm_corr_rcs as E
from emm_site_stratified import model_rho, BIO

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", default="results_dilution")
ap.add_argument("--min-n", type=int, default=30)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
df = E.load_data(a.data)
df["log_cr"] = np.log10(df["creatinine"])
pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 300)

# ---------------- Part A ----------------
rows = []
for (site, dx), d in df.groupby(["sample_origin", "diagnosis"]):
    if len(d) < a.min_n:
        continue
    r = E.spearman_vec(d[E.TARGETS].values)  # first three pairs are creatinine~LYVE1/REG1B/TFF1
    rows.append(dict(site=site, diagnosis=E.DIAG_LABEL[dx], n=len(d), median_creatinine=d.creatinine.median(),
                     sd_log10_creat=d.log_cr.std(), iqr_log10_creat=d.log_cr.quantile(.75) - d.log_cr.quantile(.25),
                     rho_cr_LYVE1=r[0], rho_cr_REG1B=r[1], rho_cr_TFF1=r[2], mean_rho_creat=r[:3].mean()))
A = pd.DataFrame(rows)
A.to_csv(f"{a.out}/A_creatinine_spread_vs_coupling.csv", index=False)
print("\nPART A: creatinine spread and coupling per site x diagnosis (n >= %d)" % a.min_n)
print(A.round(3).to_string(index=False))
if len(A) >= 4:
    rho, p = stats.spearmanr(A.sd_log10_creat, A.mean_rho_creat)
    print(f"\nAcross {len(A)} cells: Spearman(SD of log10 creatinine, mean creatinine-biomarker rho) = {rho:.2f} (p = {p:.3f})")
    print("  (only a handful of cells, so descriptive; positive = narrower creatinine range goes with weaker coupling)")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(A.sd_log10_creat, A.mean_rho_creat)
    for _, r in A.iterrows():
        ax.annotate(f"{r.site}-{r.diagnosis}", (r.sd_log10_creat, r.mean_rho_creat), fontsize=7)
    ax.set_xlabel("SD of log10 creatinine"); ax.set_ylabel("mean ρ(creatinine, biomarkers)")
    fig.savefig(f"{a.out}/fig_range_restriction.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------------- Part B ----------------
def design(d, method):
    if method == "ratio":
        return np.column_stack([d[b] / d["creatinine"] for b in BIO]).astype(float)
    return d[E.TARGETS].values.astype(float)

def rho_of(d, method):
    r = model_rho(design(d, method), "partial" if method == "partial" else "raw")
    return r[3:] if method == "raw" else r   # raw: keep only biomarker~biomarker pairs

names = {"raw": [f"{BIO[i]}~{BIO[j]}" for i, j in itertools.combinations(range(3), 2)]}
names["partial"] = [n + "|creat" for n in names["raw"]]
names["ratio"] = [n.replace("~", "/cr~") + "/cr" for n in names["raw"]]
extra = {"raw": 0, "partial": 1, "ratio": 0}

def bplus(d): return d[d.sample_origin == "BPTB"]
comparisons = []
for code1, code2 in itertools.combinations((1, 2, 3), 2):   # diagnosis contrasts within BPTB
    comparisons.append((f"{E.DIAG_LABEL[code1]} vs {E.DIAG_LABEL[code2]} (BPTB only)",
                        bplus(df)[bplus(df).diagnosis == code1], bplus(df)[bplus(df).diagnosis == code2]))
for code in (2, 3):                                           # site contrasts within diagnosis
    d = df[df.diagnosis == code]
    comparisons.append((f"{E.DIAG_LABEL[code]}: BPTB vs non-BPTB", d[d.sample_origin == "BPTB"], d[d.sample_origin != "BPTB"]))
    comparisons.append((f"{E.DIAG_LABEL[code]}: BPTB vs LIV", d[d.sample_origin == "BPTB"], d[d.sample_origin == "LIV"]))
    comparisons.append((f"{E.DIAG_LABEL[code]}: Cohort1 vs Cohort2", d[d.patient_cohort == "Cohort1"], d[d.patient_cohort == "Cohort2"]))
rows = []
for method in ("raw", "partial", "ratio"):
    for label, g1, g2 in comparisons:
        if len(g1) < a.min_n or len(g2) < a.min_n:
            continue
        r1, r2 = rho_of(g1, method), rho_of(g2, method)
        for k, pn in enumerate(names[method]):
            se = np.sqrt(E.se_fisher(len(g1) - extra[method]) ** 2 + E.se_fisher(len(g2) - extra[method]) ** 2)
            z = (E.fisher(r1[k]) - E.fisher(r2[k])) / se
            rows.append(dict(method=method, comparison=label, n1=len(g1), n2=len(g2), pair=pn,
                             rho1=r1[k], rho2=r2[k], diff=r1[k] - r2[k], z=z, p=2 * stats.norm.sf(abs(z))))
B = pd.DataFrame(rows)
B["p_bonf_within_method"] = B.groupby("method").p.transform(lambda s: (s * len(s)).clip(upper=1))
B["sig_bonf"] = B.p_bonf_within_method < 0.05
B.to_csv(f"{a.out}/B_biomarker_pairs_with_creatinine_controlled.csv", index=False)
print("\nPART B: biomarker-biomarker pairs, creatinine controlled (diff = group1 - group2)")
print(B.round(3)[["method", "comparison", "n1", "n2", "pair", "rho1", "rho2", "diff", "p", "p_bonf_within_method"]].to_string(index=False))
print("\nSignificant after Bonferroni (within method):")
sig = B[B.sig_bonf]
print(sig[["method", "comparison", "pair", "diff", "p_bonf_within_method"]].round(3).to_string(index=False) if len(sig) else "  none")
print("\nNominal p<0.05 counts per method (of %d tests each):" % (len(B) // 3))
print(B.groupby("method").p.apply(lambda s: int((s < 0.05).sum())).to_string())
