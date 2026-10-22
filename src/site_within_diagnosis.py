#!/usr/bin/env python3
"""Does collection site / cohort change biomarker correlations WITHIN one diagnosis?
Compares Spearman correlations (Fisher-z test) for site and cohort contrasts inside benign and PDAC patients.
Needs emm_corr_rcs.py in the same folder.
Usage: python site_within_diagnosis.py --data "Debernardi et al 2020 data.csv" --out results_baseline
"""
import argparse, os
import numpy as np, pandas as pd
from scipy import stats
from emm_corr_rcs import load_data, spearman_vec, fisher, se_fisher, TARGETS, PAIR_NAMES, DIAG_LABEL

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", default="results_baseline")
ap.add_argument("--min-n", type=int, default=30)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
df = load_data(a.data)

contrasts = [  # (diagnosis code, label, mask A, mask B)
    (3, "BPTB vs non-BPTB", lambda d: d.sample_origin == "BPTB", lambda d: d.sample_origin != "BPTB"),
    (3, "BPTB vs LIV", lambda d: d.sample_origin == "BPTB", lambda d: d.sample_origin == "LIV"),
    (3, "Cohort1 vs Cohort2", lambda d: d.patient_cohort == "Cohort1", lambda d: d.patient_cohort == "Cohort2"),
    (2, "BPTB vs non-BPTB", lambda d: d.sample_origin == "BPTB", lambda d: d.sample_origin != "BPTB"),
    (2, "BPTB vs LIV", lambda d: d.sample_origin == "BPTB", lambda d: d.sample_origin == "LIV"),
    (2, "Cohort1 vs Cohort2", lambda d: d.patient_cohort == "Cohort1", lambda d: d.patient_cohort == "Cohort2"),
]
rows = []
for code, label, fa, fb in contrasts:
    d = df[df.diagnosis == code]
    A, B = d[fa(d)], d[fb(d)]
    if len(A) < a.min_n or len(B) < a.min_n:
        print(f"skip {DIAG_LABEL[code]} {label}: n={len(A)}/{len(B)}")
        continue
    ra, rb = spearman_vec(A[TARGETS].values), spearman_vec(B[TARGETS].values)
    for k, pn in enumerate(PAIR_NAMES):
        zz = (fisher(ra[k]) - fisher(rb[k])) / np.sqrt(se_fisher(len(A)) ** 2 + se_fisher(len(B)) ** 2)
        rows.append(dict(diagnosis=DIAG_LABEL[code], contrast=label, nA=len(A), rhoA=ra[k], nB=len(B), rhoB=rb[k],
                         pair=pn, diff=ra[k] - rb[k], z=zz, p=2 * stats.norm.sf(abs(zz))))
out = pd.DataFrame(rows)
m = len(out)
out["p_bonf"] = (out.p * m).clip(upper=1)
out["n_tests_total"] = m
out["significant_bonf_0.05"] = out.p_bonf < 0.05
out.to_csv(f"{a.out}/site_within_diagnosis_tests.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 20, "display.max_rows", 500)
print(out.round(3)[["diagnosis", "contrast", "nA", "rhoA", "nB", "rhoB", "pair", "diff", "p", "p_bonf"]].to_string(index=False))
print(f"\nBonferroni over {m} tests. Significant:")
print(out[out["significant_bonf_0.05"]][["diagnosis", "contrast", "pair", "diff", "p_bonf"]].round(3).to_string(index=False))
