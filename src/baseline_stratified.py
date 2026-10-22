#!/usr/bin/env python3
"""Pairwise diagnosis comparisons of biomarker Spearman correlations, overall and
within strata (site BPTB, Cohort1, Cohort2) to separate diagnosis effects from site/cohort effects.
Needs emm_corr_rcs.py in the same folder.
Usage: python baseline_stratified.py --data "Debernardi et al 2020 data.csv" --out results_baseline
"""
import argparse, itertools, os
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

strata = {"all patients": df,
          "site BPTB only": df[df.sample_origin == "BPTB"],
          "Cohort1 only": df[df.patient_cohort == "Cohort1"],
          "Cohort2 only": df[df.patient_cohort == "Cohort2"]}
rows = []
for sname, d in strata.items():
    groups = {DIAG_LABEL[k]: d[d.diagnosis == k] for k in (1, 2, 3) if (d.diagnosis == k).sum() >= a.min_n}
    rho = {g: spearman_vec(x[TARGETS].values) for g, x in groups.items()}
    comps = list(itertools.combinations(groups, 2))
    n_tests = len(comps) * len(PAIR_NAMES)
    for g1, g2 in comps:
        n1, n2 = len(groups[g1]), len(groups[g2])
        for k, pn in enumerate(PAIR_NAMES):
            zz = (fisher(rho[g1][k]) - fisher(rho[g2][k])) / np.sqrt(se_fisher(n1) ** 2 + se_fisher(n2) ** 2)
            p = 2 * stats.norm.sf(abs(zz))
            rows.append(dict(stratum=sname, group1=g1, n1=n1, rho1=rho[g1][k], group2=g2, n2=n2, rho2=rho[g2][k],
                             pair=pn, diff=rho[g1][k] - rho[g2][k], z=zz, p=p,
                             p_bonf=min(1, p * n_tests), n_tests_in_stratum=n_tests))
out = pd.DataFrame(rows)
out["significant_bonf_0.05"] = out.p_bonf < 0.05
out.to_csv(f"{a.out}/baseline_stratified_tests.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 20, "display.max_rows", 500)
print(out.round(3)[["stratum", "group1", "n1", "rho1", "group2", "n2", "rho2", "pair", "diff", "p", "p_bonf"]].to_string(index=False))
print("\nSignificant after Bonferroni within stratum:")
print(out[out["significant_bonf_0.05"]].groupby(["stratum", "group1", "group2"]).pair.apply(lambda s: ", ".join(s)).to_string())
