#!/usr/bin/env python3
"""
Robustness of the BPTB diagnosis effect on creatinine-controlled biomarker correlations.
Checks whether control-vs-benign / control-vs-PDAC differences in partial Spearman correlations survive
restriction to one cohort, a common age band, or one sex.
Needs emm_corr_rcs.py and emm_site_stratified.py in the same folder.
Usage: python bptb_robustness.py --data "Debernardi et al 2020 data.csv" --out results_dilution
"""
import argparse, itertools, os
import numpy as np, pandas as pd
from scipy import stats
import emm_corr_rcs as E
from emm_site_stratified import model_rho, BIO

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", default="results_dilution")
ap.add_argument("--min-n", type=int, default=25)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
df = E.load_data(a.data)
b = df[df.sample_origin == "BPTB"]
strata = {
    "BPTB all": b,
    "BPTB Cohort2": b[b.patient_cohort == "Cohort2"],
    "BPTB Cohort1": b[b.patient_cohort == "Cohort1"],
    "BPTB age 45-75": b[(b.age >= 45) & (b.age <= 75)],
    "BPTB Cohort2 + age 45-75": b[(b.patient_cohort == "Cohort2") & (b.age >= 45) & (b.age <= 75)],
    "BPTB female": b[b.sex == "F"],
    "BPTB male": b[b.sex == "M"],
}
names = [f"{BIO[i]}~{BIO[j]}" for i, j in itertools.combinations(range(3), 2)]
rows = []
for sname, d in strata.items():
    cnt = {k: int((d.diagnosis == k).sum()) for k in (1, 2, 3)}
    for c1, c2 in itertools.combinations((1, 2, 3), 2):
        g1, g2 = d[d.diagnosis == c1], d[d.diagnosis == c2]
        if len(g1) < a.min_n or len(g2) < a.min_n:
            continue
        for method in ("partial", "raw"):
            r1 = model_rho(g1[E.TARGETS].values, "partial")  if method == "partial" else model_rho(g1[E.TARGETS].values, "raw")[3:]
            r2 = model_rho(g2[E.TARGETS].values, "partial")  if method == "partial" else model_rho(g2[E.TARGETS].values, "raw")[3:]
            ex = 1 if method == "partial" else 0
            for k, pn in enumerate(names):
                se = np.sqrt(E.se_fisher(len(g1) - ex) ** 2 + E.se_fisher(len(g2) - ex) ** 2)
                z = (E.fisher(r1[k]) - E.fisher(r2[k])) / se
                rows.append(dict(stratum=sname, contrast=f"{E.DIAG_LABEL[c1]} vs {E.DIAG_LABEL[c2]}", n1=len(g1), n2=len(g2),
                                 method=method, pair=pn, rho1=r1[k], rho2=r2[k], diff=r1[k] - r2[k], z=z,
                                 p=2 * stats.norm.sf(abs(z))))
R = pd.DataFrame(rows)
R["p_bonf_within_method"] = R.groupby("method").p.transform(lambda s: (s * len(s)).clip(upper=1))
R.to_csv(f"{a.out}/bptb_robustness.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 400)
print("Group sizes per stratum (control/benign/PDAC):")
for s, d in strata.items():
    print(f"  {s:28s}", [int((d.diagnosis == k).sum()) for k in (1, 2, 3)])
key = R[(R.method == "partial") & (R.pair.isin(["LYVE1~REG1B", "REG1B~TFF1"])) & (R.contrast != "benign vs PDAC")]
print("\nPartial (creatinine-controlled) correlations, key pairs:")
print(key.round(3)[["stratum", "contrast", "n1", "n2", "pair", "rho1", "rho2", "diff", "p", "p_bonf_within_method"]].to_string(index=False))
k2 = key[key.pair == "LYVE1~REG1B"]
print(f"\nLYVE1~REG1B|creat, control minus disease: negative in {(k2['diff'] < 0).sum()} of {len(k2)} comparisons, "
      f"nominal p<0.05 in {(k2.p < 0.05).sum()}")
