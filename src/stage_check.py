#!/usr/bin/env python3
"""
Targeted check of the recurring EMM hint: do advanced-stage PDAC patients (stage III-IV) show weaker
REG1B~TFF1 (and other) correlations than early-stage PDAC (stage I-IIB)?  Overall and within site.
This is a HYPOTHESIS GENERATED from the same data, so treat results as exploratory.
Usage: python stage_check.py --data "Debernardi et al 2020 data.csv" --out results_dilution
"""
import argparse, os
import numpy as np, pandas as pd
from scipy import stats
import emm_corr_rcs as E

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", default="results_dilution")
ap.add_argument("--min-n", type=int, default=20)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
df = E.load_data(a.data)
p = df[(df.diagnosis == 3) & df.stage_ord.notna()].copy()
p["stage_group"] = np.where(p.stage_ord >= 7, "advanced (III-IV)", "early (I-IIB)")
pd.set_option("display.width", 250, "display.max_columns", 30)
print("PDAC stage by site:")
print(pd.crosstab(p.stage, p.sample_origin, margins=True))
print("\nStage group by site:")
print(pd.crosstab(p.stage_group, p.sample_origin, margins=True))
rows = []
for sname, d in {"all PDAC": p, "PDAC BPTB": p[p.sample_origin == "BPTB"], "PDAC LIV": p[p.sample_origin == "LIV"]}.items():
    e, ad = d[d.stage_group.str.startswith("early")], d[d.stage_group.str.startswith("adv")]
    if len(e) < a.min_n or len(ad) < a.min_n:
        print(f"skip {sname}: early n={len(e)}, advanced n={len(ad)}")
        continue
    r1, r2 = E.spearman_vec(e[E.TARGETS].values), E.spearman_vec(ad[E.TARGETS].values)
    for k, pn in enumerate(E.PAIR_NAMES):
        z = (E.fisher(r1[k]) - E.fisher(r2[k])) / np.sqrt(E.se_fisher(len(e)) ** 2 + E.se_fisher(len(ad)) ** 2)
        rows.append(dict(stratum=sname, n_early=len(e), n_advanced=len(ad), pair=pn, rho_early=r1[k], rho_advanced=r2[k],
                         diff_early_minus_adv=r1[k] - r2[k], z=z, p=2 * stats.norm.sf(abs(z))))
R = pd.DataFrame(rows)
R["p_bonf"] = (R.p * len(R)).clip(upper=1)
R.to_csv(f"{a.out}/stage_check.csv", index=False)
print(f"\nEarly vs advanced stage, Spearman by pair (Bonferroni over {len(R)} tests):")
print(R.round(3).to_string(index=False))
