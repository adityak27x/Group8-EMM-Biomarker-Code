#!/usr/bin/env python3
"""Generate a synthetic data set with the same schema as the Debernardi et al. (2020) urinary biomarker data.

The values are random (with a weak planted effect) and carry no clinical meaning; the file only lets you test
that the pipeline runs end to end without the real data.
Usage: python tests/make_synthetic_data.py data/synthetic.csv
"""
import sys
import numpy as np
import pandas as pd


def make(n=590, seed=0):
    rng = np.random.default_rng(seed)
    diag = np.r_[np.ones(183), 2 * np.ones(208), 3 * np.ones(199)].astype(int)
    age = rng.integers(26, 90, n)
    sex = rng.choice(["M", "F"], n)
    base = rng.normal(size=(n, 1))
    Z = np.exp(0.8 * base + rng.normal(size=(n, 4)) * 0.7)
    planted = (diag == 3) & (age > 60)
    Z[planted, 0] = np.exp(rng.normal(size=planted.sum()))
    stage = np.where(diag == 3, rng.choice(["IA", "IIB", "III", "IV"], n), None)
    return pd.DataFrame(dict(
        sample_id=range(n),
        patient_cohort=np.where(rng.random(n) < .6, "Cohort1", "Cohort2"),
        sample_origin=rng.choice(["BPTB", "ESP", "LIV", "UCL"], n),
        age=age, sex=sex, diagnosis=diag, stage=stage,
        benign_sample_diagnosis=np.where(diag == 2, "Cholecystitis", None),
        plasma_CA19_9=np.where(rng.random(n) < .4, rng.lognormal(3, 2, n), np.nan),
        creatinine=Z[:, 0], LYVE1=Z[:, 1], REG1B=Z[:, 2], TFF1=Z[:, 3], REG1A=np.nan))


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "data/synthetic.csv"
    make().to_csv(out, index=False)
    print("wrote", out)
