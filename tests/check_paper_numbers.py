#!/usr/bin/env python3
"""Reproduction check: recompute the key numbers of the paper from the real data and compare them.

Usage:
  python tests/check_paper_numbers.py --data data/Debernardi_et_al_2020_data.csv
  python tests/check_paper_numbers.py --data data/Debernardi_et_al_2020_data.csv --null E4_PDAC_only E5_PDAC_with_origin E6_controls_only

The default check is deterministic and takes about half a minute: search statistics (Table 1), baseline correlations,
and the supporting analyses. With --null it also reruns the 1000-randomization null for the listed experiments
(roughly 15-20 minutes per experiment on one core) and checks the null percentiles and smallest p-values of Table 1.
"""
import argparse, json, os, subprocess, sys, tempfile
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import emm_corr_rcs as E  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.join(ROOT, "data", "Debernardi_et_al_2020_data.csv"))
ap.add_argument("--null", nargs="*", default=[], help="experiments for which the full 1000-randomization null is rerun")
ap.add_argument("--workers", default="1")
a = ap.parse_args()

# ---- values reported in the paper (Table 1) ------------------------------------------------
TABLE1 = {  # code label: (conditions, evaluated, best phi, null95, null99, min p)
    "E1_clinical": (136, 4861, 5.01, 5.93, 6.26, 1.00),
    "E4_PDAC_only": (40, 1067, 2.98, 4.18, 4.56, 0.79),
    "E5_PDAC_with_origin": (46, 1973, 3.59, 4.78, 5.31, 0.87),
    "E6_controls_only": (28, 277, 3.37, 3.80, 4.26, 0.16),
    "E7_benign_only": (128, 5137, 3.55, 4.51, 5.05, 0.69),
}
PAPER = {"E1_clinical": "E1", "E4_PDAC_only": "E2", "E5_PDAC_with_origin": "E3", "E6_controls_only": "E4", "E7_benign_only": "E5"}
results = []


def check(name, got, want, tol):
    ok = got is not None and abs(float(got) - float(want)) <= tol
    results.append((name, got, want, "PASS" if ok else "FAIL"))


def run(script, *args):
    subprocess.run([sys.executable, os.path.join(ROOT, "src", script), *args], check=True, stdout=subprocess.DEVNULL, cwd=ROOT)


tmp = tempfile.mkdtemp()
common = ["--data", a.data]
# 1. search statistics
out = os.path.join(tmp, "search")
run("emm_heavy.py", *common, "--preset", "heavy", "--n-perm", "0", "--power-reps", "0", "--workers", a.workers, "--out", out,
    "--drop-descriptors", "plasma_CA19_9")
for exp, (cond, ev, phi, *_rest) in TABLE1.items():
    s = json.load(open(os.path.join(out, exp, "summary.json")))
    check(f"{PAPER[exp]} conditions", s["conditions"], cond, 0)
    check(f"{PAPER[exp]} subgroups evaluated", s["evaluated"], ev, 0)
    check(f"{PAPER[exp]} best phi", s["best_phi"], phi, 0.006)
    v = json.load(open(os.path.join(out, exp, "validation_exhaustive3.json")))
    check(f"{PAPER[exp]} top-20 recovered by hybrid search", v["top20_of_exhaustive_found_by_hybrid"], 20, 0)

# 2. baseline correlations (Figure S1 / baseline subsection)
df = E.load_data(a.data)
rho = {g: E.spearman_vec(d[E.TARGETS].values) for g, d in
       {"all": df, "control": df[df.diagnosis == 1], "benign": df[df.diagnosis == 2], "PDAC": df[df.diagnosis == 3]}.items()}
for g, pair, want in [("control", 0, .659), ("benign", 0, .412), ("PDAC", 0, .342), ("all", 3, .648), ("control", 3, .459), ("PDAC", 5, .655)]:
    check(f"Spearman {g} {E.PAIR_NAMES[pair]}", rho[g][pair], want, 0.0015)

# 3. supporting analyses
run("stage_check.py", *common, "--out", os.path.join(tmp, "dil"))
st = pd.read_csv(os.path.join(tmp, "dil", "stage_check.csv"))
r = st[(st.stratum == "all PDAC") & (st.pair == "REG1B~TFF1")].iloc[0]
check("stage I-IIB REG1B~TFF1", r.rho_early, .735, 0.0015); check("stage III-IV REG1B~TFF1", r.rho_advanced, .565, 0.0015)
check("stage contrast p (unadjusted)", r.p, .043, 0.0015)
run("bptb_robustness.py", *common, "--out", os.path.join(tmp, "dil"))
rb = pd.read_csv(os.path.join(tmp, "dil", "bptb_robustness.csv"))
r = rb[(rb.stratum == "BPTB Cohort2") & (rb.contrast == "control vs benign") & (rb.method == "partial") & (rb.pair == "LYVE1~REG1B")].iloc[0]
check("Cohort2 control partial LYVE1~REG1B", r.rho1, .207, 0.0015); check("Cohort2 benign partial LYVE1~REG1B", r.rho2, .582, 0.0015)
r = rb[(rb.stratum == "BPTB all") & (rb.contrast == "control vs benign") & (rb.method == "partial") & (rb.pair == "LYVE1~REG1B")].iloc[0]
check("BPTB control partial LYVE1~REG1B", r.rho1, .192, 0.0015); check("BPTB benign partial LYVE1~REG1B", r.rho2, .558, 0.0015)
check("BPTB adjusted p (51 tests)", r.p_bonf_within_method, .009, 0.0015)
run("site_within_diagnosis.py", *common, "--out", os.path.join(tmp, "base"))
sd = pd.read_csv(os.path.join(tmp, "base", "site_within_diagnosis_tests.csv"))
r = sd[(sd.diagnosis == "PDAC") & (sd.contrast == "BPTB vs LIV") & (sd.pair == "creatinine~REG1B")].iloc[0]
check("PDAC creatinine~REG1B BPTB", r.rhoA, .422, 0.0015); check("PDAC creatinine~REG1B LIV", r.rhoB, -.121, 0.0015)
check("site contrast Bonferroni p (36 tests)", r.p_bonf, .011, 0.0015)

# 4. optional: full randomization null
if a.null:
    out = os.path.join(tmp, "null")
    run("emm_heavy.py", *common, "--preset", "heavy", "--power-reps", "0", "--workers", a.workers, "--out", out,
        "--drop-descriptors", "plasma_CA19_9", "--experiments", *a.null)
    for exp in a.null:
        s = json.load(open(os.path.join(out, exp, "summary.json"))); _, _, _, p95, p99, pmin = TABLE1[exp]
        check(f"{PAPER[exp]} null 95th percentile", s["null_p95"], p95, 0.006)
        check(f"{PAPER[exp]} null 99th percentile", s["null_p99"], p99, 0.006)
        check(f"{PAPER[exp]} smallest p", s["min_perm_p"], pmin, 0.006)

w = max(len(r[0]) for r in results)
for name, got, want, status in results:
    print(f"{status}  {name:<{w}}  reproduced {got!s:>10.10}   paper {want}")
bad = [r for r in results if r[3] == "FAIL"]
print(f"\n{len(results) - len(bad)}/{len(results)} checks passed")
sys.exit(1 if bad else 0)
