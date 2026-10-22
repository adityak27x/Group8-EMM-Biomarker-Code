#!/usr/bin/env python3
"""Reproduce every analysis behind the paper, in order.

Usage:  python scripts/run_all.py --data data/Debernardi_et_al_2020_data.csv --out results [--workers N] [--quick]

--quick uses tiny settings (for testing the pipeline); without it the full settings of the paper are used
(heavy preset, 1000 randomizations, 50 power repetitions; allow a few hours on a multi-core machine).
"""
import argparse, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", default="results")
ap.add_argument("--workers", type=int, default=None)
ap.add_argument("--quick", action="store_true")
a = ap.parse_args()


def run(script, *args):
    cmd = [sys.executable, os.path.join(ROOT, script), *map(str, args)]
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


W = ["--workers", a.workers] if a.workers else []
HEAVY, STANDARD = ("quick", "quick") if a.quick else ("heavy", "standard")
NOCA = ["--drop-descriptors", "plasma_CA19_9"]
O = lambda name: os.path.join(a.out, name)
D = ["--data", a.data]

# 1 primary analysis (paper E1-E5 = E1_clinical, E4_PDAC_only, E5_PDAC_with_origin, E6_controls_only, E7_benign_only)
run("src/emm_heavy.py", *D, "--preset", HEAVY, "--out", O("heavy_noCA199"), *NOCA, "--power-reps", 0, *W)
# 2 plain null (mentor's randomization test) for the comparison in Table/Figure 1
run("src/emm_heavy.py", *D, "--preset", HEAVY, "--out", O("heavy_plainnull"), *NOCA, "--perm-strata", "none",
    "--experiments", "E1_clinical", "E4_PDAC_only", "E5_PDAC_with_origin", "E7_benign_only", "--power-reps", 0, *W)
# 3 CA19-9 allowed as descriptor (sensitivity)
run("src/emm_heavy.py", *D, "--preset", HEAVY, "--out", O("heavy_withCA199"), "--power-reps", 0, *W)
# 4 partial-correlation model (sensitivity)
run("src/emm_heavy.py", *D, "--preset", STANDARD, "--model", "partial", "--out", O("heavy_partial"), *NOCA,
    "--experiments", "E1_clinical", "E4_PDAC_only", "E7_benign_only", "--power-reps", 0, *W)
# 5 power simulation (E1 and E2 of the paper)
power = ["--power-reps", 2, "--power-sizes", 40, 100, "--power-deltas", 0.5] if a.quick else \
        ["--power-reps", 50, "--power-sizes", 30, 50, 80, 120, "--power-deltas", 0.3, 0.4, 0.5, 0.6]
run("src/emm_heavy.py", *D, "--preset", HEAVY, "--out", O("heavy_power"), *NOCA,
    "--experiments", "E1_clinical", "E4_PDAC_only", "--power-experiments", "E1_clinical", "E4_PDAC_only", *power, *W)
# 6 E4 (controls) artifact checks: cohort-stratified null and minimum size 50
run("src/emm_heavy.py", *D, "--preset", HEAVY, "--out", O("heavy_E6_cohort"), "--experiments", "E6_controls_only",
    "--perm-strata", "patient_cohort", "--power-reps", 0, *W)
run("src/emm_heavy.py", *D, "--preset", HEAVY, "--out", O("heavy_E6_min50"), "--experiments", "E6_controls_only",
    "--min-size", 50, "--perm-strata", "patient_cohort", "--power-reps", 0, *W)
# 7 supporting analyses (group comparisons, site/cohort effects, creatinine, robustness, stage)
run("src/baseline_stratified.py", *D, "--out", O("baseline"))
run("src/site_within_diagnosis.py", *D, "--out", O("baseline"))
run("src/dilution_check.py", *D, "--out", O("dilution"))
run("src/bptb_robustness.py", *D, "--out", O("dilution"))
run("src/stage_check.py", *D, "--out", O("dilution"))
# 8 figures
run("scripts_svg/make_null_figure.py", "--stratified", O("heavy_noCA199"), "--plain", O("heavy_plainnull"),
    "--out", os.path.join(ROOT, "figures", "fig_null.svg"))
run("scripts_svg/make_paper_figures.py")
print("\nAll analyses finished. Results:", os.path.abspath(a.out))
