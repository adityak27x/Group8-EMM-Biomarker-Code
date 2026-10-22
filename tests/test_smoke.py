#!/usr/bin/env python3
"""Smoke test: runs a tiny version of the pipeline on synthetic data (takes about a minute).
Run with `python tests/test_smoke.py` (or `pytest tests`)."""
import os, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(*args):
    subprocess.run([sys.executable, *args], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)


def test_smoke():
    with tempfile.TemporaryDirectory() as tmp:
        data = os.path.join(tmp, "synthetic.csv")
        run("tests/make_synthetic_data.py", data)
        out = os.path.join(tmp, "res")
        run("src/emm_heavy.py", "--data", data, "--preset", "quick", "--out", out, "--workers", "1",
            "--experiments", "E4_PDAC_only", "--power-experiments", "E4_PDAC_only", "--power-reps", "1",
            "--power-sizes", "40", "--power-deltas", "0.5", "--n-perm", "5", "--drop-descriptors", "plasma_CA19_9")
        for f in ("E4_PDAC_only/summary.json", "E4_PDAC_only/FINAL_subgroups_main_tau.csv",
                  "E4_PDAC_only/permutation_null.npy", "E4_PDAC_only/validation_exhaustive3.json",
                  "E4_PDAC_only/power_simulation.csv", "HEAVY_SUMMARY.csv", "run_config.json"):
            assert os.path.exists(os.path.join(out, f)), f
        run("src/baseline_stratified.py", "--data", data, "--out", os.path.join(tmp, "base"))
        run("src/stage_check.py", "--data", data, "--out", os.path.join(tmp, "dil"))
        assert os.path.exists(os.path.join(tmp, "base", "baseline_stratified_tests.csv"))
        assert os.path.exists(os.path.join(tmp, "dil", "stage_check.csv"))
    print("smoke test passed")


if __name__ == "__main__":
    test_smoke()
