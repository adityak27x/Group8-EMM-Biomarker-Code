# Do Urinary Pancreatic Cancer Biomarkers Interact Differently in Patient Subgroups? An Exceptional Model Mining Study

Implementation accompanying the paper *Do Urinary Pancreatic Cancer Biomarkers Interact Differently in Patient Subgroups? An Exceptional Model Mining Study*.
The code searches for patient subgroups in which the correlations among four urinary biomarkers
(creatinine, LYVE1, REG1B, TFF1) deviate exceptionally from the rest of the data, validates the result with a
stratified randomization test, and quantifies what the search can detect with a planted-subgroup power simulation.

## Contents

```
src/        pipeline and supporting analyses
scripts/    figure scripts and a driver that reproduces all analyses
tests/      reproduction check against the paper, smoke test and synthetic-data generator
data/       the data file and a description of its columns
paper_figures/   the final figures of the paper
```

| File | Purpose |
|---|---|
| `src/emm_corr_rcs.py` | Core functions (Spearman model class, Fisher-z quality, description language, response-indicator refinement, redundancy filters) and the first, preliminary beam-search runs |
| `src/emm_heavy.py` | **Primary analysis**: hybrid search, validation against exhaustive depth 3, stratified randomization test, redundancy filtering, power simulation, partial-correlation model |
| `src/emm_site_stratified.py` | Site-as-occasion EMM and the partial-correlation model (the site-stratified runs are not part of the paper) |
| `src/baseline_stratified.py`, `src/site_within_diagnosis.py` | Group comparisons of correlations by diagnosis, site and cohort (Fisher test, Bonferroni) |
| `src/dilution_check.py`, `src/bptb_robustness.py`, `src/stage_check.py` | Creatinine-range check and creatinine-controlled contrasts; robustness within site BPTB; stage III-IV check |
| `scripts/make_null_figure.py` | Figure 1 (null distributions) from the saved randomization arrays |
| `scripts/make_paper_figures.py` | Remaining figures, drawn from numbers transcribed from the result files |
| `scripts/run_all.py` | Runs every analysis and figure in order |

## Installation

Python 3.9 or newer.

```
pip install -r requirements.txt
```

## Data

The urinary biomarker data of Debernardi et al. (2020), *PLOS Medicine* 17(12): e1003489, are included unchanged as
`data/Debernardi_et_al_2020_data.csv` (590 patients: 183 controls, 208 benign, 199 PDAC), so every command below runs as written.
See `data/README.md` for the columns and the attribution note.

## Check the installation

```
pip install -r requirements.txt
python tests/check_paper_numbers.py
```

This recomputes the key numbers of the paper from the data (search statistics of Table 1, the validation against the exhaustive
search, baseline correlations and the supporting analyses) and compares them with the published values. It takes about
half a minute and should report `37/37 checks passed`. (We verified this on the data file in this repository, and also that the full
1000-randomization null of the controls experiment reproduces the published percentiles exactly.) Add `--null E4_PDAC_only E5_PDAC_with_origin E6_controls_only` to also rerun the
1000-randomization null for those experiments and check the null percentiles and smallest p-values (a few minutes per experiment).

Without the real data you can still test the code on synthetic data:

```
python tests/make_synthetic_data.py data/synthetic.csv
python tests/test_smoke.py
```

`python scripts/run_all.py --data data/Debernardi_et_al_2020_data.csv --quick` runs the complete pipeline with tiny settings (a few minutes).

## Reproducing the paper

Run everything (full settings: 1000 randomizations, 50 power repetitions; allow several hours on a multi-core machine):

```
python scripts/run_all.py --data data/Debernardi_et_al_2020_data.csv --out results --workers 6
```

Or run the steps individually (from the repository root; results go to `results/`).

**Experiment labels.** The paper uses E1-E5; the code uses the labels on the right.

| Paper | Population | Code label |
|---|---|---|
| E1 | all patients | `E1_clinical` |
| E2 | PDAC only | `E4_PDAC_only` |
| E3 | PDAC, site as descriptor | `E5_PDAC_with_origin` |
| E4 | controls only | `E6_controls_only` |
| E5 | benign only | `E7_benign_only` |

(`E2_with_diagnosis` and `E3_with_origin_cohort` exist only for the preliminary runs of `emm_corr_rcs.py`.)

**Primary analysis** (Table 1, Table 2, Figure 1): no CA19-9 as descriptor, 1000 randomizations, null shuffled within diagnosis x site.

```
python src/emm_heavy.py --data data/Debernardi_et_al_2020_data.csv --preset heavy --out results/heavy_noCA199 --drop-descriptors plasma_CA19_9 --power-reps 0
```

**Plain null** (comparison in the confounding subsection and Figure 1):

```
python src/emm_heavy.py --data data/Debernardi_et_al_2020_data.csv --preset heavy --out results/heavy_plainnull --drop-descriptors plasma_CA19_9 --perm-strata none --experiments E1_clinical E4_PDAC_only E5_PDAC_with_origin E7_benign_only --power-reps 0
```

**Sensitivity analyses**

```
# CA19-9 allowed as descriptor
python src/emm_heavy.py --data data/Debernardi_et_al_2020_data.csv --preset heavy --out results/heavy_withCA199 --power-reps 0
# partial-correlation model (controls for creatinine)
python src/emm_heavy.py --data data/Debernardi_et_al_2020_data.csv --preset standard --model partial --out results/heavy_partial --drop-descriptors plasma_CA19_9 --experiments E1_clinical E4_PDAC_only E7_benign_only --power-reps 0
# controls only (paper E4): cohort-stratified null, and minimum size 50
python src/emm_heavy.py --data data/Debernardi_et_al_2020_data.csv --preset heavy --out results/heavy_E6_cohort --experiments E6_controls_only --perm-strata patient_cohort --power-reps 0
python src/emm_heavy.py --data data/Debernardi_et_al_2020_data.csv --preset heavy --out results/heavy_E6_min50 --experiments E6_controls_only --min-size 50 --perm-strata patient_cohort --power-reps 0
```

**Power simulation** (Figure 2):

```
python src/emm_heavy.py --data data/Debernardi_et_al_2020_data.csv --preset heavy --out results/heavy_power --drop-descriptors plasma_CA19_9 --experiments E1_clinical E4_PDAC_only --power-experiments E1_clinical E4_PDAC_only --power-reps 50 --power-sizes 30 50 80 120 --power-deltas 0.3 0.4 0.5 0.6
```

**Supporting analyses** (design, site and cohort effects, creatinine-controlled contrasts, robustness, stage):

```
python src/baseline_stratified.py --data data/Debernardi_et_al_2020_data.csv --out results/baseline
python src/site_within_diagnosis.py --data data/Debernardi_et_al_2020_data.csv --out results/baseline
python src/dilution_check.py --data data/Debernardi_et_al_2020_data.csv --out results/dilution
python src/bptb_robustness.py --data data/Debernardi_et_al_2020_data.csv --out results/dilution
python src/stage_check.py --data data/Debernardi_et_al_2020_data.csv --out results/dilution
```

**Preliminary beam-search runs** (cited as preliminary in the paper; also writes the baseline correlation tables):

```
python src/emm_corr_rcs.py --data data/Debernardi_et_al_2020_data.csv --out results/preliminary --n-perm 500 --n-boot 500
```

**Figures**

```
python scripts/make_null_figure.py --stratified results/heavy_noCA199 --plain results/heavy_plainnull --out figures/fig_null.png
python scripts/make_paper_figures.py
```

The final figures as used in the paper are in `paper_figures/`.

## Main options of `emm_heavy.py`

| Option | Meaning | Default |
|---|---|---|
| `--preset` | `quick`, `standard`, `heavy`, `extreme` (randomizations, thresholds, search width and depth) | `standard` |
| `--experiments` | which experiments to run | E1, E4, E5, E6, E7 |
| `--model` | `raw` (6 Spearman pairs) or `partial` (3 pairs, controlling for creatinine) | `raw` |
| `--reference` | compare each subgroup with the `dataset` or with its `complement` (not used in the paper) | `dataset` |
| `--perm-strata` | `auto` (diagnosis x site), `none` (plain null) or column names, e.g. `patient_cohort` | `auto` |
| `--drop-descriptors` | descriptors to exclude, e.g. `plasma_CA19_9` | none |
| `--min-size` | minimum size of subgroup and complement | 30 |
| `--only-site` | restrict the analysis to given sites, e.g. `BPTB` | all |
| `--n-perm`, `--workers`, `--seed` | number of randomizations, processes, random seed | preset, CPU count - 1, 1 |
| `--power-experiments`, `--power-reps`, `--power-sizes`, `--power-deltas` | power simulation settings | see `--help` |

The `heavy` preset corresponds to the paper: 12 quantile cut points, exhaustive search to 2 conditions, beam width 150 to depth 4,
1000 randomizations, top 500 subgroups passed to the redundancy filters, q = 20 reported, tau = 0.10, Jaccard threshold 0.5.

## Output

Each experiment folder contains `FINAL_subgroups_main_tau.csv` (reported subgroups with rule, size, pair, correlations,
quality, permutation p-value and diagnosis/site/cohort composition), the unfiltered and filtered lists
(`1_raw_pool_top.csv`, `2_after_row_overlap_filter.csv`, `3_after_model_filter_tau_*.csv`), `retained_counts.csv`,
`diversity_quality_metrics.csv`, `permutation_null.npy`, `validation_exhaustive3.json`, `summary.json` and figures.
The top level holds `HEAVY_SUMMARY.csv` and `run_config.json` with the exact settings of the run.

## Method in brief

* Model class: pairwise Spearman correlations (partial correlations as sensitivity model).
* Quality: `z = |atanh(rho_SG) - atanh(rho_ref)| / sqrt(1.06 / (n - 3 - k))` per pair; subgroup quality is the maximum over pairs.
* Missing descriptors: response-indicator conditions (`attribute is missing`), as in EMM-RCS.
* Validation: biomarker vectors are shuffled jointly within diagnosis x site strata 1000 times, the whole search is repeated each
  time, and a subgroup is significant if its quality exceeds the 95th percentile of the best qualities (family-wise control).
* Power: subgroups with a known change in correlation are planted into the real data and the pipeline is rerun.

Details and equations are in the paper (Methodology).

## Reproducibility notes

* All random choices use a fixed seed (`--seed`, default 1); results can differ slightly across platforms and library versions.
* `--workers` sets the number of processes. On Windows each worker re-imports the script, which is handled by the
  `if __name__ == "__main__"` guards.
* A full run of the primary analysis takes a few minutes per experiment on 6-15 cores; the full reproduction several hours.


