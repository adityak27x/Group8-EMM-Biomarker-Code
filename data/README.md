# Data

`Debernardi_et_al_2020_data.csv` is the public urinary biomarker data set of

> Debernardi S. et al. (2020). A combination of urinary biomarker panel and PancRISK score for earlier detection of
> pancreatic cancer: A case-control study. *PLOS Medicine* 17(12): e1003489.

The file is included here unchanged so that the analyses can be reproduced. PLOS Medicine articles and their supporting data are
open access under a Creative Commons Attribution license (please confirm on the publication page); keep the attribution when
reusing the data.

Columns (590 rows):

| column | meaning |
|---|---|
| `sample_id` | identifier (not used) |
| `patient_cohort` | sampling cohort (`Cohort1`: 332, `Cohort2`: 258) |
| `sample_origin` | collection site (`BPTB`: 409, `LIV`: 132, `ESP`: 29, `UCL`: 20) |
| `age`, `sex` | demographics (`F`: 299, `M`: 291) |
| `diagnosis` | 1 = control (183), 2 = benign hepatobiliary disease (208), 3 = PDAC (199) |
| `stage` | PDAC stage (I, IA, IB, II, IIA, IIB, III, IV); missing for 391 non-PDAC patients |
| `benign_sample_diagnosis` | benign diagnosis; missing for 382 non-benign patients |
| `plasma_CA19_9` | plasma CA19-9; missing for 240 patients |
| `creatinine`, `LYVE1`, `REG1B`, `TFF1` | the four urinary targets (never missing) |
| `REG1A` | present in the file but missing for 284 patients and not part of the four-marker panel; not used |

A synthetic file with the same schema (random values, no clinical meaning) can be generated for testing:
`python tests/make_synthetic_data.py data/synthetic.csv`.
