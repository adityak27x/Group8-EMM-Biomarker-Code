#!/usr/bin/env python3
"""
EMM-Corr: Exceptional Model Mining with a Spearman-correlation model class
for the Debernardi et al. (2020) urinary pancreatic-cancer biomarker data.

Ideas borrowed from EMM-RCS (Schouten, Duivesteijn, Pechenizkiy, SDM 2022):
  * standard-error-scaled quality   z = |theta_SG - theta_0| / se(theta_SG)   (Eq. 4.2)
  * aggregation f over several model parameters (here: the 6 biomarker pairs)  (Eq. 4.1)
  * refinement of incomplete descriptors with a response indicator R_a=0        (Def. 4.3)
  * ordinal refinement (a <= w_h, a > w_h)                                      (Def. 4.4)
  * constraint on minimum subgroup size                                         (c_size)

Own extensions:
  * model = vector of 6 Spearman correlations; quality uses Fisher-z + Fieller SE
  * three-stage redundancy control: raw -> row-overlap (Jaccard) -> model-based (tau)
  * max-statistic permutation test over the WHOLE search (family-wise error control)
  * parametric Fisher test vs complement + Benjamini-Hochberg over all evaluated subgroups
  * bootstrap CI, split-half replication, per-cohort consistency
  * confounding diagnostics: diagnosis / origin / cohort purity of every subgroup

Usage:
  python emm_corr_rcs.py --data Debernardi_et_al_2020_data.csv --out results
"""
import argparse, itertools, json, os, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TARGETS = ["creatinine", "LYVE1", "REG1B", "TFF1"]
PAIRS = list(itertools.combinations(range(len(TARGETS)), 2))
PAIR_NAMES = [f"{TARGETS[i]}~{TARGETS[j]}" for i, j in PAIRS]
IU = np.triu_indices(len(TARGETS), 1)
STAGE_ORDER = ["I", "IA", "IB", "II", "IIA", "IIB", "III", "IV"]
DIAG_LABEL = {1: "control", 2: "benign", 3: "PDAC"}
AGG = {"max": np.max, "avg": np.mean, "sum": np.sum}
TAUS = [0.01, 0.05, 0.10, 0.20]


# ----------------------------------------------------------------------------
# Model class and quality measure
# ----------------------------------------------------------------------------
def spearman_vec(Y):
    """Vector of the 6 pairwise Spearman correlations of an (n x 4) matrix."""
    R = stats.rankdata(Y, axis=0)
    with np.errstate(all="ignore"):
        C = np.corrcoef(R, rowvar=False)
    return np.nan_to_num(C[IU])


def fisher(r):
    return np.arctanh(np.clip(r, -0.999, 0.999))


def se_fisher(n):
    """SE of Fisher-z of a Spearman rho (Fieller, Hartley & Pearson 1957)."""
    return np.sqrt(1.06 / (n - 3))


def zscores(r, n, r_ref):
    """Eq. 4.2 of EMM-RCS, with theta = Fisher-z(rho) per biomarker pair."""
    return np.abs(fisher(r) - fisher(r_ref)) / se_fisher(n)


# ----------------------------------------------------------------------------
# Descriptor conditions (refinement operator)
# ----------------------------------------------------------------------------
def build_conditions(df, desc_spec, n_bins=8):
    """Return list of (name, boolean mask). Missing-value logic follows Def. 4.3:
    value-conditions are False for missing entries, and an explicit
    '<attr> is missing' (R_a = 0) condition is added for incomplete attributes."""
    conds, seen = [], set()

    def add(name, mask):
        key = np.packbits(mask).tobytes()
        if key not in seen and 0 < mask.sum() < len(mask):
            seen.add(key)
            conds.append((name, mask))

    for col, kind in desc_spec.items():
        s = df[col]
        miss = s.isna().values
        if s.notna().sum() < 10:   # attribute (almost) never observed in this subset
            continue
        if kind in ("numeric", "ordinal"):
            vals = s.dropna().values
            if kind == "ordinal":
                th = np.unique(vals)[:-1]
            else:
                th = np.unique(np.quantile(vals, np.linspace(0.1, 0.9, n_bins)))
            for w in th:
                lab = f"{w:g}"
                if col == "stage_ord":
                    lab = STAGE_ORDER[int(w) - 1]
                nm = "stage" if col == "stage_ord" else col
                add(f"{nm} <= {lab}", (s <= w).values & ~miss)
                add(f"{nm} > {lab}", (s > w).values & ~miss)
        else:  # nominal
            for v in s.dropna().unique():
                add(f"{col} = {v}", (s == v).values & ~miss)
                add(f"{col} != {v}", (s != v).values & ~miss)
        if miss.any():
            nm = "stage" if col == "stage_ord" else col
            add(f"{nm} is missing", miss)
    return conds


# ----------------------------------------------------------------------------
# Beam search
# ----------------------------------------------------------------------------
def search(Y, masks, ref, cfg, keep=True):
    """Beam search. Returns (best score, list of result tuples or None).
    Result tuple: (score, desc_idx_tuple, mask, rho_vec, z_vec)."""
    N = len(Y)
    f = AGG[cfg["agg"]]
    seen, results, best = set(), [], -np.inf
    beam = [((), np.ones(N, bool))]
    for _ in range(cfg["depth"]):
        cand = []
        for desc, m in beam:
            msum = m.sum()
            for ci, cm in enumerate(masks):
                if ci in desc:
                    continue
                nm = m & cm
                n = nm.sum()
                if n < cfg["min_size"] or N - n < cfg["min_size"] or n == msum:
                    continue
                key = np.packbits(nm).tobytes()
                if key in seen:
                    continue
                seen.add(key)
                r = spearman_vec(Y[nm])
                z = zscores(r, n, ref)
                sc = f(z)
                best = max(best, sc)
                item = (sc, tuple(sorted(desc + (ci,))), nm, r, z)
                cand.append(item)
                if keep:
                    results.append(item)
        cand.sort(key=lambda x: -x[0])
        beam = [(c[1], c[2]) for c in cand[: cfg["width"]]]
        if not beam:
            break
    return best, (results if keep else None)


_G = {}


def _init(Y, masks, ref, cfg):
    _G.update(Y=Y, masks=masks, ref=ref, cfg=cfg)


def _perm_job(seed):
    rng = np.random.default_rng(seed)
    Yp = _G["Y"][rng.permutation(len(_G["Y"]))]  # rows stay intact, link to descriptors broken
    return search(Yp, _G["masks"], _G["ref"], _G["cfg"], keep=False)[0]


def permutation_null(Y, masks, ref, cfg, n_perm, workers, seed):
    if n_perm == 0:
        return np.array([])
    seeds = [seed * 100003 + i for i in range(n_perm)]
    if workers > 1:
        with ProcessPoolExecutor(workers, initializer=_init, initargs=(Y, masks, ref, cfg)) as ex:
            return np.array(list(ex.map(_perm_job, seeds, chunksize=4)))
    _init(Y, masks, ref, cfg)
    return np.array([_perm_job(s) for s in seeds])


# ----------------------------------------------------------------------------
# Redundancy filters
# ----------------------------------------------------------------------------
def jaccard(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def row_overlap_filter(rows, thr):
    kept = []
    for r in rows:  # rows sorted by score desc
        if all(jaccard(r["mask"], k["mask"]) <= thr for k in kept):
            kept.append(r)
    return kept


def model_distance(r1, r2):
    """Mean absolute difference between the two 6-vectors of Spearman correlations."""
    return float(np.mean(np.abs(r1 - r2)))


def model_filter(rows, tau):
    kept = []
    for r in rows:
        if all(model_distance(r["rho"], k["rho"]) >= tau for k in kept):
            kept.append(r)
    return kept


def diversity(rows, N, label):
    if not rows:
        return {"method": label, "k": 0}
    k = len(rows)
    M = np.array([r["mask"] for r in rows]).astype(int)
    inter = M @ M.T
    sz = M.sum(1)
    J = inter / (sz[:, None] + sz[None, :] - inter)
    rho = np.array([r["rho"] for r in rows])
    D = np.abs(rho[:, None, :] - rho[None, :, :]).mean(2)
    iu = np.triu_indices(k, 1)
    cnt = np.bincount([r["best_pair"] for r in rows], minlength=len(PAIRS)) / k
    ent = -np.sum(cnt[cnt > 0] * np.log2(cnt[cnt > 0]))
    attrs = {c.split(" ")[0] for r in rows for c in r["desc"].split(" AND ")}
    return {
        "method": label, "k": k,
        "mean_score": np.mean([r["score"] for r in rows]),
        "median_score": np.median([r["score"] for r in rows]),
        "mean_size": sz.mean(), "mean_coverage": sz.mean() / N,
        "union_coverage": float(M.max(0).mean()),
        "mean_pairwise_jaccard": J[iu].mean() if k > 1 else np.nan,
        "mean_pairwise_model_dist": D[iu].mean() if k > 1 else np.nan,
        "distinct_biomarker_pairs": int((cnt > 0).sum()),
        "pair_entropy_bits": ent,
        "distinct_attributes": len(attrs),
        "mean_description_length": np.mean([r["desc"].count(" AND ") + 1 for r in rows]),
    }


# ----------------------------------------------------------------------------
# Statistics for individual subgroups
# ----------------------------------------------------------------------------
def p_vs_complement(Y, mask, pair):
    i, j = PAIRS[pair]
    n1, n2 = mask.sum(), (~mask).sum()
    r1 = spearman_vec(Y[mask])[pair]
    r2 = spearman_vec(Y[~mask])[pair]
    zz = (fisher(r1) - fisher(r2)) / np.sqrt(se_fisher(n1) ** 2 + se_fisher(n2) ** 2)
    return 2 * stats.norm.sf(abs(zz)), r2


def bh(p):
    p = np.asarray(p)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    prev = 1.0
    for rank, idx in zip(range(m, 0, -1), order[::-1]):
        prev = min(prev, p[idx] * m / rank)
        adj[idx] = prev
    return adj


def bootstrap_ci(Y, mask, pair, n_boot, rng):
    N = len(Y)
    d = []
    for _ in range(n_boot):
        idx = rng.integers(0, N, N)
        m = mask[idx]
        if m.sum() < 10:
            continue
        d.append(spearman_vec(Y[idx][m])[pair] - spearman_vec(Y[idx])[pair])
    return np.percentile(d, [2.5, 97.5]) if len(d) > 20 else (np.nan, np.nan)


def split_half_rate(Y, mask, pair, obs_delta, reps, rng):
    N, ok, tot = len(Y), 0, 0
    for _ in range(reps):
        perm = rng.permutation(N)
        for half in (perm[: N // 2], perm[N // 2:]):
            m = mask[half]
            if m.sum() < 10:
                continue
            d = spearman_vec(Y[half][m])[pair] - spearman_vec(Y[half])[pair]
            tot += 1
            ok += (np.sign(d) == np.sign(obs_delta)) and (abs(d) >= 0.5 * abs(obs_delta))
    return ok / tot if tot else np.nan


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_data(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    for c in ["stage", "sex", "benign_sample_diagnosis", "sample_origin"]:
        if c in df:
            df[c] = df[c].astype("string").str.strip().replace({"": pd.NA})
    smap = {s: i + 1 for i, s in enumerate(STAGE_ORDER)}
    df["stage_ord"] = df["stage"].map(smap).astype(float) if "stage" in df else np.nan
    for c in ["plasma_CA19_9", "REG1A", "age"] + TARGETS:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["patient_cohort"] = df["patient_cohort"].astype("string")
    df["diagnosis_label"] = df["diagnosis"].map(DIAG_LABEL)
    miss_t = df[TARGETS].isna().any(axis=1).sum()
    if miss_t:
        print(f"[warn] dropping {miss_t} rows with missing target values")
        df = df.dropna(subset=TARGETS).reset_index(drop=True)
    return df


def experiments():
    base = {"age": "numeric", "sex": "nominal", "plasma_CA19_9": "numeric",
            "stage_ord": "ordinal", "benign_sample_diagnosis": "nominal"}
    return {
        "E1_clinical": dict(desc=dict(base), subset=None,
                            note="all patients; age, sex, CA19-9, stage, benign dx (+missing refinement)"),
        "E2_with_diagnosis": dict(desc={**base, "diagnosis_label": "nominal"}, subset=None,
                                  note="E1 + diagnosis as descriptor"),
        "E3_with_origin_cohort": dict(desc={**base, "sample_origin": "nominal", "patient_cohort": "nominal"},
                                      subset=None, note="E1 + origin + cohort as descriptors"),
        "E4_PDAC_only": dict(desc={"age": "numeric", "sex": "nominal", "plasma_CA19_9": "numeric",
                                   "stage_ord": "ordinal"},
                             subset=lambda d: d["diagnosis"] == 3,
                             note="PDAC patients only: stage is fully observed -> clean stage analysis"),
        "E5_PDAC_with_origin": dict(desc={"age": "numeric", "sex": "nominal", "plasma_CA19_9": "numeric",
                                          "stage_ord": "ordinal", "sample_origin": "nominal"},
                                    subset=lambda d: d["diagnosis"] == 3,
                                    note="PDAC only + collection site: is the LIV effect still there within one diagnosis?"),
        "E6_controls_only": dict(desc={"age": "numeric", "sex": "nominal", "plasma_CA19_9": "numeric",
                                       "sample_origin": "nominal", "patient_cohort": "nominal"},
                                 subset=lambda d: d["diagnosis"] == 1,
                                 note="controls only: no diagnosis shortcut possible"),
        "E7_benign_only": dict(desc={"age": "numeric", "sex": "nominal", "plasma_CA19_9": "numeric",
                                     "benign_sample_diagnosis": "nominal", "sample_origin": "nominal"},
                               subset=lambda d: d["diagnosis"] == 2,
                               note="benign only: no diagnosis shortcut possible"),
    }


# ----------------------------------------------------------------------------
# Baseline
# ----------------------------------------------------------------------------
def baseline_tables(df, out):
    rows = []
    groups = {"all": df, **{DIAG_LABEL[k]: df[df.diagnosis == k] for k in (1, 2, 3)}}
    for g, d in groups.items():
        r = spearman_vec(d[TARGETS].values)
        for p, name in enumerate(PAIR_NAMES):
            se = se_fisher(len(d))
            lo, hi = np.tanh(fisher(r[p]) - 1.96 * se), np.tanh(fisher(r[p]) + 1.96 * se)
            rows.append(dict(group=g, n=len(d), pair=name, rho=r[p], ci_low=lo, ci_high=hi))
    b = pd.DataFrame(rows)
    b.to_csv(f"{out}/baseline_correlations_by_diagnosis.csv", index=False)
    # check of the "creatinine~LYVE1: control vs PDAC" claim
    chk = []
    for p, name in enumerate(PAIR_NAMES):
        c = b[(b.group == "control") & (b.pair == name)].iloc[0]
        t = b[(b.group == "PDAC") & (b.pair == name)].iloc[0]
        zz = (fisher(c.rho) - fisher(t.rho)) / np.sqrt(se_fisher(c.n) ** 2 + se_fisher(t.n) ** 2)
        chk.append(dict(pair=name, rho_control=c.rho, rho_PDAC=t.rho, diff=c.rho - t.rho,
                        z=zz, p=2 * stats.norm.sf(abs(zz))))
    pd.DataFrame(chk).to_csv(f"{out}/baseline_control_vs_PDAC_test.csv", index=False)
    fig, ax = plt.subplots(1, 4, figsize=(17, 4))
    for a, (g, d) in zip(ax, groups.items()):
        C = np.eye(4)
        C[IU] = spearman_vec(d[TARGETS].values)
        C = np.triu(C) + np.triu(C, 1).T
        im = a.imshow(C, vmin=-1, vmax=1, cmap="RdBu_r")
        a.set_xticks(range(4)); a.set_yticks(range(4))
        a.set_xticklabels(TARGETS, rotation=45); a.set_yticklabels(TARGETS)
        a.set_title(f"{g} (n={len(d)})")
        for i in range(4):
            for j in range(4):
                a.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(f"{out}/fig_baseline_heatmaps.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return b


# ----------------------------------------------------------------------------
# One experiment
# ----------------------------------------------------------------------------
def run_experiment(name, spec, df_all, args):
    out = os.path.join(args.out, name)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    df = df_all if spec["subset"] is None else df_all[spec["subset"](df_all)].reset_index(drop=True)
    Y = df[TARGETS].values.astype(float)
    N = len(Y)
    desc_spec = {k: v for k, v in spec["desc"].items() if k not in args.drop_descriptors}
    conds = build_conditions(df, desc_spec, args.n_bins)
    masks = [c[1] for c in conds]
    ref = spearman_vec(Y)
    cfg = dict(depth=args.depth, width=args.width, min_size=args.min_size, agg=args.agg)
    print(f"\n=== {name}: N={N}, {len(conds)} conditions | {spec['note']}")

    # --- main search -------------------------------------------------------
    best, res = search(Y, masks, ref, cfg, keep=True)
    res.sort(key=lambda x: -x[0])
    M_eval = len(res)
    print(f"   evaluated {M_eval} distinct subgroups, best phi = {best:.2f}")

    rows = []
    for sc, desc, m, r, z in res:
        n = int(m.sum())
        bp = int(np.argmax(z))
        rows.append(dict(
            desc=" AND ".join(conds[i][0] for i in desc), mask=m, n=n, coverage=n / N, rho=r, z=z,
            score=sc, score_avg=float(z.mean()), score_sum=float(z.sum()),
            best_pair=bp, pair=PAIR_NAMES[bp], rho_sg=r[bp], rho_dataset=ref[bp], delta=r[bp] - ref[bp]))

    # parametric p-values vs complement (argmax pair) for ALL evaluated subgroups -> BH
    pv = []
    for r in rows:
        p, rc = p_vs_complement(Y, r["mask"], r["best_pair"])
        r["p_param"], r["rho_complement"] = p, rc
        pv.append(p)
    q = bh(pv)
    for r, qq in zip(rows, q):
        r["q_BH"] = qq

    # --- permutation null of the maximum quality ----------------------------
    null = permutation_null(Y, masks, ref, cfg, args.n_perm, args.workers, args.seed)
    if len(null):
        for r in rows:
            r["p_perm_FWER"] = (1 + (null >= r["score"]).sum()) / (1 + len(null))
        np.save(f"{out}/permutation_null_max_phi.npy", null)
        print(f"   permutation null (B={len(null)}): 95th pct of max-phi = {np.percentile(null, 95):.2f}")
    else:
        for r in rows:
            r["p_perm_FWER"] = np.nan

    # --- redundancy filtering ------------------------------------------------
    pool = rows[: args.pool]
    row_f = row_overlap_filter(pool, args.row_jaccard)
    model_f = {t: model_filter(row_f, t) for t in TAUS}
    model_only = {t: model_filter(pool, t) for t in TAUS}
    counts = [dict(method="raw pool", tau=np.nan, retained=len(pool)),
              dict(method=f"row-overlap (J<={args.row_jaccard})", tau=np.nan, retained=len(row_f))]
    for t in TAUS:
        counts.append(dict(method="row-overlap + model-based", tau=t, retained=len(model_f[t])))
    for t in TAUS:
        counts.append(dict(method="model-based only", tau=t, retained=len(model_only[t])))
    cdf = pd.DataFrame(counts)
    cdf.to_csv(f"{out}/retained_counts.csv", index=False)
    div = [diversity(pool[: args.q], N, "raw top-q"), diversity(row_f[: args.q], N, "row-overlap top-q")]
    div += [diversity(model_f[t][: args.q], N, f"row+model tau={t} top-q") for t in TAUS]
    pd.DataFrame(div).to_csv(f"{out}/diversity_quality_metrics.csv", index=False)

    # --- enrichment (CI, replication, confounding, cohort consistency) -------
    rng = np.random.default_rng(args.seed)
    cohorts = df["patient_cohort"].fillna("NA").values

    def enrich(r):
        bp = r["best_pair"]
        lo, hi = bootstrap_ci(Y, r["mask"], bp, args.n_boot, rng)
        r["delta_ci_low"], r["delta_ci_high"] = lo, hi
        r["split_half_rep"] = split_half_rate(Y, r["mask"], bp, r["delta"], 50, rng)
        r["direction"] = ("reversed" if np.sign(r["rho_sg"]) != np.sign(r["rho_dataset"])
                          else "stronger" if abs(r["rho_sg"]) > abs(r["rho_dataset"]) else "weaker")
        for col, tag in (("diagnosis_label", "diag"), ("sample_origin", "origin"), ("patient_cohort", "cohort")):
            s = df.loc[r["mask"], col].astype("string")
            r[f"{tag}_purity"] = s.value_counts(normalize=True).max() if len(s) else np.nan
            r[f"{tag}_top"] = s.value_counts().index[0] if len(s) else ""
        for c in sorted(set(cohorts)):
            cm = cohorts == c
            if (r["mask"] & cm).sum() >= 15 and (~r["mask"] & cm).sum() >= 15:
                r[f"delta_cohort_{c}"] = (spearman_vec(Y[r["mask"] & cm])[bp] - spearman_vec(Y[cm])[bp])
            else:
                r[f"delta_cohort_{c}"] = np.nan
        return r

    def to_df(lst, tag):
        out_rows = []
        for k, r in enumerate(lst):
            d = {kk: vv for kk, vv in r.items() if kk not in ("mask", "rho", "z", "best_pair")}
            d = {"rank": k + 1, **d}
            for p, pn in enumerate(PAIR_NAMES):
                d[f"rho[{pn}]"] = r["rho"][p]
                d[f"z[{pn}]"] = r["z"][p]
            out_rows.append(d)
        df_ = pd.DataFrame(out_rows)
        df_.to_csv(f"{out}/{tag}.csv", index=False)
        return df_

    # raw/row lists are saved lightweight; enriched lists only for top-q of each tau
    to_df(pool, "1_raw_pool_top")
    to_df(row_f, "2_after_row_overlap_filter")
    final = {}
    for t in TAUS:
        lst = [enrich(r) if k < args.q else r for k, r in enumerate(model_f[t])]
        final[t] = lst
        to_df(lst, f"3_after_model_filter_tau_{t:.2f}")
    main = final[args.tau_main][: args.q]
    fdf = to_df(main, "FINAL_subgroups_main_tau")

    # --- plots -----------------------------------------------------------------
    if len(null):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(null, bins=25, color="grey", alpha=.8)
        ax.axvline(best, color="red", label=f"observed best = {best:.2f}")
        ax.axvline(np.percentile(null, 95), color="k", ls="--", label="95th pct of null")
        ax.set_xlabel("max quality phi (permuted data)"); ax.legend(); ax.set_title(f"{name}: permutation null")
        fig.savefig(f"{out}/fig_permutation_null.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    if len(main):
        fig, ax = plt.subplots(figsize=(9, 0.38 * len(main) + 1.5))
        cmap = plt.get_cmap("tab10")
        ax.barh(range(len(main)), [r["score"] for r in main][::-1],
                color=[cmap(r["best_pair"]) for r in main][::-1])
        ax.set_yticks(range(len(main)))
        ax.set_yticklabels([(r["desc"][:60] + "…") if len(r["desc"]) > 60 else r["desc"] for r in main][::-1], fontsize=7)
        for p in range(6):
            ax.barh([], [], color=cmap(p), label=PAIR_NAMES[p])
        ax.legend(fontsize=7); ax.set_xlabel(f"quality phi ({args.agg})"); ax.set_title(f"{name}: top subgroups")
        if len(null):
            ax.axvline(np.percentile(null, 95), color="k", ls="--")
        fig.savefig(f"{out}/fig_top_subgroups.png", dpi=200, bbox_inches="tight"); plt.close(fig)

        ok = fdf.dropna(subset=["delta_ci_low"])
        if len(ok):
            fig, ax = plt.subplots(figsize=(8, 0.38 * len(ok) + 1.5))
            y = np.arange(len(ok))[::-1]
            ax.errorbar(ok["delta"], y, xerr=[ok["delta"] - ok["delta_ci_low"], ok["delta_ci_high"] - ok["delta"]],
                        fmt="o", color="C0", capsize=3)
            ax.axvline(0, color="k", lw=.8)
            ax.set_yticks(y); ax.set_yticklabels([f"{d[:45]} [{p.split('~')[0][:4]}~{p.split('~')[1][:4]}]"
                                                  for d, p in zip(ok["desc"], ok["pair"])], fontsize=7)
            ax.set_xlabel("Δρ = ρ_subgroup − ρ_dataset (bootstrap 95% CI)")
            fig.savefig(f"{out}/fig_forest_delta_rho.png", dpi=200, bbox_inches="tight"); plt.close(fig)

        r0 = main[0]
        i, j = PAIRS[r0["best_pair"]]
        fig, ax = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
        for a, m, tt in ((ax[0], r0["mask"], "subgroup"), (ax[1], ~r0["mask"], "rest")):
            rx, ry = stats.rankdata(Y[m][:, i]) / m.sum(), stats.rankdata(Y[m][:, j]) / m.sum()
            a.scatter(rx, ry, s=8, alpha=.6)
            a.set_title(f"{tt} (n={m.sum()}, ρ={spearman_vec(Y[m])[r0['best_pair']]:.2f})")
            a.set_xlabel(f"rank {TARGETS[i]}"); a.set_ylabel(f"rank {TARGETS[j]}")
        fig.suptitle(r0["desc"][:90], fontsize=8)
        fig.savefig(f"{out}/fig_rank_scatter_top1.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for meth, sty in (("row-overlap + model-based", "o-"), ("model-based only", "s--")):
        s = cdf[cdf.method == meth]
        ax.plot(s.tau, s.retained, sty, label=meth)
    ax.axhline(len(row_f), color="grey", ls=":", label="after row-overlap")
    ax.set_xlabel("tau"); ax.set_ylabel("retained subgroups"); ax.legend(); ax.set_title(f"{name}: redundancy filtering")
    fig.savefig(f"{out}/fig_retained_vs_tau.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ns = np.array([r["n"] for r in rows]); sc = np.array([r["score"] for r in rows])
    ax.scatter(ns, sc, s=3, alpha=.3, color="grey")
    ax.scatter([r["n"] for r in main], [r["score"] for r in main], s=18, color="red", label="final")
    ax.set_xlabel("subgroup size"); ax.set_ylabel("quality phi"); ax.legend(); ax.set_title(f"{name}: size vs quality")
    fig.savefig(f"{out}/fig_size_vs_quality.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    summary = {"experiment": name, "N": N, "n_conditions": len(conds), "n_evaluated": M_eval,
               "best_phi": float(best), "null_p95": float(np.percentile(null, 95)) if len(null) else None,
               "n_final_perm_p_lt_0.05": int((fdf["p_perm_FWER"] < 0.05).sum()) if len(fdf) else 0,
               "n_final_qBH_lt_0.05": int((fdf["q_BH"] < 0.05).sum()) if len(fdf) else 0,
               "runtime_s": round(time.time() - t0, 1)}
    json.dump(summary, open(f"{out}/summary.json", "w"), indent=2)
    print("   ", summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="emm_results")
    ap.add_argument("--experiments", nargs="+", default=list(experiments()))
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--width", type=int, default=40)
    ap.add_argument("--min-size", type=int, default=30)
    ap.add_argument("--q", type=int, default=20)
    ap.add_argument("--agg", choices=list(AGG), default="max")
    ap.add_argument("--n-bins", type=int, default=8)
    ap.add_argument("--pool", type=int, default=300)
    ap.add_argument("--row-jaccard", type=float, default=0.5)
    ap.add_argument("--tau-main", type=float, default=0.10, choices=TAUS)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--drop-descriptors", nargs="*", default=[], help="e.g. plasma_CA19_9")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    df = load_data(args.data)
    json.dump(vars(args), open(f"{args.out}/run_config.json", "w"), indent=2)
    print(f"Loaded {len(df)} patients; diagnosis counts: {df.diagnosis_label.value_counts().to_dict()}")
    print("Missing values:", df[["age", "sex", "stage", "benign_sample_diagnosis", "plasma_CA19_9"]].isna().sum().to_dict())
    baseline_tables(df, args.out)
    specs = experiments()
    sums = [run_experiment(e, specs[e], df, args) for e in args.experiments]
    pd.DataFrame(sums).to_csv(f"{args.out}/all_experiments_summary.csv", index=False)


if __name__ == "__main__":
    main()
