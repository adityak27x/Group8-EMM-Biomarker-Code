#!/usr/bin/env python3
"""
emm_heavy.py : high-power EMM for the Debernardi biomarker correlations.
Needs emm_corr_rcs.py and emm_site_stratified.py in the same folder.

What is different from emm_corr_rcs.py
 1. Hybrid search: EXHAUSTIVE over all conjunctions of up to --exh-depth conditions (counts are pruned with
    matrix products so this is fast), then beam extension to --max-depth with a wide beam.
 2. Optional validation: re-runs the observed search with exhaustive depth 3 and checks the hybrid found the same top subgroups.
 3. Stratified permutation null: target rows are shuffled WITHIN diagnosis x site cells, so diagnosis and site
    proxies cannot count as discoveries; only structure BEYOND those factors is tested.
 4. Power simulation: subgroups with a known change in correlation are planted into the real data and the whole
    pipeline (search + critical value from the null) is run on them. The output is the detection rate by subgroup size
    and effect size, i.e. the smallest effect this analysis could have found.
 5. Models: --model raw (6 Spearman pairs) or partial (3 biomarker pairs controlling for creatinine).

Presets:  quick (test, minutes)  standard  heavy (hours)  extreme (long)
Example:
  python emm_heavy.py --data "Debernardi et al 2020 data.csv" --preset quick  --out heavy_quick
  python emm_heavy.py --data "Debernardi et al 2020 data.csv" --preset heavy  --out heavy_run --workers 6 --drop-descriptors plasma_CA19_9
"""
import argparse, itertools, json, os, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import emm_corr_rcs as E
from emm_site_stratified import model_rho, pair_names
from emm_corr_rcs import fisher, se_fisher, TAUS

PRESETS = {
    "quick": dict(n_perm=20, n_bins=8, exh_depth=2, max_depth=3, width=40, power_reps=2, pool=200),
    "standard": dict(n_perm=500, n_bins=10, exh_depth=2, max_depth=3, width=80, power_reps=6, pool=300),
    "heavy": dict(n_perm=1000, n_bins=12, exh_depth=2, max_depth=4, width=150, power_reps=10, pool=500),
    "extreme": dict(n_perm=2000, n_bins=14, exh_depth=3, max_depth=4, width=250, power_reps=15, pool=500),
}
POWER_SIZES = [30, 50, 80, 120]
POWER_DELTAS = [0.2, 0.3, 0.4, 0.5]


# --------------------------------------------------------------------------- search
def hybrid_search(Y, ctx, keep=True):
    N, M, Mi, ms = ctx["N"], ctx["M"], ctx["Mi"], ctx["min_size"]
    mode, extra = ctx["mode"], ctx["extra"]
    ref = model_rho(Y, mode)
    seen, res, best, levels = set(), [], -np.inf, {}

    def record(nm, desc):
        nonlocal best
        key = np.packbits(nm).tobytes()
        if key in seen:
            return False
        seen.add(key)
        n = int(nm.sum())
        r = model_rho(Y[nm], mode)
        if ctx.get("reference", "dataset") == "complement":
            rc = model_rho(Y[~nm], mode)
            se = np.sqrt(se_fisher(n - extra) ** 2 + se_fisher(N - n - extra) ** 2)
            z = np.abs(fisher(r) - fisher(rc)) / se
        else:
            z = np.abs(fisher(r) - fisher(ref)) / se_fisher(n - extra)
        sc = float(z.max())
        best = max(best, sc)
        if keep:
            res.append((sc, desc, nm, r, z))
        levels.setdefault(len(desc), []).append((sc, desc, nm))
        return True

    def viable(m):
        cnt = Mi @ m.astype(np.int32)
        return np.where((cnt >= ms) & (cnt < int(m.sum())) & (N - cnt >= ms))[0]

    def dfs(m, desc, start):
        if len(desc) >= ctx["exh_depth"]:
            return
        cnt = Mi[start:] @ m.astype(np.int32)
        ok = np.where((cnt >= ms) & (cnt < int(m.sum())) & (N - cnt >= ms))[0]
        for k in ok:
            kk = start + int(k)
            nm = m & M[kk]
            if record(nm, desc + (kk,)):
                dfs(nm, desc + (kk,), kk + 1)

    dfs(np.ones(N, bool), (), 0)
    d = ctx["exh_depth"]
    cur = sorted(levels.get(d, []), key=lambda x: -x[0])[: ctx["width"]]
    while d < ctx["max_depth"] and cur:
        d += 1
        for sc, desc, m in cur:
            for k in viable(m):
                k = int(k)
                if k not in desc:
                    record(m & M[k], tuple(sorted(desc + (k,))))
        cur = sorted(levels.get(d, []), key=lambda x: -x[0])[: ctx["width"]]
    return best, (res if keep else None)


_G = {}


def _init(ctx):
    _G.update(ctx)


def permute_within(Y, groups, rng):
    Yp = Y.copy()
    for idx in groups:
        Yp[idx] = Y[rng.permutation(idx)]
    return Yp


def _perm_job(seed):
    rng = np.random.default_rng(seed)
    return hybrid_search(permute_within(_G["Y"], _G["groups"], rng), _G, keep=False)[0]


def run_pool(fn, args, workers, ctx, chunk=2):
    if workers > 1:
        with ProcessPoolExecutor(workers, initializer=_init, initargs=(ctx,)) as ex:
            return list(ex.map(fn, args, chunksize=chunk))
    _init(ctx)
    return [fn(a) for a in args]


# --------------------------------------------------------------------------- power simulation
def plant(Y, mask, pair, target, rng):
    i, j = E.PAIRS[pair] if Y.shape[1] == 4 and _G["mode"] == "raw" else _G["pairs"][pair]
    idx = np.where(mask)[0]
    n = len(idx)
    ui = (stats.rankdata(Y[idx, i]) - 0.5) / n
    zi = stats.norm.ppf(ui)
    rp = 2 * np.sin(np.pi * target / 6)
    zj = rp * zi + np.sqrt(max(1 - rp ** 2, 1e-6)) * rng.standard_normal(n)
    new = np.empty(n)
    new[np.argsort(zj)] = np.sort(Y[idx, j])
    Y2 = Y.copy()
    Y2[idx, j] = new
    return Y2


def find_rule(ctx, n_t, rng):
    M, Mi, N, ms = ctx["M"], ctx["Mi"], ctx["N"], ctx["min_size"]
    sizes = M.sum(1)
    for tol in (0.2, 0.4):
        lo, hi = n_t * (1 - tol), n_t * (1 + tol)
        singles = [(k,) for k in np.where((sizes >= lo) & (sizes <= hi) & (N - sizes >= ms))[0]]
        G = np.triu(Mi @ Mi.T, 1)
        pr = np.argwhere((G >= lo) & (G <= hi) & (N - G >= ms))
        cands = singles + [tuple(p) for p in pr]
        if cands:
            c = cands[rng.integers(len(cands))]
            m = np.ones(N, bool)
            for k in c:
                m &= M[k]
            return m
    return None


def _power_job(a):
    n_t, delta, seed, crit = a
    rng = np.random.default_rng(seed)
    ctx = _G
    mask = find_rule(ctx, n_t, rng)
    if mask is None:
        return dict(n_target=n_t, delta=delta, detected=np.nan, realized=np.nan, n=np.nan)
    ref = model_rho(ctx["Y"], ctx["mode"])
    P = len(ref)
    pair = int(rng.integers(P))
    s = rng.choice([-1, 1])
    target = ref[pair] + s * delta
    if abs(target) > 0.95:
        target = ref[pair] - s * delta
    Y2 = plant(ctx["Y"], mask, pair, target, rng)
    best, res = hybrid_search(Y2, ctx, keep=True)
    res.sort(key=lambda x: -x[0])
    det = any(sc >= crit and E.jaccard(m, mask) >= 0.5 for sc, _, m, _, _ in res[: ctx["pool"]])
    refv = model_rho(Y2[~mask], ctx["mode"])[pair] if ctx.get("reference") == "complement" else model_rho(Y2, ctx["mode"])[pair]
    real = model_rho(Y2[mask], ctx["mode"])[pair] - refv
    return dict(n_target=n_t, delta=delta, detected=float(det), realized=abs(real), n=int(mask.sum()))


# --------------------------------------------------------------------------- one experiment
def run_experiment(name, spec, df_all, a):
    out = os.path.join(a.out, name)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    df = df_all if spec["subset"] is None else df_all[spec["subset"](df_all)].reset_index(drop=True)
    desc = {k: v for k, v in spec["desc"].items() if k not in a.drop_descriptors}
    conds = E.build_conditions(df, desc, a.n_bins)
    K, N = len(conds), len(df)
    M = np.array([c[1] for c in conds])
    Y = df[E.TARGETS].values.astype(float)
    mode = a.model
    extra = 1 if mode == "partial" else 0
    pn = pair_names(mode)
    pairs = list(itertools.combinations(range(4), 2)) if mode == "raw" else list(itertools.combinations(range(1, 4), 2))
    cols = [c for c in ["diagnosis", "sample_origin"] if df[c].nunique() > 1] if a.perm_strata == ["auto"] else (
        [] if a.perm_strata == ["none"] else a.perm_strata)
    groups = list(df.groupby(cols).indices.values()) if cols else [np.arange(N)]
    ctx = dict(Y=Y, M=M, Mi=M.astype(np.int32), N=N, min_size=a.min_size, exh_depth=a.exh_depth, max_depth=a.max_depth,
               width=a.width, mode=mode, extra=extra, groups=groups, pool=a.pool, pairs=pairs, reference=a.reference)
    print(f"\n=== {name}: N={N}, {K} conditions, model={mode}, reference={a.reference}, permutation strata={cols or 'none (global)'}")

    ts = time.time()
    best, res = hybrid_search(Y, ctx, keep=True)
    t_search = time.time() - ts
    res.sort(key=lambda x: -x[0])
    print(f"   hybrid search: {len(res)} distinct subgroups evaluated in {t_search:.1f}s, best phi = {best:.2f}")
    print(f"   estimated permutation time: ~{t_search * a.n_perm / max(a.workers, 1) / 60:.1f} min "
          f"({a.n_perm} perms, {a.workers} workers)")

    # ---- validation against exhaustive depth 3
    val = {}
    if a.validate and a.exh_depth < 3 and a.max_depth >= 3:
        ctx3 = dict(ctx, exh_depth=3, max_depth=3)
        tv = time.time()
        b3, r3 = hybrid_search(Y, ctx3, keep=True)
        r3.sort(key=lambda x: -x[0])
        keys = {np.packbits(r[2]).tobytes() for r in res}
        found = sum(np.packbits(r[2]).tobytes() in keys for r in r3[:20])
        val = dict(hybrid_best=float(best), exhaustive_depth3_best=float(b3), exhaustive_n_evaluated=len(r3),
                   top20_of_exhaustive_found_by_hybrid=int(found), seconds=round(time.time() - tv, 1))
        print(f"   validation: exhaustive depth-3 best = {b3:.2f} (hybrid {best:.2f}); top-20 recovered: {found}/20")
        json.dump(val, open(f"{out}/validation_exhaustive3.json", "w"), indent=2)

    # ---- permutation null
    seeds = [a.seed * 100003 + i for i in range(a.n_perm)]
    null = np.array(run_pool(_perm_job, seeds, a.workers, ctx, 2)) if a.n_perm else np.array([])
    crit = float(np.percentile(null, 95)) if len(null) else np.nan
    if len(null):
        np.save(f"{out}/permutation_null.npy", null)
        print(f"   null (B={len(null)}): median {np.median(null):.2f}, 95th {crit:.2f}, 99th {np.percentile(null, 99):.2f}")

    rows = []
    for sc, d, m, r, z in res:
        bp = int(np.argmax(z))
        rows.append(dict(desc=" AND ".join(conds[i][0] for i in d), mask=m, n=int(m.sum()), score=sc, best_pair=bp,
                         pair=pn[bp], rho=r, z=z, rho_sg=r[bp],
                         rho_ref=(model_rho(Y[~m], mode)[bp] if a.reference == "complement" else model_rho(Y, mode)[bp]),
                         p_perm=(1 + (null >= sc).sum()) / (1 + len(null)) if len(null) else np.nan))
    for r in rows:
        r["delta"] = r["rho_sg"] - r["rho_ref"]
    pool = rows[: a.pool]
    row_f = E.row_overlap_filter(pool, a.row_jaccard)
    mf = {t: E.model_filter(row_f, t) for t in TAUS}
    mo = {t: E.model_filter(pool, t) for t in TAUS}
    cdf = pd.DataFrame([dict(method="raw pool", tau=np.nan, retained=len(pool)),
                        dict(method=f"row-overlap (J<={a.row_jaccard})", tau=np.nan, retained=len(row_f))] +
                       [dict(method="row-overlap + model-based", tau=t, retained=len(mf[t])) for t in TAUS] +
                       [dict(method="model-based only", tau=t, retained=len(mo[t])) for t in TAUS])
    cdf.to_csv(f"{out}/retained_counts.csv", index=False)
    div = [E.diversity(pool[: a.q], N, "raw top-q"), E.diversity(row_f[: a.q], N, "row-overlap top-q")]
    div += [E.diversity(mf[t][: a.q], N, f"row+model tau={t} top-q") for t in TAUS]
    pd.DataFrame(div).to_csv(f"{out}/diversity_quality_metrics.csv", index=False)

    strata_cols = [c for c in ("diagnosis_label", "sample_origin", "patient_cohort") if df[c].nunique() > 1]

    def enrich(r):
        bp = r["best_pair"]
        for col in strata_cols:
            s = df.loc[r["mask"], col].astype("string")
            r[f"purity[{col}]"] = s.value_counts(normalize=True).max()
            r[f"top[{col}]"] = s.value_counts().index[0]
            for v in df[col].dropna().unique():
                cm = (df[col] == v).values
                a_, b_ = r["mask"] & cm, ~r["mask"] & cm
                key = f"delta_within[{col}={v}]"
                r[key] = (model_rho(Y[a_], mode)[bp] - model_rho(Y[cm], mode)[bp]) if a_.sum() >= 15 and b_.sum() >= 15 else np.nan
        return r

    def to_df(lst, tag, rich=False):
        o = []
        for k, r in enumerate(lst):
            if rich:
                enrich(r)
            d = {kk: vv for kk, vv in r.items() if kk not in ("mask", "rho", "z", "best_pair")}
            d = {"rank": k + 1, **d}
            for p, nm in enumerate(pn):
                d[f"rho[{nm}]"], d[f"z[{nm}]"] = r["rho"][p], r["z"][p]
            o.append(d)
        x = pd.DataFrame(o)
        x.to_csv(f"{out}/{tag}.csv", index=False)
        return x

    to_df(pool, "1_raw_pool_top")
    to_df(row_f, "2_after_row_overlap_filter")
    for t in TAUS:
        to_df(mf[t], f"3_after_model_filter_tau_{t:.2f}", rich=(t == a.tau_main))
    fdf = pd.read_csv(f"{out}/3_after_model_filter_tau_{a.tau_main:.2f}.csv").head(a.q)
    fdf.to_csv(f"{out}/FINAL_subgroups_main_tau.csv", index=False)

    # ---- power simulation
    power = pd.DataFrame()
    if a.power_reps > 0 and len(null) and name in a.power_experiments:
        jobs = [(n_t, dl, a.seed * 7919 + 131 * i + 17 * j + rep, crit)
                for i, n_t in enumerate(a.power_sizes) for j, dl in enumerate(a.power_deltas) for rep in range(a.power_reps)]
        print(f"   power simulation: {len(jobs)} planted-subgroup searches (critical value {crit:.2f}) ...")
        pr = pd.DataFrame(run_pool(_power_job, jobs, a.workers, ctx, 1))
        power = pr.groupby(["n_target", "delta"]).agg(detection_rate=("detected", "mean"), reps=("detected", "count"),
                                                      mean_realized_delta=("realized", "mean"),
                                                      mean_actual_n=("n", "mean")).reset_index()
        power.to_csv(f"{out}/power_simulation.csv", index=False)
        pv = power.pivot(index="n_target", columns="delta", values="detection_rate")
        print("   detection rate (rows: planted subgroup size, columns: planted change in rho):")
        print(pv.round(2).to_string())
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(pv.values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(pv.columns))); ax.set_xticklabels(pv.columns)
        ax.set_yticks(range(len(pv.index))); ax.set_yticklabels(pv.index)
        for i in range(pv.shape[0]):
            for j in range(pv.shape[1]):
                ax.text(j, i, f"{pv.values[i, j]:.2f}", ha="center", va="center", color="w" if pv.values[i, j] < .6 else "k")
        ax.set_xlabel("planted change in ρ"); ax.set_ylabel("planted subgroup size")
        ax.set_title(f"{name}: detection rate"); fig.colorbar(im)
        fig.savefig(f"{out}/fig_power.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # ---- plots
    if len(null):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(null, bins=30, color="grey", alpha=.8)
        ax.axvline(best, color="red", label=f"observed best {best:.2f}")
        ax.axvline(crit, color="k", ls="--", label="null 95th pct")
        ax.legend(); ax.set_xlabel("max φ (stratified permutation)"); ax.set_title(name)
        fig.savefig(f"{out}/fig_permutation_null.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    if len(fdf):
        fig, ax = plt.subplots(figsize=(9, 0.4 * len(fdf) + 1.5))
        ax.barh(range(len(fdf)), fdf.score.values[::-1])
        ax.set_yticks(range(len(fdf)))
        ax.set_yticklabels([(d[:62] + "…") if len(d) > 62 else d for d in fdf.desc][::-1], fontsize=7)
        if len(null):
            ax.axvline(crit, color="k", ls="--")
        ax.set_xlabel("quality φ"); ax.set_title(f"{name}: top subgroups")
        fig.savefig(f"{out}/fig_top_subgroups.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    summ = dict(experiment=name, N=N, conditions=K, evaluated=len(res), best_phi=float(best), null_median=float(np.median(null)) if len(null) else None,
                null_p95=crit if len(null) else None, null_p99=float(np.percentile(null, 99)) if len(null) else None,
                min_perm_p=float(min(r["p_perm"] for r in rows)) if len(null) else None,
                n_perm_sig_in_pool=int(sum(r["p_perm"] < 0.05 for r in rows)) if len(null) else None,
                validation=val, runtime_min=round((time.time() - t0) / 60, 1))
    json.dump(summ, open(f"{out}/summary.json", "w"), indent=2, default=float)
    pd.set_option("display.width", 250, "display.max_colwidth", 70)
    print(fdf[["rank", "desc", "n", "pair", "rho_sg", "rho_ref", "delta", "score", "p_perm"]].head(10).round(3).to_string(index=False))
    print(f"   summary: best φ {best:.2f} vs null 95th {crit:.2f}; subgroups with perm p<0.05: {summ['n_perm_sig_in_pool']}; "
          f"min p = {summ['min_perm_p']}; runtime {summ['runtime_min']} min")
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="heavy_results")
    ap.add_argument("--preset", choices=list(PRESETS), default="standard")
    ap.add_argument("--experiments", nargs="+", default=["E1_clinical", "E4_PDAC_only", "E5_PDAC_with_origin", "E6_controls_only", "E7_benign_only"])
    ap.add_argument("--model", choices=["raw", "partial"], default="raw")
    ap.add_argument("--reference", choices=["dataset", "complement"], default="dataset",
                    help="compare each subgroup with the whole dataset (default) or with its complement (two-sample SE)")
    ap.add_argument("--min-size", type=int, default=30)
    ap.add_argument("--q", type=int, default=20)
    ap.add_argument("--row-jaccard", type=float, default=0.5)
    ap.add_argument("--tau-main", type=float, default=0.10, choices=TAUS)
    ap.add_argument("--perm-strata", nargs="*", default=["auto"], help="auto | none | column names (e.g. diagnosis sample_origin)")
    ap.add_argument("--only-site", nargs="*", default=[], help="restrict the whole analysis to these sites, e.g. BPTB")
    ap.add_argument("--drop-descriptors", nargs="*", default=[])
    ap.add_argument("--power-experiments", nargs="*", default=["E1_clinical", "E4_PDAC_only"])
    ap.add_argument("--power-sizes", nargs="+", type=int, default=POWER_SIZES)
    ap.add_argument("--power-deltas", nargs="+", type=float, default=POWER_DELTAS + [0.6])
    ap.add_argument("--no-validate", dest="validate", action="store_false")
    ap.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 1, 1))
    ap.add_argument("--seed", type=int, default=1)
    for k in ("n_perm", "n_bins", "exh_depth", "max_depth", "width", "power_reps", "pool"):
        ap.add_argument(f"--{k.replace('_', '-')}", type=int, default=None)
    a = ap.parse_args()
    for k, v in PRESETS[a.preset].items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(f"{a.out}/run_config.json", "w"), indent=2)
    print(f"preset={a.preset} workers={a.workers} perms={a.n_perm} bins={a.n_bins} exhaustive depth={a.exh_depth} "
          f"max depth={a.max_depth} beam width={a.width} power reps={a.power_reps}")
    df = E.load_data(a.data)
    if a.only_site:
        df = df[df.sample_origin.isin(a.only_site)].reset_index(drop=True)
        print(f"restricted to sites {a.only_site}: {len(df)} patients")
    specs = E.experiments()
    sums = [run_experiment(e, specs[e], df, a) for e in a.experiments]
    pd.DataFrame([{k: v for k, v in s.items() if k != "validation"} for s in sums]).to_csv(f"{a.out}/HEAVY_SUMMARY.csv", index=False)
    print("\nDONE. Summary table:")
    print(pd.DataFrame([{k: v for k, v in s.items() if k != "validation"} for s in sums]).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
