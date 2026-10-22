#!/usr/bin/env python3
"""
Site-stratified EMM (extension of emm_corr_rcs.py; keep both files in the same folder).

Idea (borrowed from EMM-RCS, where T measurement occasions are combined, Eq. 4.1):
treat each collection SITE as an "occasion". For a subgroup, the correlation in each site is compared with the
SAME-SITE reference (so site can no longer confound the result). Per site s and pair p:
    z_sp = (atanh(rho_SG,s,p) - atanh(rho_ref,s,p)) / se(n_s)          (signed)
Sites are combined with Stouffer's rule  Z_p = sum_s z_sp / sqrt(S)  -> rewards deviations that point the
SAME direction in every site.  Quality  phi = max_p |Z_p|.   With one site this is a plain single-site EMM.
Permutation null: target rows are shuffled WITHIN each site (keeps the site structure intact).

--model raw      4 targets, 6 Spearman pairs
--model partial  3 biomarker pairs, partial Spearman controlling for creatinine (dilution control)
--model ratio    3 biomarker/creatinine ratios, 3 Spearman pairs (note: shared denominator induces correlation)

Examples
  python emm_site_stratified.py --data "Debernardi et al 2020 data.csv" --sites BPTB --name E8_BPTB
  python emm_site_stratified.py --data "Debernardi et al 2020 data.csv" --sites LIV  --name E9_LIV
  python emm_site_stratified.py --data "Debernardi et al 2020 data.csv" --sites BPTB LIV --name E10_BPTB_LIV
  python emm_site_stratified.py --data "Debernardi et al 2020 data.csv" --sites BPTB LIV --model partial --name E11_BPTB_LIV_partial
"""
import argparse, itertools, json, os, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import emm_corr_rcs as E
from emm_corr_rcs import fisher, se_fisher, TAUS

BIO = ["LYVE1", "REG1B", "TFF1"]


def pair_names(mode):
    if mode == "raw":
        c = ["creatinine"] + BIO
        return [f"{c[i]}~{c[j]}" for i, j in itertools.combinations(range(4), 2)]
    if mode == "partial":
        return [f"{BIO[i]}~{BIO[j]}|creat" for i, j in itertools.combinations(range(3), 2)]
    return [f"{BIO[i]}/cr~{BIO[j]}/cr" for i, j in itertools.combinations(range(3), 2)]


def model_rho(Y, mode):
    R = stats.rankdata(Y, axis=0)
    with np.errstate(all="ignore"):
        C = np.nan_to_num(np.corrcoef(R, rowvar=False))
    if mode == "partial":
        out = []
        for i, j in ((1, 2), (1, 3), (2, 3)):
            den = np.sqrt(max((1 - C[i, 0] ** 2) * (1 - C[j, 0] ** 2), 1e-9))
            out.append((C[i, j] - C[i, 0] * C[j, 0]) / den)
        return np.array(out)
    return C[np.triu_indices(C.shape[0], 1)]


def evaluate(nm, Y, G):
    Zs, Rs = [], []
    for sm, ref in zip(G["site_masks"], G["refs"]):
        m = nm & sm
        n = int(m.sum())
        if n < G["c_site"] or int(sm.sum()) - n < G["c_site"]:
            return None
        r = model_rho(Y[m], G["mode"])
        Zs.append((fisher(r) - fisher(ref)) / se_fisher(n - G["extra"]))
        Rs.append(r)
    Z = np.array(Zs)
    comb = Z.sum(0) / np.sqrt(len(Zs))
    return np.abs(comb), comb, Z, np.array(Rs)


def search(Y, G, keep=True):
    N = len(Y)
    f = E.AGG[G["agg"]]
    seen, results, best = set(), [], -np.inf
    beam = [((), np.ones(N, bool))]
    for _ in range(G["depth"]):
        cand = []
        for desc, m in beam:
            msum = m.sum()
            for ci, cm in enumerate(G["masks"]):
                if ci in desc:
                    continue
                nm = m & cm
                if nm.sum() == msum:
                    continue
                key = np.packbits(nm).tobytes()
                if key in seen:
                    continue
                seen.add(key)
                ev = evaluate(nm, Y, G)
                if ev is None:
                    continue
                absZ, comb, Z, Rs = ev
                sc = f(absZ)
                best = max(best, sc)
                item = (sc, tuple(sorted(desc + (ci,))), nm, comb, Z, Rs)
                cand.append(item)
                if keep:
                    results.append(item)
        cand.sort(key=lambda x: -x[0])
        beam = [(c[1], c[2]) for c in cand[: G["width"]]]
        if not beam:
            break
    return best, (results if keep else None)


_G = {}


def _init(G):
    _G.update(G)


def _perm_job(seed):
    rng = np.random.default_rng(seed)
    Y = _G["Y"]
    Yp = Y.copy()
    for sm in _G["site_masks"]:
        idx = np.where(sm)[0]
        Yp[idx] = Y[rng.permutation(idx)]
    return search(Yp, _G, keep=False)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results_site")
    ap.add_argument("--name", required=True)
    ap.add_argument("--sites", nargs="+", default=["BPTB"])
    ap.add_argument("--model", choices=["raw", "partial", "ratio"], default="raw")
    ap.add_argument("--dx", nargs="+", type=int, default=[1, 2, 3], help="diagnosis codes to keep (1 control,2 benign,3 PDAC)")
    ap.add_argument("--with-diagnosis", action="store_true", help="allow diagnosis as descriptor")
    ap.add_argument("--c-site", type=int, default=25, help="min subgroup AND complement size within EACH site")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--width", type=int, default=40)
    ap.add_argument("--q", type=int, default=20)
    ap.add_argument("--agg", choices=list(E.AGG), default="max")
    ap.add_argument("--n-bins", type=int, default=8)
    ap.add_argument("--pool", type=int, default=300)
    ap.add_argument("--row-jaccard", type=float, default=0.5)
    ap.add_argument("--tau-main", type=float, default=0.10, choices=TAUS)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--drop-descriptors", nargs="*", default=[], help="e.g. plasma_CA19_9")
    a = ap.parse_args()
    out = os.path.join(a.out, a.name)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()

    df = E.load_data(a.data)
    df = df[df.sample_origin.isin(a.sites) & df.diagnosis.isin(a.dx)].reset_index(drop=True)
    df["cr"] = df["creatinine"]
    if a.model == "ratio":
        Y = np.column_stack([df[b] / df["creatinine"] for b in BIO]).astype(float)
    else:
        Y = df[E.TARGETS].values.astype(float)
    N = len(df)
    pn = pair_names(a.model)
    P, S = len(pn), len(a.sites)
    extra = 1 if a.model == "partial" else 0
    site_masks = [(df.sample_origin == s).values for s in a.sites]
    refs = [model_rho(Y[sm], a.model) for sm in site_masks]
    desc = {"age": "numeric", "sex": "nominal", "plasma_CA19_9": "numeric", "stage_ord": "ordinal",
            "benign_sample_diagnosis": "nominal"}
    if a.with_diagnosis:
        desc["diagnosis_label"] = "nominal"
    desc = {k: v for k, v in desc.items() if k not in a.drop_descriptors}
    conds = E.build_conditions(df, desc, a.n_bins)
    G = dict(Y=Y, masks=[c[1] for c in conds], site_masks=site_masks, refs=refs, mode=a.model, extra=extra,
             c_site=a.c_site, depth=a.depth, width=a.width, agg=a.agg)
    print(f"\n=== {a.name}: sites={a.sites} model={a.model} N={N} "
          f"(per site {[int(m.sum()) for m in site_masks]}), {len(conds)} conditions")

    best, res = search(Y, G, keep=True)
    res.sort(key=lambda x: -x[0])
    print(f"   evaluated {len(res)} subgroups, best phi = {best:.2f}")
    if not res:
        print("   no subgroup satisfies the size constraints; lower --c-site")
        return

    rows = []
    for sc, d, m, comb, Z, Rs in res:
        bp = int(np.argmax(np.abs(comb)))
        rows.append(dict(
            desc=" AND ".join(conds[i][0] for i in d), mask=m, n=int(m.sum()), score=float(sc), best_pair=bp,
            pair=pn[bp], comb_z=float(comb[bp]), Z=Z, Rs=Rs, rho=Rs.ravel(),
            consistent=bool(np.all(np.sign(Z[:, bp]) == np.sign(comb[bp])))))

    null = np.array([])
    if a.n_perm:
        seeds = [a.seed * 100003 + i for i in range(a.n_perm)]
        if a.workers > 1:
            with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(G,)) as ex:
                null = np.array(list(ex.map(_perm_job, seeds, chunksize=4)))
        else:
            _init(G)
            null = np.array([_perm_job(s) for s in seeds])
        np.save(f"{out}/permutation_null.npy", null)
        print(f"   permutation null (B={len(null)}): 95th pct = {np.percentile(null, 95):.2f}")
    for r in rows:
        r["p_perm_FWER"] = (1 + (null >= r["score"]).sum()) / (1 + len(null)) if len(null) else np.nan

    pool = rows[: a.pool]
    row_f = E.row_overlap_filter(pool, a.row_jaccard)
    mf = {t: E.model_filter(row_f, t) for t in TAUS}
    mo = {t: E.model_filter(pool, t) for t in TAUS}
    cnt = [dict(method="raw pool", tau=np.nan, retained=len(pool)),
           dict(method=f"row-overlap (J<={a.row_jaccard})", tau=np.nan, retained=len(row_f))]
    cnt += [dict(method="row-overlap + model-based", tau=t, retained=len(mf[t])) for t in TAUS]
    cnt += [dict(method="model-based only", tau=t, retained=len(mo[t])) for t in TAUS]
    cdf = pd.DataFrame(cnt)
    cdf.to_csv(f"{out}/retained_counts.csv", index=False)
    div = [E.diversity(pool[: a.q], N, "raw top-q"), E.diversity(row_f[: a.q], N, "row-overlap top-q")]
    div += [E.diversity(mf[t][: a.q], N, f"row+model tau={t} top-q") for t in TAUS]
    pd.DataFrame(div).to_csv(f"{out}/diversity_quality_metrics.csv", index=False)

    def purity(r):
        for col, tag in (("diagnosis_label", "diag"), ("sample_origin", "origin"), ("patient_cohort", "cohort")):
            s = df.loc[r["mask"], col].astype("string")
            r[f"{tag}_purity"] = s.value_counts(normalize=True).max()
            r[f"{tag}_top"] = s.value_counts().index[0]
        return r

    def to_df(lst, tag):
        o = []
        for k, r in enumerate(lst):
            d = {"rank": k + 1, "desc": r["desc"], "n": r["n"], "pair": r["pair"], "score_phi": r["score"],
                 "combined_Z_signed": r["comb_z"], "same_direction_all_sites": r["consistent"],
                 "p_perm_FWER": r["p_perm_FWER"]}
            for si, s in enumerate(a.sites):
                m = r["mask"] & site_masks[si]
                bp = r["best_pair"]
                d[f"n[{s}]"] = int(m.sum())
                d[f"rho_sg[{s}]"] = r["Rs"][si, bp]
                d[f"rho_ref[{s}]"] = refs[si][bp]
                d[f"delta[{s}]"] = r["Rs"][si, bp] - refs[si][bp]
                d[f"z[{s}]"] = r["Z"][si, bp]
            for k2 in ("diag_purity", "diag_top", "origin_purity", "origin_top", "cohort_purity", "cohort_top"):
                d[k2] = r.get(k2, np.nan)
            o.append(d)
        x = pd.DataFrame(o)
        x.to_csv(f"{out}/{tag}.csv", index=False)
        return x

    for lst in (pool, row_f):
        for r in lst:
            purity(r)
    to_df(pool, "1_raw_pool_top")
    to_df(row_f, "2_after_row_overlap_filter")
    for t in TAUS:
        for r in mf[t]:
            purity(r)
        to_df(mf[t], f"3_after_model_filter_tau_{t:.2f}")
    main_lst = mf[a.tau_main][: a.q]
    fdf = to_df(main_lst, "FINAL_subgroups_main_tau")

    if len(null):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(null, bins=25, color="grey", alpha=.8)
        ax.axvline(best, color="red", label=f"observed best = {best:.2f}")
        ax.axvline(np.percentile(null, 95), color="k", ls="--", label="95th pct null")
        ax.legend(); ax.set_xlabel("max |Z| (permuted within site)"); ax.set_title(a.name)
        fig.savefig(f"{out}/fig_permutation_null.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    if len(main_lst):
        fig, ax = plt.subplots(figsize=(9, 0.4 * len(main_lst) + 1.5))
        ax.barh(range(len(main_lst)), [r["score"] for r in main_lst][::-1], color="C0")
        ax.set_yticks(range(len(main_lst)))
        ax.set_yticklabels([(r["desc"][:62] + "…") if len(r["desc"]) > 62 else r["desc"] for r in main_lst][::-1], fontsize=7)
        if len(null):
            ax.axvline(np.percentile(null, 95), color="k", ls="--")
        ax.set_xlabel("quality phi"); ax.set_title(f"{a.name}: top subgroups")
        fig.savefig(f"{out}/fig_top_subgroups.png", dpi=200, bbox_inches="tight"); plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 0.4 * len(main_lst) + 1.5))
        for si, s in enumerate(a.sites):
            ax.scatter(fdf[f"delta[{s}]"], np.arange(len(fdf))[::-1] + 0.12 * (si - (S - 1) / 2), label=s, s=24)
        ax.axvline(0, color="k", lw=.8)
        ax.set_yticks(np.arange(len(fdf))[::-1])
        ax.set_yticklabels([f"{d[:40]} [{p[:14]}]" for d, p in zip(fdf.desc, fdf.pair)], fontsize=7)
        ax.set_xlabel("Δρ vs same-site reference"); ax.legend()
        fig.savefig(f"{out}/fig_per_site_delta.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for meth, sty in (("row-overlap + model-based", "o-"), ("model-based only", "s--")):
        s_ = cdf[cdf.method == meth]
        ax.plot(s_.tau, s_.retained, sty, label=meth)
    ax.set_xlabel("tau"); ax.set_ylabel("retained"); ax.legend()
    fig.savefig(f"{out}/fig_retained_vs_tau.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    summ = dict(name=a.name, sites=a.sites, model=a.model, N=N, n_conditions=len(conds), n_evaluated=len(res),
                best_phi=float(best), null_p95=float(np.percentile(null, 95)) if len(null) else None,
                n_final_perm_p_lt_0_05=int((fdf.p_perm_FWER < 0.05).sum()), runtime_s=round(time.time() - t0, 1))
    json.dump(summ, open(f"{out}/summary.json", "w"), indent=2)
    print("   ", summ)
    pd.set_option("display.width", 250, "display.max_colwidth", 70)
    print(fdf[["rank", "desc", "n", "pair", "score_phi", "same_direction_all_sites", "p_perm_FWER"]].head(8).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
