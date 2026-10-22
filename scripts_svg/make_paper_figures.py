"""
Paper figures that are drawn from numbers already produced by the pipeline.

The values below are transcribed from the result files of the runs described in the README:
  fig_sens.png      <- HEAVY_SUMMARY.csv of results/heavy_noCA199 and results/heavy_withCA199
  fig_baseline.png  <- baseline_correlations_by_diagnosis.csv (preliminary run) and
                       A_creatinine_spread_vs_coupling.csv (results/dilution)
  fig_partial.png   <- bptb_robustness.csv (results/dilution)
  fig_power.png     <- power_simulation.csv of results/heavy_power (E1 and E4 -> paper E1 and E2)
  fig_topE1.png     <- FINAL_subgroups_main_tau.csv of results/heavy_noCA199/E1_clinical
The null-distribution figure (Figure 1) is built directly from the saved null arrays by make_null_figure.py.
Usage: python scripts/make_paper_figures.py
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False,
                     "axes.titlesize":10,"axes.titleweight":"bold","figure.dpi":150})
import os
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","figures_svg")+os.sep
os.makedirs(OUT,exist_ok=True)
NAVY="#1F3864"; C1="#2E7D32"; C2="#1976D2"; C3="#C62828"; GREY="#9E9E9E"

# ---------- Fig 1 observed vs null
exps=["E1 all patients","E2 PDAC only","E3 PDAC + site","E4 controls only","E5 benign only"]
noCA=[(5.19,5.93,6.26,5.01),(3.31,4.18,4.56,2.98),(4.01,4.78,5.31,3.59),(2.89,3.80,4.26,3.37),(3.73,4.51,5.05,3.55)]
withCA=[(5.31,6.08,6.71,5.01),(3.76,4.57,5.08,3.60),(4.22,4.98,5.41,3.79),(3.28,4.17,4.70,4.59),(3.97,4.86,5.27,4.59)]
pmin_no=[1.00,0.79,0.87,0.16,0.69]; pmin_with=[1.00,0.67,0.91,0.016,0.11]
fig,axs=plt.subplots(1,2,figsize=(9.2,3.5),sharey=True)
for ax,data,pm,title in ((axs[0],noCA,pmin_no,"Without CA19-9 as descriptor"),(axs[1],withCA,pmin_with,"With CA19-9 as descriptor")):
    for i,(med,p95,p99,obs) in enumerate(data):
        y=len(data)-1-i
        ax.plot([med,p99],[y,y],color=GREY,lw=6,solid_capstyle="butt",alpha=.55)
        ax.plot([p95,p95],[y-.28,y+.28],color="k",lw=1.8)
        ax.scatter(obs,y,marker="D",s=55,color=C3,zorder=3)
        ax.text(max(p99,obs)+.12,y,f"p$_{{min}}$={pm[i]:.2f}" if pm[i]>=0.1 else f"p$_{{min}}$={pm[i]:.3f}*",va="center",fontsize=8)
    ax.set_yticks(range(len(data))); ax.set_yticklabels(exps[::-1]); ax.set_xlabel("quality φ (max over biomarker pairs)")
    ax.set_title(title); ax.set_xlim(2.2,8.2); ax.grid(axis="x",alpha=.25)
from matplotlib.lines import Line2D
h=[Line2D([0],[0],color=GREY,lw=6,alpha=.55,label="null: median to 99th percentile"),
   Line2D([0],[0],color="k",lw=1.8,label="null 95th percentile"),
   Line2D([0],[0],marker="D",color="w",markerfacecolor=C3,markersize=7,label="observed best φ")]
fig.legend(handles=h,loc="lower center",ncol=3,frameon=False,fontsize=8,bbox_to_anchor=(0.5,-0.02))
fig.tight_layout(rect=(0,0.07,1,1)); fig.savefig(OUT+"fig_sens.svg",dpi=300,bbox_inches="tight"); plt.close(fig)

# ---------- Fig 2 baseline
pairs=["creat~LYVE1","creat~REG1B","creat~TFF1","LYVE1~REG1B","LYVE1~TFF1","REG1B~TFF1"]
rho={"Control (n=183)":[.659,.509,.701,.459,.725,.549],"Benign (n=208)":[.412,.306,.565,.586,.626,.620],"PDAC (n=199)":[.342,.188,.422,.514,.572,.655]}
fig,axs=plt.subplots(1,2,figsize=(9.2,3.6),gridspec_kw={"width_ratios":[1.6,1]})
x=np.arange(6); w=.26
for k,((lab,v),c) in enumerate(zip(rho.items(),(C1,C2,C3))):
    axs[0].bar(x+(k-1)*w,v,w,label=lab,color=c)
axs[0].scatter(x,[.386,.296,.493,.648,.736,.682],marker="_",s=380,color="k",lw=2,zorder=3,label="pooled (all 590)")
axs[0].set_xticks(x); axs[0].set_xticklabels(pairs,rotation=30,ha="right"); axs[0].set_ylabel("Spearman ρ"); axs[0].set_ylim(0,.85)
axs[0].set_title("A  Biomarker correlations by diagnosis"); axs[0].legend(fontsize=7.5,frameon=False,ncol=2,loc="upper left")
cells=["BPTB\ncontrol","BPTB\nbenign","BPTB\nPDAC","LIV\nbenign","LIV\nPDAC"]; vals=[.623,.485,.493,.045,.081]; ns=[183,143,83,39,93]
cols=[C1,C2,C3,C2,C3]
b=axs[1].bar(range(5),vals,color=cols,edgecolor=[NAVY]*3+["#E65100"]*2,lw=1.5)
for i,(v,n) in enumerate(zip(vals,ns)): axs[1].text(i,v+.015,f"n={n}",ha="center",fontsize=7.5)
axs[1].set_xticks(range(5)); axs[1].set_xticklabels(cells,fontsize=8); axs[1].set_ylim(0,.75)
axs[1].set_ylabel("mean ρ(creatinine, biomarkers)"); axs[1].set_title("B  Creatinine coupling by site")
fig.tight_layout(); fig.savefig(OUT+"fig_baseline.svg",dpi=300,bbox_inches="tight"); plt.close(fig)

# ---------- Fig 3 BPTB partial rho with CI (single-column size)
def ci(r,n):
    z=np.arctanh(r); se=np.sqrt(1.06/(n-4)); return np.tanh(z-1.96*se),np.tanh(z+1.96*se)
rows=[("BPTB, all",(.192,183),(.558,143),(.468,83)),("Cohort 2 only",(.207,102),(.582,119),(.314,31)),
      ("Cohort 1 only",(.487,81),None,(.566,52)),("Age 45–75",(.195,145),(.593,101),(.506,66)),
      ("Cohort 2, age 45–75",(.214,79),(.629,80),None),("Female",(.195,115),(.587,76),(.545,35)),("Male",(.238,68),(.514,67),(.498,48))]
fig,ax=plt.subplots(figsize=(3.45,3.35))
for i,row in enumerate(rows):
    y=len(rows)-1-i
    for k,(g,c) in enumerate(zip(row[1:],(C1,C2,C3))):
        if g is None: continue
        lo,hi=ci(*g); off=(k-1)*.24
        ax.plot([lo,hi],[y+off]*2,color=c,lw=1.6); ax.scatter(g[0],y+off,color=c,s=14,zorder=3)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1],fontsize=7); ax.axvline(0,color="k",lw=.6)
ax.tick_params(axis="x",labelsize=7)
ax.set_xlabel("partial ρ, LYVE1~REG1B | creatinine\n(95% CI)",fontsize=7); ax.grid(axis="x",alpha=.25)
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([0],[0],color=c,lw=1.6,marker="o",markersize=3,label=l) for c,l in zip((C1,C2,C3),("Control","Benign","PDAC"))],loc="lower center",ncol=3,frameon=False,fontsize=7,bbox_to_anchor=(0.5,-0.01))
fig.tight_layout(rect=(0,0.07,1,1)); fig.savefig(OUT+"fig_partial.svg",dpi=300,bbox_inches="tight"); plt.close(fig)

# ---------- Fig 4 power
d=[0.3,0.4,0.5,0.6]
E1={30:[.02,.04,.14,.18],50:[.30,.32,.34,.48],80:[.54,.52,.86,.94],120:[.64,.84,.94,.98]}
E4={30:[.00,.08,.18,.20],50:[.14,.40,.52,.64],80:[.30,.38,.58,.70],120:[.12,.40,.22,.32]}
fig,axs=plt.subplots(1,2,figsize=(9.2,3.5),sharey=True)
cm=plt.get_cmap("viridis")
for ax,data,t in ((axs[0],E1,"E1: full cohort (N = 590)"),(axs[1],E4,"E2: PDAC only (N = 199)")):
    for j,(n,v) in enumerate(data.items()):
        ax.plot(d,v,marker="o",color=cm(j/3.2),label=f"n = {n}",lw=2)
    ax.axhline(.8,color=C3,ls="--",lw=1); ax.text(.598,.82,"80% power",color=C3,fontsize=8,ha="right")
    ax.set_xlabel("planted change in Spearman ρ"); ax.set_title(t); ax.set_xticks(d); ax.set_ylim(0,1.03); ax.grid(alpha=.25)
axs[0].set_ylabel("detection rate")
h,l=axs[0].get_legend_handles_labels()
fig.legend(h,l,loc="lower center",ncol=4,frameon=False,fontsize=8,bbox_to_anchor=(0.5,-0.02),title="planted subgroup size",title_fontsize=8)
fig.tight_layout(rect=(0,0.1,1,1)); fig.savefig(OUT+"fig_power.svg",dpi=300,bbox_inches="tight"); plt.close(fig)



# ---------- Fig S2: top subgroups of E1 with diagnosis composition
rows=[("R(stage)=0 ∧ R(benign dx)=0",183,5.006,"control"),
      ("age > 67 ∧ stage > IB",89,4.126,"PDAC"),
      ("67 < age ≤ 72 ∧ stage > IB",34,4.080,"PDAC"),
      ("56 < age ≤ 76 ∧ sex ≠ F ∧ stage > IIB",37,3.869,"PDAC"),
      ("age ≤ 58 ∧ sex ≠ F ∧ R(stage)=0 ∧ R(benign dx)=0",44,3.806,"control"),
      ("age ≤ 46 ∧ benign dx ∉ {3 pancreatitis types}",57,3.755,"benign"),
      ("age > 69 ∧ sex ≠ F ∧ stage ≤ III",36,3.752,"PDAC"),
      ("age > 41 ∧ sex ≠ F ∧ R(benign dx)=0",175,3.355,"control + PDAC")]
colmap={"control":C1,"PDAC":C3,"benign":C2,"control + PDAC":GREY}
fig,ax=plt.subplots(figsize=(7.6,3.6))
for i,(lab,n,phi,dx) in enumerate(rows):
    y=len(rows)-1-i
    ax.barh(y,phi,color=colmap[dx],alpha=.9)
    ax.text(phi+0.05,y,f"n={n}",va="center",fontsize=7.5)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1],fontsize=7.5)
ax.axvline(5.93,color="k",ls="--",lw=1.3); ax.text(5.95,len(rows)-0.45,"stratified null 95th (5.93)",fontsize=7,va="top")
ax.axvline(4.73,color=GREY,ls="--",lw=1.3); ax.text(4.71,-0.75,"plain null 95th (4.73)",fontsize=7,ha="right",va="bottom",color="#555")
ax.set_xlim(0,7.2); ax.set_xlabel("quality φ")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=c,label=dx) for dx,c in colmap.items()],frameon=False,fontsize=7.5,loc="lower right",title="patients in subgroup",title_fontsize=7.5,bbox_to_anchor=(1.0,0.12))
fig.tight_layout(); fig.savefig(OUT+"fig_topE1.svg",dpi=300,bbox_inches="tight"); plt.close(fig)
print("figures written to",os.path.abspath(OUT))
