"""Supplementary: D_KL x P/Delta_gap with ALL 150 stars (OSARG P1 from OSARGs.dat). Does NOT modify twpe_analysis.ipynb.
Same functions, sample and nulls as dkl_tests_mira_srv.py, which used only the 100 Mira+SRV.

Original docstring of the base script:

(1) Mann-Whitney between the three classes (N=50 each), Holm- and Bonferroni-adjusted.
(2) Spearman D_KL x P/Delta_gap (Mira+SRV, N=100), plus partial Spearman controlling for
    amplitude A1, SNR and N_obs.
(3) Nulls on the REAL OGLE timestamps of each Mira/SRV star:
      B) periodic synthetic series (sinusoid at the catalogue P1 with the fitted amplitude,
         Gaussian noise equal to the fit residual) with RANDOM PHASE in each realisation;
      C) the real magnitudes shuffled over the real timestamps (destroys temporal order,
         keeps cadence and marginal distribution).
Official functions (Cell 1: permutation_entropy, twpe, kl_sym; m=4, w1) are read from the
notebook with ast. The sample draw reproduces Cell 2A (seed 42).

Run from the repo root with the conda python (needs scipy):
    /opt/anaconda3/bin/python supplementary/dkl_vs_period_all150.py
Outputs in supplementary/results/: dkl_vs_period_all150.csv, dkl_vs_period_all150.png, dkl_vs_period_all150_log.txt
"""
import ast, itertools, json, math, os, random, sys
from collections import Counter
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.getcwd()
DATA = os.path.join(ROOT, "data", "observed", "ogle3_lpv")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"); os.makedirs(OUT, exist_ok=True)
M, N_PER, SEED, GAP_FIXED, GAP_THR, N_REAL = 4, 50, 42, 185.0, 60.0, 30

log = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); log.append(s)

# ── official functions (Cell 1) ─────────────────────────────────────────────
nb = json.load(open(os.path.join(ROOT, "twpe_analysis.ipynb")))
ns = {"np": np, "itertools": itertools, "math": math, "Counter": Counter}
for node in ast.parse("".join(nb["cells"][1]["source"])).body:
    t = ast.dump(node)
    if isinstance(node, (ast.FunctionDef, ast.Import, ast.ImportFrom, ast.Assign)) \
            and "scipy" not in t and "sympy" not in t:
        try: exec(compile(ast.Module([node], []), "cell1", "exec"), ns)
        except Exception: pass
perm_entropy, twpe, kl_sym = ns["permutation_entropy"], ns["twpe"], ns["kl_sym"]
import warnings; warnings.filterwarnings("ignore")

def dkl(t, x):
    _, pd = perm_entropy(x, m=M, normalized=True)
    _, pw, _ = twpe(x, t, m=M, w_func="w1", normalized=True)
    return kl_sym(pd, pw) * 1e3

# ── Cell 2A: pool, sample, periods ──────────────────────────────────────────
stars = []
for line in open(os.path.join(DATA, "ident.dat")):
    p = line.split()
    if len(p) < 4 or p[3] not in ("Mira", "SRV", "OSARG"): continue
    stars.append({"id": p[0], "type": p[3]})
assert len(stars) == 91689
periods = {}
for fn in ("Miras.dat", "SRVs.dat", "OSARGs.dat"):
    for line in open(os.path.join(DATA, fn)):
        p = line.split()
        if len(p) >= 5:
            try: periods[p[0]] = {"P1": float(p[3]), "A1": float(p[4])}
            except ValueError: pass
assert len(periods) == 12795 + 79200
random.seed(SEED)
sample = (random.sample([s for s in stars if s["type"] == "Mira"], N_PER)
          + random.sample([s for s in stars if s["type"] == "SRV"], N_PER)
          + random.sample([s for s in stars if s["type"] == "OSARG"], N_PER))

def load(sid):
    a = np.loadtxt(os.path.join(DATA, "phot_I", sid + ".dat"), usecols=(0, 1, 2))
    a = a[np.argsort(a[:, 0])]
    return a[:, 0], a[:, 1]

rows = []
for s in sample:
    t, x = load(s["id"]); d = np.diff(t)
    r = dict(id=s["id"], cls=s["type"], n=len(t), dkl=dkl(t, x),
             gap_star=float(np.median(d[d > GAP_THR])) if (d > GAP_THR).any() else np.nan)
    pr = periods.get(s["id"])
    r["P1"] = pr["P1"] if pr else np.nan; r["A1"] = pr["A1"] if pr else np.nan
    rows.append(r)
D = {c: np.array([r["dkl"] for r in rows if r["cls"] == c]) for c in ("Mira", "SRV", "OSARG")}
_, p_chk = stats.mannwhitneyu(D["Mira"], D["OSARG"], alternative="greater")
assert abs(p_chk - 2.25e-8) < 0.01e-8, p_chk
say(f"check OK: pool 91689, 150 D_KL values reproduced (Mira>OSARG one-sided p = {p_chk:.2e})")


def holm(ps):
    ps = np.asarray(ps); o = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for rank, i in enumerate(o):
        run = max(run, (len(ps) - rank) * ps[i]); adj[i] = min(1, run)
    return adj
def partial_spearman(x, y, Z):
    rk = lambda v: stats.rankdata(v)
    X = np.column_stack([np.ones(len(x))] + [np.asarray(z, float) if z.dtype == bool or set(np.unique(z)) <= {0, 1} else rk(z) for z in Z])
    res = lambda v: rk(v) - X @ np.linalg.lstsq(X, rk(v), rcond=None)[0]
    rx, ry = res(x), res(y); r = float(np.corrcoef(rx, ry)[0, 1]); dof = len(x) - 2 - (X.shape[1] - 1)
    tt = r * math.sqrt(dof / (1 - r * r)); return r, 2 * stats.t.sf(abs(tt), dof)

R = [r for r in rows if not np.isnan(r["P1"])]
say(f"N = {len(R)} (" + ", ".join(f"{c} {sum(r['cls']==c for r in R)}" for c in ('Mira','SRV','OSARG')) + ") with catalogue P1")
dk = np.array([r["dkl"] for r in R]); P = np.array([r["P1"] for r in R]); A = np.array([r["A1"] for r in R])
N = np.array([r["n"] for r in R], float); gs = np.array([r["gap_star"] for r in R]); cls = np.array([r["cls"] for r in R])
dM = (cls == "Mira").astype(float); dS = (cls == "SRV").astype(float)      # class dummies (OSARG = reference)
def fit_sine(t, x, P1):
    w = 2 * np.pi / P1; Xd = np.column_stack([np.ones_like(t), np.sin(w * t), np.cos(w * t)])
    c, *_ = np.linalg.lstsq(Xd, x, rcond=None); res = x - Xd @ c
    return c[0], float(np.hypot(c[1], c[2])), float(res.std())
LC = {}; nullB, nullC, amp_fit, sig_res = [], [], [], []
for i, r in enumerate(R):
    t, x = load(r["id"]); c0, amp, sres = fit_sine(t, x, r["P1"]); rng = np.random.default_rng(SEED + i)
    b = [dkl(t, c0 + amp * np.sin(2 * np.pi * t / r["P1"] + rng.uniform(0, 2 * np.pi)) + rng.normal(0, sres, len(t))) for _ in range(N_REAL)]
    c = [dkl(t, rng.permutation(x)) for _ in range(N_REAL)]
    nullB.append(np.median(b)); nullC.append(np.median(c)); amp_fit.append(amp); sig_res.append(sres)
nullB, nullC, amp_fit, sig_res = map(np.array, (nullB, nullC, amp_fit, sig_res)); snr = amp_fit / sig_res

lp = np.log(P / GAP_FIXED); lnabs = np.abs(lp)
say("\n== (A) Class-level medians ==")
for c in ("Mira", "SRV", "OSARG"):
    m = cls == c
    say(f"{c:6s} median P1 = {np.median(P[m]):6.1f} d (P1/185 = {np.median(P[m])/GAP_FIXED:.2f}) | median D_KL real {np.median(dk[m]):6.1f} | null B {np.median(nullB[m]):6.1f} | null C {np.median(nullC[m]):6.1f} | median A1 {np.median(A[m]):.3f} mag | median SNR {np.median(snr[m]):.2f}")

say("\n== (B) Pooled Spearman, 150 stars (uncorrected p) ==")
fam = {}
def sp(x, y, label, key=None):
    rho, p = stats.spearmanr(x, y); say(f"{label:66s} rho = {rho:+.3f}  p = {p:.3g}")
    if key: fam[key] = p
    return rho, p
sp(dk, P / GAP_FIXED, "D_KL vs P1/185 (same ranks as vs P1)", "pooled P")
sp(dk, lnabs, "D_KL vs |ln(P1/185)|", "pooled |ln|")
sp(dk, P / gs, "D_KL vs P1/gap_star (median of gaps>60 d, per star)")
sp(dk, A, "D_KL vs A1 (control)"); sp(dk, N, "D_KL vs N_obs (control)"); sp(dk, snr, "D_KL vs SNR of the P1 sinusoid (control)")

say("\n== (C) Within each class ==")
for c in ("Mira", "SRV", "OSARG"):
    m = cls == c
    sp(dk[m], P[m] / GAP_FIXED, f"{c}: D_KL vs P1/185"); sp(dk[m], lnabs[m], f"{c}: D_KL vs |ln(P1/185)|")
    sp(dk[m], A[m], f"{c}: D_KL vs A1"); sp(dk[m], snr[m], f"{c}: D_KL vs SNR")

say("\n== (D) Partial Spearman of D_KL with P1 (t-test approximation, uncorrected p) ==")
r1, p1_ = partial_spearman(dk, P, [dM, dS]); say(f"| class (OSARG = reference)                 rho = {r1:+.3f}  p = {p1_:.3g}"); fam["partial |class"] = p1_
r2, p2_ = partial_spearman(dk, P, [dM, dS, A, N, snr]); say(f"| class, A1, N_obs, SNR                     rho = {r2:+.3f}  p = {p2_:.3g}"); fam["partial |class,A,N,SNR"] = p2_
r3, p3_ = partial_spearman(dk, P, [A, N, snr]); say(f"| A1, N_obs, SNR (no class)                 rho = {r3:+.3f}  p = {p3_:.3g}")
r4, p4_ = partial_spearman(dk, lnabs, [dM, dS, A, N, snr]); say(f"|ln(P1/185)| | class, A1, N_obs, SNR         rho = {r4:+.3f}  p = {p4_:.3g}"); fam["partial |ln| |class,A,N,SNR"] = p4_
ks = list(fam); hp = holm([fam[k] for k in ks])
say("Holm over the 5 pre-specified tests: " + "; ".join(f"{k}: p {fam[k]:.3g} -> {h:.3g}" for k, h in zip(ks, hp)))

say("\n== (E) Nulls on the real timestamps (median of %d realisations per star) ==" % N_REAL)
for lab, nl in (("B periodic+random phase", nullB), ("C shuffled values", nullC)):
    say(f"null {lab}: per-star Spearman with real D_KL: all {stats.spearmanr(dk, nl)[0]:+.2f} | " +
        " | ".join(f"{c} {stats.spearmanr(dk[cls==c], nl[cls==c])[0]:+.2f}" for c in ("Mira", "SRV", "OSARG")))
    pairs = [("Mira", "SRV"), ("SRV", "OSARG"), ("Mira", "OSARG")]
    ps = [stats.mannwhitneyu(nl[cls == a], nl[cls == b], alternative="greater")[1] for a, b in pairs]
    say("   one-sided MW (a>b) on the null: " + "; ".join(f"{a}>{b} p={p:.2g}" for (a, b), p in zip(pairs, ps)))
ps = [stats.mannwhitneyu(dk[cls == a], dk[cls == b], alternative="greater")[1] for a, b in [("Mira", "SRV"), ("SRV", "OSARG"), ("Mira", "OSARG")]]
say("   one-sided MW on the real D_KL (same subset): " + "; ".join(f"p={p:.2g}" for p in ps))
say(f"null B: D_KL vs P1/185 pooled rho = {stats.spearmanr(nullB, P/GAP_FIXED)[0]:+.3f}; null C: {stats.spearmanr(nullC, P/GAP_FIXED)[0]:+.3f}")
for c in ("Mira", "SRV", "OSARG"):
    m = cls == c; say(f"   {c:6s} real/null-B median ratio = {np.median(dk[m])/np.median(nullB[m]):.2f}")

with open(os.path.join(OUT, "dkl_vs_period_all150.csv"), "w") as f:
    f.write("id,class,P1_d,A1,N_obs,gap_star_d,DKL_real,DKL_nullB_median,DKL_nullC_median,fit_amp,fit_resid_std\n")
    for r, b, c, a, s_ in zip(R, nullB, nullC, amp_fit, sig_res):
        f.write(f"{r['id']},{r['cls']},{r['P1']},{r['A1']},{r['n']},{r['gap_star']:.1f},{r['dkl']:.1f},{b:.1f},{c:.1f},{a:.4f},{s_:.4f}\n")
fig, ax = plt.subplots(1, 3, figsize=(13, 4)); col = {"Mira": "C3", "SRV": "C2", "OSARG": "C0"}
for c in col:
    m = cls == c
    ax[0].scatter(P[m] / GAP_FIXED, dk[m], s=14, c=col[c], label=c); ax[1].scatter(dk[m], nullB[m], s=14, c=col[c], label=c); ax[2].scatter(dk[m], nullC[m], s=14, c=col[c], label=c)
ax[0].set(xscale="log", xlabel="P1 / 185 d", ylabel=r"$D_{KL}\times10^3$ (real)"); ax[0].legend(fontsize=8)
for a_, nm in ((ax[1], "null B: periodic, random phase"), (ax[2], "null C: shuffled values")):
    a_.plot([0, 800], [0, 800], "k:", lw=0.8); a_.set(xlabel=r"$D_{KL}$ real", ylabel=nm)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "dkl_vs_period_all150.png"), dpi=150)
import scipy
say(f"\nnumpy {np.__version__}, scipy {scipy.__version__}, python {sys.version.split()[0]}, seed {SEED}, m={M}, N_REAL={N_REAL}")
open(os.path.join(OUT, "dkl_vs_period_all150_log.txt"), "w").write("\n".join(log) + "\n")
