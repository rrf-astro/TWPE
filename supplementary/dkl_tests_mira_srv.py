"""Supplementary tests of D_KL on the OGLE-III sample (does NOT modify twpe_analysis.ipynb).

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
    /opt/anaconda3/bin/python supplementary/dkl_tests_mira_srv.py
Outputs in supplementary/results/: dkl_tests_mira_srv_per_star.csv, dkl_tests_mira_srv.png,
dkl_tests_mira_srv_log.txt
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
for fn in ("Miras.dat", "SRVs.dat"):
    for line in open(os.path.join(DATA, fn)):
        p = line.split()
        if len(p) >= 5:
            try: periods[p[0]] = {"P1": float(p[3]), "A1": float(p[4])}
            except ValueError: pass
assert len(periods) == 12795
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

# ═══ (1) Mann-Whitney, three pairs ═════════════════════════════════════════
say("\n== (1) Mann-Whitney on D_KL, N=50 per class (pre-specified direction: Mira > SRV > OSARG) ==")
pairs = [("Mira", "SRV"), ("SRV", "OSARG"), ("Mira", "OSARG")]
raw2, raw1, cles = [], [], []
for a, b in pairs:
    u2, p2 = stats.mannwhitneyu(D[a], D[b], alternative="two-sided")
    u1, p1 = stats.mannwhitneyu(D[a], D[b], alternative="greater")
    raw2.append(p2); raw1.append(p1); cles.append(u1 / (len(D[a]) * len(D[b])))
def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for rank, i in enumerate(o):
        run = max(run, (len(ps) - rank) * ps[i]); adj[i] = min(1, run)
    return adj
h2, h1 = holm(np.array(raw2)), holm(np.array(raw1))
for k, (a, b) in enumerate(pairs):
    say(f"{a:5s} vs {b:5s}: median {np.median(D[a]):6.1f} vs {np.median(D[b]):6.1f} | "
        f"P(X_{a}>X_{b}) = {cles[k]:.2f} | two-sided p = {raw2[k]:.2e} (Holm {h2[k]:.2e}, Bonferroni {min(1,3*raw2[k]):.2e})"
        f" | one-sided p = {raw1[k]:.2e} (Holm {h1[k]:.2e})")

# ═══ (2) D_KL x P/Delta_gap ════════════════════════════════════════════════
say("\n== (2) Spearman, Mira+SRV (those with catalogue P1) ==")
R = [r for r in rows if r["cls"] in ("Mira", "SRV") and not np.isnan(r["P1"])]
say(f"N = {len(R)} (Mira {sum(r['cls']=='Mira' for r in R)}, SRV {sum(r['cls']=='SRV' for r in R)})")
dk = np.array([r["dkl"] for r in R]); P = np.array([r["P1"] for r in R]); A = np.array([r["A1"] for r in R])
N = np.array([r["n"] for r in R], float); gs = np.array([r["gap_star"] for r in R])
cls = np.array([r["cls"] for r in R])
def sp(x, y, label):
    ok = ~(np.isnan(x) | np.isnan(y)); rho, p = stats.spearmanr(x[ok], y[ok])
    say(f"{label:58s} rho = {rho:+.3f}  p = {p:.3g}  (n={ok.sum()})"); return p
fam = []
fam.append(sp(dk, P / GAP_FIXED, "D_KL vs P/185 d  (identical ranks to D_KL vs P)"))
fam.append(sp(dk, P / gs, "D_KL vs P/gap_star (median of gaps>60 d, per star)"))
fam.append(sp(dk, np.abs(np.log(P / GAP_FIXED)), "D_KL vs |ln(P/185)|  (peak near P=gap would give rho<0)"))
fam.append(sp(dk, A, "D_KL vs amplitude A1 (control variable)"))
fam.append(sp(dk, N, "D_KL vs N_obs (control variable)"))
say("Holm over these 5 tests:", ", ".join(f"{x:.3g}" for x in holm(np.array(fam))))
for c in ("Mira", "SRV"):
    m = cls == c
    sp(dk[m], P[m] / GAP_FIXED, f"within {c}: D_KL vs P/185")
    sp(dk[m], np.abs(np.log(P[m] / GAP_FIXED))[...], f"within {c}: D_KL vs |ln(P/185)|")

def partial_spearman(x, y, Z):
    rk = lambda v: stats.rankdata(v)
    X = np.column_stack([np.ones(len(x))] + [rk(z) for z in Z])
    res = lambda v: rk(v) - X @ np.linalg.lstsq(X, rk(v), rcond=None)[0]
    rx, ry = res(x), res(y); r = float(np.corrcoef(rx, ry)[0, 1]); dof = len(x) - 2 - len(Z)
    tt = r * math.sqrt(dof / (1 - r * r)); return r, 2 * stats.t.sf(abs(tt), dof)
r_, p_ = partial_spearman(dk, P, [A, N]); say(f"partial Spearman D_KL vs P | A1, N_obs:  rho = {r_:+.3f}  p = {p_:.3g}")

# SNR proxy: amplitude of the sinusoid fitted at P1 over the residual std (computed below)
# ═══ (3) nulls on real timestamps ══════════════════════════════════════════
def fit_sine(t, x, P1):
    w = 2 * np.pi / P1; Xd = np.column_stack([np.ones_like(t), np.sin(w * t), np.cos(w * t)])
    c, *_ = np.linalg.lstsq(Xd, x, rcond=None); res = x - Xd @ c
    return c[0], float(np.hypot(c[1], c[2])), float(res.std())

nullB, nullC, amp_fit, sig_res = [], [], [], []
for i, r in enumerate(R):
    t, x = load(r["id"]); c0, amp, sres = fit_sine(t, x, r["P1"])
    rng = np.random.default_rng(SEED + i)
    b = [dkl(t, c0 + amp * np.sin(2 * np.pi * t / r["P1"] + rng.uniform(0, 2 * np.pi))
             + rng.normal(0, sres, len(t))) for _ in range(N_REAL)]
    c = [dkl(t, rng.permutation(x)) for _ in range(N_REAL)]
    nullB.append(np.median(b)); nullC.append(np.median(c)); amp_fit.append(amp); sig_res.append(sres)
nullB, nullC, amp_fit, sig_res = map(np.array, (nullB, nullC, amp_fit, sig_res))
snr = amp_fit / sig_res
say(f"\n== (3) nulls on the real timestamps (median of {N_REAL} realisations per star) ==")
for c in ("Mira", "SRV"):
    m = cls == c
    say(f"{c:5s} median D_KL: real {np.median(dk[m]):6.1f} | B periodic+random phase {np.median(nullB[m]):6.1f} | "
        f"C shuffled values {np.median(nullC[m]):6.1f}")
    say(f"      real-vs-B  per-star Spearman {stats.spearmanr(dk[m], nullB[m])[0]:+.2f} | "
        f"real-vs-C {stats.spearmanr(dk[m], nullC[m])[0]:+.2f}")
say(f"Mira vs SRV (two-sided MW):  real p = {stats.mannwhitneyu(dk[cls=='Mira'], dk[cls=='SRV'])[1]:.2e} | "
    f"null B p = {stats.mannwhitneyu(nullB[cls=='Mira'], nullB[cls=='SRV'])[1]:.2e} | "
    f"null C p = {stats.mannwhitneyu(nullC[cls=='Mira'], nullC[cls=='SRV'])[1]:.2e}")
say(f"all 100: real-vs-B Spearman {stats.spearmanr(dk, nullB)[0]:+.2f}; real-vs-C {stats.spearmanr(dk, nullC)[0]:+.2f}; "
    f"B-vs-C {stats.spearmanr(nullB, nullC)[0]:+.2f}")
say(f"null B: D_KL vs P/185 rho = {stats.spearmanr(nullB, P/GAP_FIXED)[0]:+.3f} (p = {stats.spearmanr(nullB, P/GAP_FIXED)[1]:.3g});  "
    f"null C: rho = {stats.spearmanr(nullC, P/GAP_FIXED)[0]:+.3f}")
say(f"fit SNR (amp/residual std): median Mira {np.median(snr[cls=='Mira']):.2f}, SRV {np.median(snr[cls=='SRV']):.2f}")
sp(dk, snr, "D_KL vs SNR of the P1 sinusoid fit (control variable)")
r_, p_ = partial_spearman(dk, P, [A, N, snr]); say(f"partial Spearman D_KL vs P | A1, N_obs, SNR:  rho = {r_:+.3f}  p = {p_:.3g}")

with open(os.path.join(OUT, "dkl_tests_mira_srv_per_star.csv"), "w") as f:
    f.write("id,class,P1_d,A1,N_obs,gap_star_d,DKL_real,DKL_nullB_median,DKL_nullC_median,fit_amp,fit_resid_std\n")
    for r, b, c, a, s_ in zip(R, nullB, nullC, amp_fit, sig_res):
        f.write(f"{r['id']},{r['cls']},{r['P1']},{r['A1']},{r['n']},{r['gap_star']:.1f},{r['dkl']:.1f},{b:.1f},{c:.1f},{a:.4f},{s_:.4f}\n")

fig, ax = plt.subplots(1, 3, figsize=(13, 4)); col = {"Mira": "C3", "SRV": "C2"}
for c in col:
    m = cls == c
    ax[0].scatter(P[m] / GAP_FIXED, dk[m], s=14, c=col[c], label=c)
    ax[1].scatter(dk[m], nullB[m], s=14, c=col[c], label=c)
    ax[2].scatter(dk[m], nullC[m], s=14, c=col[c], label=c)
ax[0].set(xscale="log", xlabel="P1 / 185 d", ylabel=r"$D_{KL}\times10^3$ (real)"); ax[0].legend(fontsize=8)
for a_, nm in ((ax[1], "null B: periodic, random phase"), (ax[2], "null C: shuffled values")):
    a_.plot([0, 800], [0, 800], "k:", lw=0.8); a_.set(xlabel=r"$D_{KL}$ real", ylabel=nm)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "dkl_tests_mira_srv.png"), dpi=150)
import scipy
say(f"\nnumpy {np.__version__}, scipy {scipy.__version__}, python {sys.version.split()[0]}, seed {SEED}, m={M}, N_REAL={N_REAL}")
open(os.path.join(OUT, "dkl_tests_mira_srv_log.txt"), "w").write("\n".join(log) + "\n")
