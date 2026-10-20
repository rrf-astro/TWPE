"""Supplementary inspection of the two OSARGs in the D_KL upper tail (~770e-3).

Does NOT modify twpe_analysis.ipynb. It re-reads the official function
definitions of notebook Cell 1 (via ast, without executing the rest), repeats
the Cell 2A sample draw (seed 42) and checks the numbers quoted in the notebook.

Run from the repository root:
    python3 -I supplementary/osarg_tail_inspection.py
Outputs (in supplementary/results/): osarg_tail_table.csv,
osarg_tail.png, osarg_tail_log.txt
"""
import ast, itertools, json, math, os, random, sys
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.getcwd()
NB = os.path.join(ROOT, "twpe_analysis.ipynb")
DATA = os.path.join(ROOT, "data", "observed", "ogle3_lpv")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"); os.makedirs(OUT, exist_ok=True)
M, N_PER, SEED, GAP = 4, 50, 42, 60.0
EXPECTED = {"OGLE-LMC-LPV-09900": 771.6, "OGLE-LMC-LPV-33726": 769.7}

log = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); log.append(s)

# ── official functions from Cell 1 (no scipy/sympy statements executed) ─────
nb = json.load(open(NB))
tree = ast.parse("".join(nb["cells"][1]["source"]))
ns = {"np": np, "itertools": itertools, "math": math, "Counter": Counter}
for node in tree.body:
    txt = ast.dump(node)
    if isinstance(node, (ast.FunctionDef, ast.Import, ast.ImportFrom, ast.Assign)) \
            and "scipy" not in txt and "sympy" not in txt:
        try:
            exec(compile(ast.Module([node], []), "cell1", "exec"), ns)
        except Exception:
            pass
perm_entropy, twpe, kl_sym = ns["permutation_entropy"], ns["twpe"], ns["kl_sym"]

def dkl(t, x):
    _, pd = perm_entropy(x, m=M, normalized=True)
    _, pw, _ = twpe(x, t, m=M, w_func="w1", normalized=True)
    return kl_sym(pd, pw) * 1e3

# ── Cell 2A: parse_ident and the sample draw, identical ─────────────────────
stars = []
for line in open(os.path.join(DATA, "ident.dat")):
    p = line.split()
    if len(p) < 4 or p[3] not in ("Mira", "SRV", "OSARG"):
        continue
    stars.append({"id": p[0], "type": p[3]})
ct = Counter(s["type"] for s in stars)
say(f"pool: {len(stars)} stars  Mira {ct['Mira']}  SRV {ct['SRV']}  OSARG {ct['OSARG']}")
assert len(stars) == 91689
random.seed(SEED)
sample = (random.sample([s for s in stars if s["type"] == "Mira"], N_PER)
          + random.sample([s for s in stars if s["type"] == "SRV"], N_PER)
          + random.sample([s for s in stars if s["type"] == "OSARG"], N_PER))
osargs = [s for s in sample if s["type"] == "OSARG"]
assert len(osargs) == 50

def load(sid):
    a = np.loadtxt(os.path.join(DATA, "phot_I", sid + ".dat"), usecols=(0, 1, 2))
    a = a[np.argsort(a[:, 0])]
    return a[:, 0], a[:, 1]

# ── table ───────────────────────────────────────────────────────────────────
rows = []
for s in osargs:
    t, x = load(s["id"]); d = np.diff(t)
    rows.append(dict(id=s["id"], n=len(t), span=t[-1] - t[0], max_gap=d.max(),
                     n_gaps=int((d > GAP).sum()), std=x.std(), dkl=dkl(t, x)))
rows.sort(key=lambda r: -r["dkl"])
for sid, v in EXPECTED.items():
    r = next(r for r in rows if r["id"] == sid)
    assert abs(r["dkl"] - v) < 0.1, (sid, r["dkl"], v)
say(f"check OK: D_KL = {rows[0]['dkl']:.1f}, {rows[1]['dkl']:.1f}; "
    f"mean over 50 = {np.mean([r['dkl'] for r in rows]):.1f}")
with open(os.path.join(OUT, "osarg_tail_table.csv"), "w") as f:
    f.write("id,N_obs,span_d,max_gap_d,n_gaps_gt60d,std_mag,DKL_x1e3\n")
    for r in rows:
        f.write(f"{r['id']},{r['n']},{r['span']:.0f},{r['max_gap']:.0f},{r['n_gaps']},"
                f"{r['std']:.4f},{r['dkl']:.1f}\n")

def ranks(v):
    return np.argsort(np.argsort(v)).astype(float)
def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    return float(np.corrcoef(ra, rb)[0, 1])
def spearman_perm_p(a, b, n=20000, seed=0):
    rng = np.random.default_rng(seed); r0 = abs(spearman(a, b)); b = np.array(b)
    c = sum(abs(spearman(a, rng.permutation(b))) >= r0 for _ in range(n))
    return (c + 1) / (n + 1)

dk = np.array([r["dkl"] for r in rows])
for key in ("max_gap", "n", "n_gaps", "std"):
    v = np.array([r[key] for r in rows], float)
    say(f"Spearman D_KL vs {key}: rho = {spearman(dk, v):+.3f}  "
        f"(permutation p = {spearman_perm_p(dk, v):.4f}, 50 OSARGs)")
med = {k: np.median([r[k] for r in rows]) for k in ("n", "max_gap", "std")}
say("median over 50 OSARGs:", {k: round(float(v), 4) for k, v in med.items()})
for sid in EXPECTED:
    r = next(r for r in rows if r["id"] == sid)
    say(f"{sid}: N={r['n']} (rank from smallest {1+sum(q['n']<r['n'] for q in rows)}), "
        f"max gap={r['max_gap']:.0f} d (rank from largest {1+sum(q['max_gap']>r['max_gap'] for q in rows)}), "
        f"gaps>{GAP:.0f}d={r['n_gaps']}, std={r['std']:.4f} mag")

# ── Lomb–Scargle (plain numpy) and spectral window ─────────────────────────
def lomb_scargle(t, y, freqs):
    y = y - y.mean(); P = np.empty(len(freqs))
    for i, fr in enumerate(freqs):
        w = 2 * np.pi * fr
        tau = np.arctan2(np.sum(np.sin(2 * w * t)), np.sum(np.cos(2 * w * t))) / (2 * w)
        c, s = np.cos(w * (t - tau)), np.sin(w * (t - tau))
        P[i] = 0.5 * ((y @ c) ** 2 / (c @ c) + (y @ s) ** 2 / (s @ s))
    return P / (0.5 * len(t) * y.var() if y.var() > 0 else 1)
def window(t, freqs):
    return np.array([abs(np.exp(2j * np.pi * fr * t).sum()) ** 2 for fr in freqs]) / len(t) ** 2

freqs = np.linspace(1 / 1500, 1 / 20, 4000)
def top_peaks(f, P, k=3, sep=3e-4):
    out = []
    for i in np.argsort(P)[::-1]:
        if all(abs(f[i] - f[j]) > sep for j in out): out.append(i)
        if len(out) == k: break
    return out

# ── control: truncate the largest gap ──────────────────────────────────────
def truncated(t, x):
    d = np.diff(t); k = int(np.argmax(d))
    parts = [(t[:k + 1], x[:k + 1]), (t[k + 1:], x[k + 1:])]
    return parts, d[k]

fig, ax = plt.subplots(2, 3, figsize=(13, 6.5))
for j, sid in enumerate(EXPECTED):
    t, x = load(sid); d = np.diff(t)
    say(f"\n== {sid}: N={len(t)}, span={t[-1]-t[0]:.0f} d, D_KL(full) = {dkl(t, x):.1f}")
    parts, g = truncated(t, x)
    for name, (tt, xx) in zip(("before gap", "after gap"), parts):
        if len(xx) > M + 5:
            say(f"   control, {name} the largest gap ({g:.0f} d): N={len(xx)}, "
                f"max gap={np.diff(tt).max():.0f} d, D_KL = {dkl(tt, xx):.1f}")
    # control 2: matched-N random subsets of the same star are not used; instead
    # the median D_KL of the other 48 OSARGs is the reference
    others = [r["dkl"] for r in rows if r["id"] not in EXPECTED]
    say(f"   reference: other 48 OSARGs, median D_KL = {np.median(others):.1f}, "
        f"max = {np.max(others):.1f}")
    fs = freqs[1 / freqs < 300]       # periods < 300 d: avoids annual-alias region
    P = lomb_scargle(t, x, fs); W = window(t, fs)
    pk = top_peaks(fs, P)
    say("   LS peaks (P<300 d): " + "; ".join(f"{1/fs[i]:.0f} d (power {P[i]:.1f})" for i in pk))
    Pf = lomb_scargle(t, x, freqs); pkf = top_peaks(freqs, Pf)
    say("   LS peaks (20-1500 d): " + "; ".join(
        f"{1/freqs[i]:.0f} d (power {Pf[i]:.1f}, window {window(t, freqs[i:i+1])[0]:.2f})" for i in pkf))
    ax[0, j].plot(t, x, ".", ms=3); ax[0, j].invert_yaxis()
    for k in np.where(d > GAP)[0]:
        ax[0, j].axvspan(t[k], t[k + 1], color="0.85", zorder=0)
    ax[0, j].set(title=f"{sid[-9:]}  N={len(t)}", xlabel="HJD − 2450000" if t.mean() > 2.4e6 else "HJD",
                 ylabel="I (mag)")
    ax[1, j].plot(1 / freqs, Pf, lw=0.8, label="Lomb–Scargle")
    ax[1, j].plot(1 / freqs, window(t, freqs) * Pf.max(), lw=0.8, alpha=0.6, label="window (scaled)")
    ax[1, j].set(xscale="log", xlabel="period (d)", ylabel="normalised power"); ax[1, j].legend(fontsize=7)
# third column: D_KL vs largest gap for all 50 OSARGs
for a in (ax[0, 2], ax[1, 2]): a.axis("off")
a = fig.add_subplot(1, 3, 3)
a.scatter([r["max_gap"] for r in rows], dk, s=14)
for r in rows[:2]: a.annotate(r["id"][-5:], (r["max_gap"], r["dkl"]), fontsize=7)
a.set(xlabel="largest gap (d)", ylabel=r"$D_{KL}\times10^3$", title="50 OSARGs")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "osarg_tail.png"), dpi=150)

say(f"\nnumpy {np.__version__}, python {sys.version.split()[0]}, seed {SEED}, m={M}, gap threshold {GAP} d")
open(os.path.join(OUT, "osarg_tail_log.txt"), "w").write("\n".join(log) + "\n")
