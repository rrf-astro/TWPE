"""Supplementary: gap/N dependence of D_KL and largest-gap truncation control, all 150 stars.

Does NOT modify twpe_analysis.ipynb. It re-reads the official function
definitions of notebook Cell 1 (via ast, without executing the rest), repeats
the Cell 2A sample draw (seed 42) and checks the numbers quoted in the notebook.

Run from the repository root:
    python3 -I supplementary/gap_truncation_all150.py
Outputs (in supplementary/results/):
gap_truncation_all150.csv, gap_truncation_all150.png, gap_truncation_all150_log.txt
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
MIN_N = 5 * 24  # 5*m!: shortest truncated piece accepted

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


def load(sid):
    a = np.loadtxt(os.path.join(DATA, "phot_I", sid + ".dat"), usecols=(0, 1, 2))
    a = a[np.argsort(a[:, 0])]
    return a[:, 0], a[:, 1]

def ranks(v): return np.argsort(np.argsort(v)).astype(float)
def spearman(a, b): return float(np.corrcoef(ranks(a), ranks(b))[0, 1])
def perm_p(a, b, n=20000, seed=0):
    rng = np.random.default_rng(seed); r0 = abs(spearman(a, b)); b = np.array(b)
    return (sum(abs(spearman(a, rng.permutation(b))) >= r0 for _ in range(n)) + 1) / (n + 1)

rows = []
for s in sample:
    t, x = load(s["id"]); d = np.diff(t); k = int(np.argmax(d))
    full = dkl(t, x)
    pieces = [(t[:k + 1], x[:k + 1]), (t[k + 1:], x[k + 1:])]
    longer = max(pieces, key=lambda p: len(p[0]))
    # control: keep the longer side of the largest gap (only if it has >= 5*m! points)
    trunc = dkl(*longer) if len(longer[0]) >= MIN_N else np.nan
    rows.append(dict(id=s["id"], cls=s["type"], n=len(t), max_gap=d[k], n_gaps=int((d > GAP).sum()),
                     dkl=full, dkl_trunc=trunc, n_trunc=len(longer[0]), gap_trunc=np.diff(longer[0]).max()))
for sid, v in EXPECTED.items():
    assert abs(next(r for r in rows if r["id"] == sid)["dkl"] - v) < 0.1
say(f"check OK (150 stars; the two OSARG tail values reproduced)")
with open(os.path.join(OUT, "gap_truncation_all150.csv"), "w") as f:
    f.write("id,class,N_obs,max_gap_d,n_gaps_gt60d,DKL_x1e3,N_trunc,max_gap_trunc_d,DKL_trunc_x1e3\n")
    for r in rows:
        f.write(f"{r['id']},{r['cls']},{r['n']},{r['max_gap']:.0f},{r['n_gaps']},{r['dkl']:.1f},{r['n_trunc']},{r['gap_trunc']:.0f},{r['dkl_trunc']:.1f}\n")

say("\n== Spearman, D_KL vs largest gap / N / number of gaps>60 d (permutation p, uncorrected) ==")
for cls in ("Mira", "SRV", "OSARG", "ALL"):
    R = [r for r in rows if cls == "ALL" or r["cls"] == cls]
    dk = np.array([r["dkl"] for r in R])
    out = []
    for key in ("max_gap", "n", "n_gaps"):
        v = np.array([r[key] for r in R], float)
        out.append(f"{key}: rho={spearman(dk, v):+.2f} (p={perm_p(dk, v):.3f})")
    say(f"{cls:6s} n={len(R):3d}  " + "  ".join(out))

say("\n== Control: keep only the longer side of the largest gap ==")
say(f"(accepted if the kept piece has >= {MIN_N} points)")
for cls in ("Mira", "SRV", "OSARG"):
    R = [r for r in rows if r["cls"] == cls and not np.isnan(r["dkl_trunc"])]
    a = np.array([r["dkl"] for r in R]); b = np.array([r["dkl_trunc"] for r in R])
    gcut = np.array([r["max_gap"] - r["gap_trunc"] for r in R])
    say(f"{cls:6s} n={len(R):2d}/50  median D_KL full {np.median(a):6.1f} -> truncated {np.median(b):6.1f}; "
        f"median change {np.median(b - a):+6.1f}; stars with decrease: {(b < a).sum()}/{len(R)}; "
        f"Spearman(full, truncated) = {spearman(a, b):+.2f}")
    drops = [r for r in R if r["max_gap"] > 500]
    for r in drops:
        say(f"        gap {r['max_gap']:.0f} d: {r['id'][-9:]} D_KL {r['dkl']:.1f} -> {r['dkl_trunc']:.1f}")

# extreme-gap subset in all classes
big = [r for r in rows if r["max_gap"] > 500]
say(f"\nstars with largest gap > 500 d: {[(r['id'][-5:], r['cls'], round(r['max_gap']), round(r['dkl'], 1)) for r in big]}")

fig, ax = plt.subplots(1, 3, figsize=(13, 4))
col = {"Mira": "C3", "SRV": "C2", "OSARG": "C0"}
for cls in col:
    R = [r for r in rows if r["cls"] == cls]
    ax[0].scatter([r["max_gap"] for r in R], [r["dkl"] for r in R], s=14, c=col[cls], label=cls)
    ax[1].scatter([r["n"] for r in R], [r["dkl"] for r in R], s=14, c=col[cls], label=cls)
    R2 = [r for r in R if not np.isnan(r["dkl_trunc"])]
    ax[2].scatter([r["dkl"] for r in R2], [r["dkl_trunc"] for r in R2], s=14, c=col[cls], label=cls)
ax[0].set(xlabel="largest gap (d)", ylabel=r"$D_{KL}\times10^3$"); ax[1].set(xlabel="N_obs", xscale="log")
lim = [0, 850]; ax[2].plot(lim, lim, "k:", lw=0.8)
ax[2].set(xlabel=r"$D_{KL}$ full", ylabel=r"$D_{KL}$ longer side of largest gap")
ax[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(OUT, "gap_truncation_all150.png"), dpi=150)
say(f"\nnumpy {np.__version__}, python {sys.version.split()[0]}, seed {SEED}, m={M}, gap threshold {GAP} d")
open(os.path.join(OUT, "gap_truncation_all150_log.txt"), "w").write("\n".join(log) + "\n")
