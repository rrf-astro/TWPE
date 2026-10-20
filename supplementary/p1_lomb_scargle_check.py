"""Supplementary: Lomb-Scargle primary period (P1) of the 150 sampled OGLE-III stars.
Does NOT modify twpe_analysis.ipynb. Sample draw = notebook Cell 2A (seed 42).
(1) Validation: LS P1 vs the catalogue P1 (Miras.dat / SRVs.dat) for the 100 Mira+SRV.
(2) LS P1 of the 50 OSARGs, in the same statistic as Table 2 (median, IQR/1.35).
Run from the repo root:  /opt/anaconda3/bin/python supplementary/p1_lomb_scargle_check.py
Outputs (in supplementary/results/): p1_lomb_scargle_check.csv, p1_lomb_scargle_check_log.txt"""
import os, random, sys
import numpy as np
from scipy.signal import lombscargle
ROOT = os.getcwd(); DATA = os.path.join(ROOT, "data", "observed", "ogle3_lpv")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"); os.makedirs(OUT, exist_ok=True); SEED, N_PER = 42, 50
PMIN, PMAX, OVERS = 3.0, 1500.0, 10
log = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); log.append(s)
stars = [(l.split()[0], l.split()[3]) for l in open(DATA + "/ident.dat")
         if len(l.split()) >= 4 and l.split()[3] in ("Mira", "SRV", "OSARG")]
assert len(stars) == 91689
cat = {}
for fn in ("Miras.dat", "SRVs.dat"):
    for l in open(os.path.join(DATA, fn)):
        p = l.split()
        if len(p) >= 5:
            try: cat[p[0]] = float(p[3])
            except ValueError: pass
random.seed(SEED)
sample = (random.sample([s for s in stars if s[1] == "Mira"], N_PER) + random.sample([s for s in stars if s[1] == "SRV"], N_PER)
          + random.sample([s for s in stars if s[1] == "OSARG"], N_PER))
def ls_period(t, y):
    T = t[-1] - t[0]; df = 1.0 / (OVERS * T)
    f = np.arange(1.0 / PMAX, 1.0 / PMIN, df)
    P = lombscargle(t, y - y.mean(), 2 * np.pi * f, normalize=True)
    i = int(np.argmax(P)); return 1.0 / f[i], float(P[i])
rows = []
for sid, ty in sample:
    a = np.loadtxt(f"{DATA}/phot_I/{sid}.dat", usecols=(0, 1, 2)); a = a[np.argsort(a[:, 0])]
    p1, pw = ls_period(a[:, 0], a[:, 1])
    rows.append(dict(id=sid, cls=ty, ls=p1, pw=pw, cat=cat.get(sid, np.nan)))
def robust(v): v = np.asarray(v); q = np.percentile(v, [25, 75]); return np.median(v), (q[1] - q[0]) / 1.35
say("== (1) Catalogue P1: reproduce Table 2 (median, IQR/1.35) ==")
for c in ("Mira", "SRV"):
    v = [r["cat"] for r in rows if r["cls"] == c and not np.isnan(r["cat"])]
    m, s = robust(v); say(f"{c:5s} n={len(v)}  catalogue P1 = {m:.0f} +- {s:.0f} d   (P1/185 = {m/185:.2f})")
say("\n== (2) Validation of the LS P1 against the catalogue P1 (Mira+SRV) ==")
for c in ("Mira", "SRV"):
    R = [r for r in rows if r["cls"] == c and not np.isnan(r["cat"])]
    ratio = np.array([r["ls"] / r["cat"] for r in R])
    close = np.abs(ratio - 1) < 0.05; half = (np.abs(ratio - 0.5) < 0.03) | (np.abs(ratio - 2) < 0.1)
    say(f"{c:5s} n={len(R)}  |LS/cat-1|<5%: {close.sum()}/{len(R)}  ratio~0.5 or 2: {half.sum()}  "
        f"median LS/cat = {np.median(ratio):.2f}  LS median {np.median([r['ls'] for r in R]):.0f} d  catalogue median {np.median([r['cat'] for r in R]):.0f} d")
say("\n== (3) LS P1 of all classes (same statistic as Table 2) ==")
for c in ("Mira", "SRV", "OSARG"):
    v = [r["ls"] for r in rows if r["cls"] == c]; m, s = robust(v)
    alias = sum(abs(x - 365.25) / 365.25 < 0.05 or abs(x - 182.6) / 182.6 < 0.05 for x in v)
    say(f"{c:5s} n={len(v)}  LS P1 = {m:.0f} +- {s:.0f} d  (P1/185 = {m/185:.2f}); range {min(v):.0f}-{max(v):.0f} d; "
        f"within 5% of 365 d or 182.6 d (alias-prone): {alias}")
O = np.array([r["ls"] for r in rows if r["cls"] == "OSARG"])
say(f"OSARG LS P1 quantiles (10, 25, 50, 75, 90 %): {np.percentile(O,[10,25,50,75,90]).round(0).tolist()}")
say(f"OSARG: {np.sum(O<100)}/50 with P1 < 100 d; {np.sum(O>=300)}/50 with P1 >= 300 d")
with open(os.path.join(OUT, "p1_lomb_scargle_check.csv"), "w") as f:
    f.write("id,class,P1_LS_d,LS_power,P1_catalogue_d\n")
    for r in rows: f.write(f"{r['id']},{r['cls']},{r['ls']:.2f},{r['pw']:.3f},{r['cat']}\n")
import scipy; say(f"\nnumpy {np.__version__}, scipy {scipy.__version__}, python {sys.version.split()[0]}, P range {PMIN}-{PMAX} d, oversampling {OVERS}")
open(os.path.join(OUT, "p1_lomb_scargle_check_log.txt"), "w").write("\n".join(log) + "\n")
