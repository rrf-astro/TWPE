"""Supplementary: catalogue primary period P1 for the 150 sampled OGLE-III stars (Table 2 statistic).
Reads Miras.dat, SRVs.dat and OSARGs.dat (OGLE-III LPV LMC; columns: ID I_mean I_max P1 A1 P2 A2 P3 A3).
Sample draw = notebook Cell 2A (seed 42). Does NOT modify twpe_analysis.ipynb.
Statistic as in the Table 2 caption: median +- IQR/1.35; P1/Delta_gap with Delta_gap = 185 d.
Run from the repo root:  python3 -I supplementary/table2_catalogue_p1.py
Outputs (in supplementary/results/): table2_catalogue_p1.csv, table2_catalogue_p1_log.txt"""
import os, random, sys
import numpy as np
ROOT = os.getcwd(); DATA = os.path.join(ROOT, "data", "observed", "ogle3_lpv"); OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"); os.makedirs(OUT, exist_ok=True)
GAP = 185.0; log = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); log.append(s)
stars = [(l.split()[0], l.split()[3]) for l in open(DATA + "/ident.dat")
         if len(l.split()) >= 4 and l.split()[3] in ("Mira", "SRV", "OSARG")]
assert len(stars) == 91689
def read(fn):
    d = {}
    for l in open(os.path.join(DATA, fn)):
        p = l.split()
        if len(p) >= 5:
            try: d[p[0]] = (float(p[3]), float(p[4]))
            except ValueError: pass
    return d
cat = {"Mira": read("Miras.dat"), "SRV": read("SRVs.dat"), "OSARG": read("OSARGs.dat")}
say({k: len(v) for k, v in cat.items()}, "entries read from the catalogue files")
random.seed(42)
sample = {c: random.sample([s for s in stars if s[1] == c], 50) for c in ("Mira", "SRV", "OSARG")}
def robust(v): q = np.percentile(v, [25, 75]); return np.median(v), (q[1] - q[0]) / 1.35
rows = []
say("\n== Table 2 statistic, catalogue P1, 50 stars per class (seed 42) ==")
say("class   n_with_P1   median P1 [d]  IQR/1.35   mean   std    P1/185 (median)   median A1 [mag]")
for c in ("Mira", "SRV", "OSARG"):
    P, A = [], []
    for sid, _ in sample[c]:
        if sid in cat[c]: P.append(cat[c][sid][0]); A.append(cat[c][sid][1]); rows.append((sid, c, cat[c][sid][0], cat[c][sid][1]))
        else: rows.append((sid, c, float("nan"), float("nan")))
    P = np.array(P); m, s = robust(P)
    say(f"{c:6s} {len(P):4d}/50   {m:8.1f}   {s:8.1f}  {P.mean():7.1f} {P.std(ddof=1):6.1f}   {m/GAP:8.2f}         {np.median(A):.3f}")
    q = np.percentile(P, [10, 25, 75, 90]).round(0).tolist()
    say(f"       P1 quantiles 10/25/75/90 %: {q}; min {P.min():.1f}, max {P.max():.1f}")
O = np.array([r[2] for r in rows if r[1] == "OSARG"])
say(f"\nOSARG: {np.sum(O < 100)}/50 with P1 < 100 d; {np.sum(O >= 300)}/50 with P1 >= 300 d")
with open(os.path.join(OUT, "table2_catalogue_p1.csv"), "w") as f:
    f.write("id,class,P1_catalogue_d,A1_mag\n")
    for r in rows: f.write(f"{r[0]},{r[1]},{r[2]},{r[3]}\n")
say(f"\nnumpy {np.__version__}, python {sys.version.split()[0]}")
open(os.path.join(OUT, "table2_catalogue_p1_log.txt"), "w").write("\n".join(log) + "\n")
