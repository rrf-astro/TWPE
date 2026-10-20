"""Re-execution of notebook Cell 10 (AR(1) rho=0.5, m=3..6, MC n=50 per sigma) WITHOUT
modifying twpe_analysis.ipynb and without writing figures. Cell-10 code is taken verbatim from
the notebook (up to the plotting part); this script only adds the bias at sigma=0.9.
Output: supplementary/results/cell10_bias_log.txt.
Run from the repo root:  python3 -I supplementary/cell10_bias_check.py"""
import ast, io, itertools, json, math, os, sys
class _Tee(io.StringIO):
    def write(self, s): sys.__stdout__.write(s); return super().write(s)
_buf = _Tee(); sys.stdout = _buf
from collections import Counter
import numpy as np
nb = json.load(open("twpe_analysis.ipynb"))
ns = {"np": np, "itertools": itertools, "math": math, "Counter": Counter, "os": os}
for node in ast.parse("".join(nb["cells"][1]["source"])).body:
    t = ast.dump(node)
    if isinstance(node, (ast.FunctionDef, ast.Import, ast.ImportFrom, ast.Assign)) \
            and "scipy" not in t and "sympy" not in t:
        try: exec(compile(ast.Module([node], []), "cell1", "exec"), ns)
        except Exception: pass
ns["SIGMA_SCAN"] = np.linspace(0, 0.9, 46)          # as in notebook cell 7
src10 = "".join(nb["cells"][10]["source"])
exec(compile(src10[:src10.index("fig, ax = plt.subplots")], "cell10", "exec"), ns)
S = ns["SIGMA_SCAN"]; i9 = int(np.argmin(abs(S - 0.9))); print("sigma index", i9, S[i9])
print("m   H_p       H_TWPE(0.9)  std      bias(%)    max|bias| over sigma (%)")
for m, D in ns["lim_data"].items():
    b = (D["mean"] - D["H_p"]) / D["H_p"] * 100
    print(f"{m}  {D['H_p']:.5f}  {D['mean'][i9]:.5f}   {D['std'][i9]:.5f}  {b[i9]:+.3f}    {abs(b).max():.3f}")
print("numpy", np.__version__)
sys.stdout = sys.__stdout__
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"); os.makedirs(OUT, exist_ok=True)
open(os.path.join(OUT, "cell10_bias_log.txt"), "w").write(_buf.getvalue())
