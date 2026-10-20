# Supplementary analyses

These scripts reproduce numbers quoted in the paper that are **not** produced by `twpe_analysis.ipynb`.
They do not modify the notebook or its results. Each script reads the official functions
(`permutation_entropy`, `twpe`, `kl_sym`) from Cell 1 of the notebook with `ast`, repeats the sample
draw of Cell 2A (`random.seed(42)`, 50 stars per class) and checks it against values printed by the
notebook before computing anything new.

## How to run

From the repository root (the notebook and `data/observed/ogle3_lpv/` must be present, including the
cached light curves in `phot_I/`, which the notebook downloads on its first run, and `OSARGs.dat`):

```bash
python3 -I supplementary/<script>.py
```

Requirements: Python 3.10+, `numpy`, `scipy`, `matplotlib` (see `requirements.txt`).
Each script writes its tables, figure and log to `supplementary/results/` (created if missing).
Seeds are fixed. The files in `results/` are the reference outputs, so a new run can be compared with them.
Logs record the library versions used; the reference outputs were produced with Python 3.12.2,
numpy 1.26.4, scipy 1.13.1 and matplotlib 3.9.2 (conda). Other versions can change the way numbers are
printed (for example `np.float64(...)` with numpy 2) and the bytes of the PNG files, but not the values.

## Scripts

| Script | What it does | Outputs in `results/` | Used in the paper |
|---|---|---|---|
| `cell10_bias_check.py` | Re-executes the code of Cell 10 and prints the mean bias H_TWPE − H_p at σ = 0.9 for m = 3–6 | `cell10_bias_log.txt` | Sect. 3.2 (bias at σ = 0.9) |
| `osarg_tail_inspection.py` | The two OSARGs with D_KL ≈ 770×10⁻³: gaps, N, Spearman against gap and N, effect of truncating the largest gap, Lomb–Scargle peaks | `osarg_tail_table.csv`, `osarg_tail.png`, `osarg_tail_log.txt` | Sect. 4.3 (OSARGs in the upper tail) |
| `gap_truncation_all150.py` | Same truncation control for all 150 stars; Spearman of D_KL against largest gap and N | `gap_truncation_all150.csv`, `.png`, `gap_truncation_all150_log.txt` | Sect. 4.3 |
| `dkl_tests_mira_srv.py` | Mann–Whitney between the three classes (Holm-adjusted); D_KL against P/Δ_gap and nulls for the 100 Mira and SRV stars | `dkl_tests_mira_srv_per_star.csv`, `dkl_tests_mira_srv.png`, `dkl_tests_mira_srv_log.txt` | Sect. 4.3 (Mann–Whitney tests) |
| `table2_catalogue_p1.py` | Catalogue primary period P1 (median ± IQR/1.35) for the 150 sampled stars; needs `OSARGs.dat` | `table2_catalogue_p1.csv`, `table2_catalogue_p1_log.txt` | Table 2 |
| `dkl_vs_period_all150.py` | D_KL against P1/Δ_gap with the 150 stars: pooled and within-class Spearman, partial Spearman, and two null models on the real timestamps | `dkl_vs_period_all150.csv`, `.png`, `dkl_vs_period_all150_log.txt` | Sect. 4.3 (period, amplitude, signal-to-noise) |
| `p1_lomb_scargle_check.py` | Validation: global Lomb–Scargle peak against the catalogue P1. It shows that the global peak is not a reliable P1 for SRVs and OSARGs | `p1_lomb_scargle_check.csv`, `p1_lomb_scargle_check_log.txt` | Not quoted (record of why catalogue periods are used) |

## Notes

- Each script is self-contained (no shared module).
- The null models (`dkl_tests_mira_srv.py`, `dkl_vs_period_all150.py`) are simple: a sinusoid at the
  catalogue period with the fitted amplitude and Gaussian noise, with random phase, and the observed
  magnitudes shuffled over the real timestamps. The signal-to-noise ratio is a rough proxy
  (fitted amplitude over residual standard deviation).
- Only m = 4 and the weight w₁ are used.
- The OGLE-III catalogue and light curves are public data (Soszyński et al. 2009).
