"""
CPRT Stage 2 validation - reproduce the reported agreement from the COMPLETED rater workbooks.

Inputs (completed files returned by raters, NOT the blank templates in gold_labeling/):
  --raters  stage2_valid_rater1_DONE_*.xlsx stage2_valid_rater2_DONE_*.xlsx stage2_valid_rater3_DONE_*.xlsx
  --key     stage2_valid_KEY.csv   (Jev AI answers; open only after all raters finish)

Example:
  python3 cprt/scripts/evaluate_stage2.py \
      --raters gold_labeling/stage2_valid_rater1_DONE_reddy.xlsx \
               gold_labeling/stage2_valid_rater2_DONE_sahil.xlsx \
               gold_labeling/stage2_valid_rater3_DONE_shusrita.xlsx \
      --key gold_labeling/stage2_valid_KEY.csv

Reference standard = majority vote (>=2 of 3). A blank rating is ignored; the case keeps a
majority if the remaining raters agree. Prints a file-completeness check first, so blank
templates are caught immediately.
"""
import argparse, hashlib, sys
import numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters

TRACKS = ["GREEN", "YELLOW", "RED", "OUT_OF_SCOPE"]
LEVEL = {"GREEN": 0, "YELLOW": 1, "RED": 2}

ap = argparse.ArgumentParser()
ap.add_argument("--raters", nargs="+", required=True)
ap.add_argument("--key", required=True)
ap.add_argument("--boot", type=int, default=4000)
ap.add_argument("--seed", type=int, default=42)
A = ap.parse_args()

# 1. Load and check completeness -------------------------------------------------------
print("FILE CHECK")
d = None
for i, f in enumerate(A.raters, 1):
    c = pd.read_excel(f, sheet_name="Cases")
    col = next(k for k in c.columns if k.startswith("track_"))
    c[col] = c[col].astype("string").str.strip().str.upper()
    n_fill = c[col].notna().sum()
    md5 = hashlib.md5(open(f, "rb").read()).hexdigest()[:8]
    blanks = c.loc[c[col].isna(), "case_no"].tolist()
    print(f"  rater{i}: {f}  md5={md5}  filled {n_fill}/{len(c)}" + (f"  blank case_no={blanks}" if blanks else ""))
    if n_fill == 0:
        sys.exit(f"  ERROR: rater{i} file is empty - this is a blank template, not a completed workbook.")
    c = c[["case_no", "op_id", col]].rename(columns={col: f"r{i}"})
    d = c if d is None else d.merge(c[["op_id", f"r{i}"]], on="op_id", how="outer")
R = [f"r{i}" for i in range(1, len(A.raters) + 1)]

key = pd.read_csv(A.key)[["op_id", "track"]].rename(columns={"track": "jev"})
d = d.merge(key, on="op_id", how="left")
if d.jev.isna().any():
    sys.exit("ERROR: some cases are missing from the KEY file.")

# 2. Majority reference standard ------------------------------------------------------
def majority(row):
    v = pd.Series([row[r] for r in R if pd.notna(row[r])]).value_counts()
    return v.index[0] if len(v) and v.iloc[0] >= 2 else pd.NA
d["maj"] = d.apply(majority, axis=1)
d["unanimous"] = d[R].nunique(axis=1, dropna=False).eq(1)
m = d[d.maj.notna()].copy()
print(f"\nCASES: {len(d)} total, {len(m)} with a majority label, {d.unanimous.sum()} unanimous")

# 3. Jev AI vs majority ---------------------------------------------------------------
agree = (m.jev == m.maj)
rng = np.random.default_rng(A.seed)
idx = [rng.integers(0, len(m), len(m)) for _ in range(A.boot)]
acc_ci = np.percentile([agree.values[i].mean() for i in idx], [2.5, 97.5])
kap = cohen_kappa_score(m.maj, m.jev)
kap_ci = np.percentile([cohen_kappa_score(m.maj.values[i], m.jev.values[i]) for i in idx], [2.5, 97.5])
print("\nJEV AI vs CLINICIAN MAJORITY")
print(f"  agreement {agree.sum()}/{len(m)} = {agree.mean():.1%} (95% CI {acc_ci[0]:.1%}-{acc_ci[1]:.1%})")
print(f"  Cohen kappa {kap:.2f} (95% CI {kap_ci[0]:.2f}-{kap_ci[1]:.2f})")

# 4. Pairwise and Fleiss kappa --------------------------------------------------------
print("\nPAIRWISE (cases rated by both)")
for a in R + ["jev"]:
    for b in R:
        if a == b or (a in R and R.index(a) >= R.index(b)): continue
        p = d[[a, b]].dropna()
        print(f"  {a:>3} vs {b}: agreement {(p[a] == p[b]).mean():.0%}  kappa {cohen_kappa_score(p[a], p[b]):.2f}  (n={len(p)})")
full = d[R].dropna()
tab, _ = aggregate_raters(full.values)
print(f"  Fleiss kappa (all raters, n={len(full)} complete cases): {fleiss_kappa(tab):.2f}")

# 5. Confusion matrix, PPV, RED safety ------------------------------------------------
cm = pd.crosstab(m.jev, m.maj).reindex(index=TRACKS, columns=TRACKS, fill_value=0)
cm["PPV"] = [f"{cm.loc[t, t] / cm.loc[t, TRACKS].sum():.0%}" if cm.loc[t, TRACKS].sum() else "-" for t in TRACKS]
print("\nCONFUSION MATRIX (rows = Jev AI, columns = majority)")
print(cm.to_string())
red = m[m.maj == "RED"]
print(f"\n  RED sensitivity {(red.jev == 'RED').sum()}/{len(red)}")
print(f"  severe under-triage (majority RED -> Jev GREEN): {(red.jev == 'GREEN').sum()}")

# 6. Disagreements --------------------------------------------------------------------
dis = m[~agree].copy()
dis["direction"] = [("under" if LEVEL[j] < LEVEL[g] else "over") if j in LEVEL and g in LEVEL else "scope"
                    for j, g in zip(dis.jev, dis.maj)]
print(f"\nDISAGREEMENTS ({len(dis)})")
print(dis[["case_no", "jev", "maj"] + R + ["direction"]].sort_values("case_no").to_string(index=False))
