"""
CPRT Stage 1 - XGBoost cardiopulmonary cohort filter (Definition A, broad).
Features: demographics, ASA, emop, procedure family, pre-op labs, medication classes.
Excluded: department (R1) and diagnoses (R2) -> leakage control.
Run from project root:  python3 cprt/scripts/train_stage1_xgboost.py
Ablation (no meds):     python3 cprt/scripts/train_stage1_xgboost.py --no-meds
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from xgboost import XGBClassifier

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default=".")
p.add_argument("--labels", default="silver_labels_v3.csv.gz")
p.add_argument("--gold", default="", help="optional gold file (.xlsx/.xltx/.csv) with op_id, gold_label")
p.add_argument("--no-meds", action="store_true", help="drop medication features (leakage ablation)")
p.add_argument("--target-recall", type=float, default=0.90)
p.add_argument("--out", default="outputs/stage1")
a = p.parse_args()
D = lambda f: os.path.join(a.data_dir, f)
os.makedirs(a.out, exist_ok=True)
SEED = 42

# ---------- load ----------
for f in ["operations_5k.csv.gz", "labs_cardiopulm_5k.csv.gz", a.labels] + ([] if a.no_meds else ["medications_cardiopulm_5k.csv.gz"]):
    if not os.path.exists(D(f)):
        sys.exit(f"ERROR: missing {D(f)}  -> check --data-dir / file name")
ops = pd.read_csv(D("operations_5k.csv.gz"))
lab = pd.read_csv(D("labs_cardiopulm_5k.csv.gz"))
lbl = pd.read_csv(D(a.labels))[["op_id", "silver_label"]]
print(f"ops={len(ops)} labs={len(lab)} labels={len(lbl)}")

# ---------- operations features ----------
X = ops[["op_id", "subject_id"]].copy()
for c in ["age", "weight", "height", "asa", "emop"]:
    if c in ops: X[c] = pd.to_numeric(ops[c], errors="coerce")
if "sex" in ops: X["male"] = (ops["sex"].astype(str).str.upper() == "M").astype(int)
if {"weight", "height"} <= set(ops):
    X["bmi"] = ops["weight"] / (ops["height"] / 100) ** 2
if "icd10_pcs" in ops:  # procedure family = first 3 chars (section/body system/operation)
    X["proc_fam"] = ops["icd10_pcs"].astype(str).str[:3].astype("category")

# pre-op window: keep events before OR-in (same subject time base in INSPIRE)
def preop(df):
    if "chart_time" in df and "orin_time" in ops:
        m = df.merge(ops[["op_id", "subject_id", "orin_time"]], on="subject_id")
        return m[m["chart_time"] < m["orin_time"]]
    print("WARN: no chart_time/orin_time -> using all-time events")
    return df.merge(ops[["op_id", "subject_id"]], on="subject_id")

# ---------- lab features: min/max/last/count per item ----------
L = preop(lab)
L["value"] = pd.to_numeric(L["value"], errors="coerce")
L = L.dropna(subset=["value"]).sort_values("chart_time" if "chart_time" in L else "op_id")
g = L.groupby(["op_id", "item_name"])["value"]
lf = pd.concat({"min": g.min(), "max": g.max(), "last": g.last(), "n": g.count()}, axis=1).unstack("item_name")
lf.columns = [f"lab_{i}_{s}" for s, i in lf.columns]
X = X.merge(lf, left_on="op_id", right_index=True, how="left")

# ---------- medication class counts ----------
MED = {
 "beta_blocker": ["olol", "carvedilol", "labetalol"], "acei_arb": ["pril", "sartan"],
 "ccb": ["dipine", "diltiazem", "verapamil"], "diuretic": ["furosemide", "torsemide", "thiazide", "spironolactone", "eplerenone", "indapamide"],
 "antiarrhythmic": ["amiodarone", "digoxin", "flecainide", "propafenone", "sotalol", "dronedarone"],
 "anticoag": ["warfarin", "apixaban", "rivaroxaban", "edoxaban", "dabigatran"],
 "antiplatelet": ["clopidogrel", "ticagrelor", "prasugrel", "cilostazol"], "statin": ["statin"],
 "nitrate": ["isosorbide", "nicorandil"], "inhaler": ["ipratropium", "tiotropium", "salbutamol", "albuterol", "formoterol", "salmeterol", "budesonide", "fluticasone"],
 "pulm_htn": ["sildenafil", "tadalafil", "bosentan", "ambrisentan", "macitentan", "riociguat"],
 "hf_specific": ["sacubitril", "ivabradine", "dapagliflozin", "empagliflozin"],
}
if not a.no_meds:
    M = preop(pd.read_csv(D("medications_cardiopulm_5k.csv.gz")))
    dn = M["drug_name"].astype(str).str.lower()
    for cls, keys in MED.items():
        M[cls] = dn.str.contains("|".join(keys)).astype(int)
    mf = M.groupby("op_id")[list(MED)].max()          # 1 = ever on class pre-op
    mf["med_class_total"] = mf.sum(axis=1)
    X = X.merge(mf.add_prefix("med_"), left_on="op_id", right_index=True, how="left")
    X[[c for c in X if c.startswith("med_")]] = X[[c for c in X if c.startswith("med_")]].fillna(0)

# ---------- target + patient-level split 60/20/20 ----------
X = X.merge(lbl, on="op_id")
y, groups = X.pop("silver_label").values, X["subject_id"].values
strat = y * 10 + X.get("asa", pd.Series(0, index=X.index)).fillna(0).astype(int)
folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, strat, groups))
te, va = folds[0][1], folds[1][1]
tr = np.setdiff1d(np.arange(len(X)), np.concatenate([te, va]))
feat = [c for c in X if c not in ("op_id", "subject_id")]
print(f"features={len(feat)} train/val/test={len(tr)}/{len(va)}/{len(te)} prevalence={y.mean():.3f}")

clf = XGBClassifier(n_estimators=1000, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                    min_child_weight=3, eval_metric="aucpr", early_stopping_rounds=50,
                    scale_pos_weight=(y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1),
                    enable_categorical=True, tree_method="hist", random_state=SEED)
clf.fit(X.iloc[tr][feat], y[tr], eval_set=[(X.iloc[va][feat], y[va])], verbose=False)

# ---------- operating point: highest threshold with val recall >= target ----------
pv = clf.predict_proba(X.iloc[va][feat])[:, 1]
thr = 0.5
for t in np.linspace(0.99, 0.01, 99):
    if ((pv >= t) & (y[va] == 1)).sum() / max((y[va] == 1).sum(), 1) >= a.target_recall:
        thr = float(t); break

def report(name, yt, pr):
    tn, fp, fn, tp = confusion_matrix(yt, (pr >= thr).astype(int), labels=[0, 1]).ravel()
    r = {"n": int(len(yt)), "roc_auc": roc_auc_score(yt, pr) if len(set(yt)) > 1 else None,
         "pr_auc": average_precision_score(yt, pr) if len(set(yt)) > 1 else None,
         "recall": tp / max(tp + fn, 1), "precision": tp / max(tp + fp, 1),
         "specificity": tn / max(tn + fp, 1), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}
    print(f"\n[{name}] " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
    return r

pt = clf.predict_proba(X.iloc[te][feat])[:, 1]
res = {"threshold": thr, "no_meds": a.no_meds, "best_iter": int(clf.best_iteration), "test_vs_silver": report("TEST vs silver", y[te], pt)}

# ---------- out-of-fold (OOF) predictions: every case scored by a model that never saw it ----------
pall = clf.predict_proba(X[feat])[:, 1]
poof = np.zeros(len(X))
params = {k: v for k, v in clf.get_params().items() if k not in ("early_stopping_rounds", "n_estimators")}
for f_tr, f_te in folds:
    m = XGBClassifier(**params, n_estimators=clf.best_iteration + 1)
    m.fit(X.iloc[f_tr][feat], y[f_tr], verbose=False)
    poof[f_te] = m.predict_proba(X.iloc[f_te][feat])[:, 1]
res["oof_vs_silver"] = report("OOF vs silver (all cases)", y, poof)
hidden = int(((y == 0) & (poof >= 0.5)).sum())
print(f"silver-negative but model p>=0.50 (possible hidden/undercoded disease): {hidden}")
res["silver_neg_high_p"] = hidden

# ---------- optional gold evaluation (uses OOF -> no train-set optimism) ----------
if a.gold:
    gd = pd.read_csv(a.gold) if a.gold.endswith(".csv") else pd.read_excel(a.gold, engine="openpyxl")
    gd = gd.dropna(subset=["gold_label"])[["op_id", "gold_label"]].merge(
        pd.DataFrame({"op_id": X["op_id"], "p": poof, "silver": y}), on="op_id")
    res["gold_oof"] = report(f"GOLD OOF (n={len(gd)})", gd["gold_label"].astype(int).values, gd["p"].values)
    res["gold_silver_rules"] = report("GOLD vs silver rules (reference)", gd["gold_label"].astype(int).values, gd["silver"].values.astype(float))
    miss = gd[(gd.gold_label == 1) & (gd.p < thr)]
    print("\nGOLD positives missed by model:\n", miss.to_string(index=False) if len(miss) else "none")

# ---------- save ----------
clf.save_model(f"{a.out}/stage1_xgb.json")
pd.DataFrame({"op_id": X["op_id"], "subject_id": X["subject_id"], "p_cardiopulm": pall, "p_oof": poof,
              "stage1_positive": (poof >= thr).astype(int), "silver_label": y}).to_csv(f"{a.out}/stage1_predictions.csv", index=False)
pd.Series(clf.get_booster().get_score(importance_type="gain")).sort_values(ascending=False).to_csv(f"{a.out}/feature_importance_gain.csv", header=["gain"])
json.dump(res, open(f"{a.out}/metrics.json", "w"), indent=2, default=float)
print(f"\nthreshold={thr:.2f}  saved -> {a.out}/")
