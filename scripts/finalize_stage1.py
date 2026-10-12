"""
CPRT Stage 1 final (lenient hybrid): positive if RULES positive OR XGBoost p_oof >= threshold.
Run from project root:
  python3 cprt/scripts/finalize_stage1.py --gold gold100_final.csv
  python3 cprt/scripts/finalize_stage1.py --model-threshold 0.4      # more lenient
"""
import argparse, pandas as pd
p = argparse.ArgumentParser()
p.add_argument("--pred", default="outputs/stage1_v2/stage1_predictions.csv")
p.add_argument("--model-threshold", type=float, default=0.5)
p.add_argument("--gold", default="")
p.add_argument("--out", default="outputs/stage1_final.csv")
a = p.parse_args()

d = pd.read_csv(a.pred)
rules = d.silver_label == 1
n = len(d)
gold = pd.read_csv(a.gold)[["op_id", "gold_label"]] if a.gold else None

print(f"Rules alone: {rules.sum()} / {n} positive ({rules.mean():.1%})\n")
print("model_thr | added by model | total Stage-1 positive | % of cases" + (" | gold sens | gold spec" if a.gold else ""))
for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
    pos = rules | (d.p_oof >= t)
    line = f"   {t:.1f}    | {int((~rules & (d.p_oof >= t)).sum()):>14} | {int(pos.sum()):>22} | {pos.mean():>8.1%}"
    if a.gold:
        g = gold.merge(d.assign(pos=pos.astype(int))[["op_id", "pos"]], on="op_id")
        tp = ((g.gold_label == 1) & (g.pos == 1)).sum(); fn = ((g.gold_label == 1) & (g.pos == 0)).sum()
        tn = ((g.gold_label == 0) & (g.pos == 0)).sum(); fp = ((g.gold_label == 0) & (g.pos == 1)).sum()
        line += f" | {tp/(tp+fn):>9.3f} | {tn/(tn+fp):.3f}"
    print(line)

t = a.model_threshold
d["stage1_route"] = "NEGATIVE"
d.loc[~rules & (d.p_oof >= t), "stage1_route"] = "MODEL_SAFETY_NET"   # possible undercoded disease
d.loc[rules, "stage1_route"] = "RULES"
d["stage1_positive"] = (d.stage1_route != "NEGATIVE").astype(int)
d[["op_id", "subject_id", "silver_label", "p_oof", "stage1_route", "stage1_positive"]].to_csv(a.out, index=False)
print(f"\nUsing threshold {t}: ", d.stage1_route.value_counts().to_dict())
print(f"saved -> {a.out}  (this is the Stage 2 input)")
