# CPRT: Cardiopulmonary Preoperative Routing Tool

A two-stage, **LLM-free** preoperative cardiopulmonary triage pipeline built on the INSPIRE 1.4.2 perioperative dataset.

| Stage | Component | What it does |
|---|---|---|
| 1 | Silver rules + **XGBoost** | Lenient screen: is the patient cardiopulmonary? (rules **or** model p ≥ 0.5) |
| 2 | **Jev AI** (jev-1.3.0) | Deterministic, type-safe (pydantic) encoding of the 2024 ACC/AHA perioperative flowchart → GREEN / YELLOW / RED / OUT_OF_SCOPE, a destination, a pulmonary flag and an audit trail of fired rules |

No component calls a large language model or any external service; the same input always gives the same output.

## Key results (see `docs/manuscript.md`)

- Stage 1 hybrid screen: 2,524 / 5,000 cases (50.5%) passed to Stage 2; sensitivity 100% against 100 clinician-labelled cases. XGBoost out-of-fold ROC-AUC 0.874.
- Stage 2 validation (48 independent cases, 3 blinded clinicians, engine frozen): 41/48 agreement with the majority (85.4%, 95% CI 75.0–93.8; κ 0.78), no majority-RED case routed GREEN.
- Whole cohort: 88.1% proceed without extra cardiac work-up; 488 (9.8%) flagged for action.

## Repository layout

```
scripts/
  train_stage1_xgboost.py   Stage 1 XGBoost (patient-level split, out-of-fold predictions)
  finalize_stage1.py        Lenient hybrid screen -> outputs/stage1_final.csv
  stage2_jev_engine.py      Jev AI rules engine (jev-1.3.0) -> outputs/stage2_jev_triage.csv
  make_gold_workbooks.py    Blinded Stage 1 rater workbooks
  make_stage2_gold.py       Blinded, stratified Stage 2 rater workbooks (blank templates)
  evaluate_stage2.py        Reproduces Stage 2 validation results from the COMPLETED rater workbooks
docs/
  manuscript.md             Manuscript draft
  figure1_patient_flow.png  Figure 1
  jev_rules_v1.3.csv        Rule registry (IDs, guideline node, class of recommendation)
```

## Data

**No data are included.** INSPIRE is distributed by PhysioNet under a credentialed data use agreement that forbids redistribution. Obtain access at PhysioNet, then build the 5,000-case subset files the scripts expect in the project root (`operations_5k.csv.gz`, `diagnosis_5k.csv.gz`, `labs_cardiopulm_5k.csv.gz`, `medications_cardiopulm_5k.csv.gz`).

## Running

```bash
pip install -r requirements.txt
python3 scripts/train_stage1_xgboost.py            # add --gold gold100_final.csv to evaluate against clinician labels
python3 scripts/finalize_stage1.py
python3 scripts/stage2_jev_engine.py
python3 scripts/make_stage2_gold.py --tag stage2_valid --exclude stage2_gold_final.csv --raters 3
# after raters return their completed files:
python3 scripts/evaluate_stage2.py \
    --raters gold_labeling/stage2_valid_rater1_DONE_reddy.xlsx \
             gold_labeling/stage2_valid_rater2_DONE_sahil.xlsx \
             gold_labeling/stage2_valid_rater3_DONE_shusrita.xlsx \
    --key gold_labeling/stage2_valid_KEY.csv
```

### Rater workbook provenance

`make_stage2_gold.py` writes **blank templates** (`stage2_valid_rater1/2/3.xlsx`). Each rater received one workbook (sheets: Guide, Cases) and returned it completed as `stage2_valid_rater<N>_DONE_<name>.xlsx`. Only the `_DONE_` files are the evaluation inputs. `evaluate_stage2.py` prints each file's fill count and MD5 checksum and stops if a blank template is supplied. In validation, rater 1 left one case (#24) blank; that case's majority came from the other two raters.

Expected output (matches the manuscript): 48 cases with a majority label, Jev AI agreement 41/48 (85.4%), Cohen κ 0.78, Fleiss κ 0.46, RED sensitivity 7/9, no severe under-triage.

Run all commands from the project root containing the data files.

## Status and caveats

Research code, not a medical device. Not for clinical use without local validation. Local rules (low-risk procedure exception, stroke timing, troponin, decompensated-HF proxy) reflect consensus at the authors' institution and are labelled separately from guideline rules. Known errors are listed in the manuscript's Limitations.

An AI assistant (Claude, Anthropic) helped write code and draft the manuscript; all clinical labels, adjudication decisions and local policies were made by the authors.
