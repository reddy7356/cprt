"""
CPRT Stage 2 - Jev AI: type-safe, deterministic triage engine
2024 ACC/AHA Perioperative Cardiovascular Guideline (noncardiac surgery) flowcharts.

  * No LLM. Same input -> same output. Every decision cites a rule ID (audit trail).
  * Missing data never becomes "normal": it becomes UNKNOWN -> needs_confirmation.
  * Input  : outputs/stage1_final.csv (+ the 5k data tables)
  * Output : outputs/stage2_jev_triage.csv, outputs/stage2_summary.txt, outputs/jev_rules_v1.csv

Run from project root:
  python3 cprt/scripts/stage2_jev_engine.py
"""
from __future__ import annotations
import argparse, json, os, re, sys
from enum import Enum
from typing import Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ConfigDict

RULES_VERSION = "jev-1.3.0 (ACC/AHA 2024 + local low-risk exception)"
DAY = 1440  # INSPIRE times are in minutes

# ============================================================================
# 1. CONFIG - every threshold in one place (edit here, never in the logic)
# ============================================================================
CFG = dict(
    acs_window_d=30, arrhythmia_window_d=30, hf_window_d=30, stroke_window_d=90,
    acute_tx_window_d=7, biomarker_window_d=30, lab_recent_d=7,
    bnp_abnormal=92.0, ntprobnp_abnormal=300.0,            # 2024 ACC/AHA biomarker cut-offs (pg/mL)
    bnp_decompensated=400.0, ntprobnp_decompensated=1800.0,  # HF decompensation proxy
    troponin_abnormal=0.04,                                  # ng/mL - set to YOUR lab's 99th percentile
    creatinine_rcri=2.0, k_low=3.0, k_high=6.0,
    frailty_age=65, rcri_elevated=2,
    # LOCAL RULE R4L (adjudication 2026-10): low-risk procedure + recent ACS/MI or stroke.
    # "GREEN" = group consensus; set "YELLOW" for the sensitivity analysis; set None to switch the rule off.
    low_risk_exception_track="GREEN",
)

def low_risk_procedure(pcs: str):
    """Low-risk (<1% 30-day MACE) procedure groups agreed at adjudication. ICD-10-PCS: [0]section [1]system [2]operation [3]body part [4]approach."""
    p = (pcs or "").upper()
    if len(p) < 5 or p[0] not in "01": return None
    sysm, part, appr = p[1], p[3], p[4]
    if p[0] == "1": return "obstetric, vaginal/minor" if appr in "78" else None           # 10* obstetrics
    if sysm == "8": return "ophthalmologic (e.g., cataract)"                              # 08* eye
    if sysm == "H": return "skin / breast"                                                # 0H* skin, breast (incl. lumpectomy)
    if sysm == "G" and part in "GHJKLMNPQR": return "endocrine: thyroid/parathyroid"      # 0G* thyroid/parathyroid only
    if sysm == "C" and part in "WX": return "dental"                                      # 0C* teeth
    if sysm == "9" and part in "012345678": return "minor ear (local addition)"          # 09* external/middle ear, tympanic membrane
    if sysm == "U" and appr in "78": return "gynecologic, minor (via natural opening)"
    if appr == "8": return "endoscopic (via natural opening)"                             # GI endoscopy, bronchoscopy, cystoscopy
    if appr == "X": return "superficial / external"
    return None
LAB = {  # regex on lower-case item_name
    "troponin": r"tropon|^tn[it]", "bnp": r"^bnp$|(?<!pro)bnp", "ntprobnp": r"pro.?bnp",
    "creatinine": r"creat", "potassium": r"potass|^k$",
}
AHT = r"amlodipine|nifedipine|felodipine|lercanidipine|cilnidipine|diltiazem|verapamil|sartan|\w+pril\b|thiazide|chlorthalidone|indapamide|atenolol|bisoprolol|carvedilol|metoprolol|nebivolol|propranolol|labetalol|doxazosin|hydralazine|clonidine"
DRUG = {
    "acute_antiarrhythmic": r"adenosine|esmolol|ibutilide|procainamide|lidocaine",     # IV-only agents
    "amiodarone": r"amiodarone", "iv_capable_rate": r"diltiazem|metoprolol",
    "loop": r"furosemide|torsemide|bumetanide",
    "p2y12": r"clopidogrel|ticagrelor|prasugrel",
    "cv_other": r"digoxin|flecainide|propafenone|sotalol|dronedarone|isosorbide|nicorandil|sacubitril|ivabradine|warfarin|apixaban|rivaroxaban|edoxaban|dabigatran",
    "insulin": r"insulin",
}
HIGH_RISK_PCS_SYSTEM = set("BDF4")  # 0B respiratory(intrathoracic), 0D GI, 0F hepatobiliary, 04 lower arteries (proxy)

# ============================================================================
# 2. RULE REGISTRY - each rule = one flowchart node (exportable for the paper)
# ============================================================================
R = {
 "S0":  ("Scope", "Cardiac surgery (ICD-10-PCS 02*) - ACC/AHA NCS guideline not applicable", "-", "Cardiac surgery pathway"),
 "N1":  ("Step 1", "No CV risk factor / disease / symptom", "-", "Proceed to surgery"),
 "N2":  ("Step 2", "Emergency surgery", "-", "Proceed to surgery; manage cardiac issues concurrently"),
 "R1":  ("Step 3", "Acute coronary syndrome (I21/I22/I24 <=30 d)", "1", "Urgent cardiology + MDT: defer / noninvasive / palliation"),
 "R1b": ("Biomarker", "Pre-op troponin elevated without ACS code (myocardial injury vs chronic elevation)", "2a", "Cardiology review / MDT before surgery (confirm ACS vs chronic elevation)"),
 "R5":  ("Step 3", "Recent cardiac arrest (I46 <=30 d)", "1", "Urgent cardiology + MDT: defer surgery"),
 "R2":  ("Step 3", "Unstable arrhythmia (I44/I47/I48/I49 <=30 d + acute treatment or K+ derangement)", "1", "Urgent cardiology/EP + MDT: defer surgery"),
 "R3":  ("Step 3", "Decompensated HF (I50 <=30 d + very high BNP/NT-proBNP or loop diuretic <=7 d)", "1", "Urgent cardiology/HF + MDT: defer surgery"),
 "R4":  ("Modifier", "Recent stroke/TIA, ischemic or hemorrhagic (I60-I64/G45 <=90 d), elective unrelated surgery", "1", "Delay elective surgery >=3 months after event"),
 "R4t": ("Modifier", "Recent stroke/TIA + malignancy (time-sensitive surgery)", "1", "MDT shared decision on timing (delay vs cancer progression)"),
 "R4sR":("Modifier", "Stroke/hemorrhage <=14 d + stroke-related procedure", "1", "Neuro/vascular team + MDT: very recent event, urgent review"),
 "R1c": ("Biomarker", "Troponin elevated + elevated-risk surgery (intraperitoneal/intrathoracic/vascular)", "1", "Urgent cardiology + MDT before surgery"),
 "R4s": ("Modifier", "Recent stroke/TIA + stroke-related procedure (PCS 00*/03*/0N*)", "-", "Neuro/vascular team: deferral rule not applicable to stroke treatment"),
 "R4L": ("Local rule", "Low-risk procedure (<1% MACE) with recent ACS/MI or stroke - group exception", "-", "Low-risk exception: proceed under MAC/local/topical; cardiology/neuro aware"),
 "R3b": ("Step 3", "Pulmonary edema (J81 <=30 d) with HF / hypertensive heart disease (I50/I11/I13)", "1", "Urgent cardiology/HF + MDT: defer surgery"),
 "Y0":  ("Step 3", "Recent arrhythmia code without acute-treatment signal", "-", "ECG + confirm rhythm stability"),
 "E1":  ("Risk", "RCRI >=2 = elevated calculated risk", "2a", "Consider GDMT initiation for long-term CV risk reduction"),
 "E0":  ("Risk", "RCRI 0-1 = low calculated risk, no modifiers", "-", "Proceed to surgery"),
 "M1":  ("Modifier", "Valvular disease (I05/I06/I08/I34-I37) - severity unknown", "1", "TTE (if none <=12 mo) -> AS/MS/AR/MR valve pathway"),
 "M2":  ("Modifier", "Pulmonary hypertension (I27) - severity unknown", "1", "TTE/RHC: if mPAP>40, PVR>5 WU or RV dysfunction -> MDT"),
 "M3":  ("Modifier", "Congenital heart disease (Q20-Q26)", "1", "ACHD specialist consultation before NCS"),
 "M4":  ("Modifier", "Prior stent/CABG (Z95 or I25 + P2Y12 inhibitor)", "1", "MDT antiplatelet plan + surgical timing"),
 "M5":  ("Modifier", "Possible CIED (Z95 - device type not coded)", "1", "Confirm CIED type/manufacturer/model; CIED plan if EMI expected"),
 "F1":  ("Modifier", "Age >=65", "2a", "Frailty assessment with validated tool"),
 "B0":  ("Biomarker", "Functional capacity (DASI/METs) unknown -> biomarker branch", "2a", "Obtain BNP/NT-proBNP (2a); consider troponin (2b)"),
 "B1":  ("Biomarker", "BNP/NT-proBNP abnormal", "2a", "MDT: risks/benefits of further evaluation (echo / stress test / CCTA)"),
 "B2":  ("Biomarker", "BNP/NT-proBNP normal", "2a", "Proceed to surgery"),
 "ECG": ("Testing", "Established CVD -> 12-lead ECG reasonable", "2a", "12-lead ECG"),
 "P1":  ("Pulmonary", "Chronic lung disease / lung cancer / lung resection (outside ACC/AHA)", "-", "Pulmonary optimization pathway (PFTs/ABG as indicated)"),
 "MS":  ("Stage 1", "Model safety-net flag only, no coded CV finding", "-", "Clinician confirms: possible undercoded disease"),
}

# ============================================================================
# 3. TYPE-SAFE SCHEMAS
# ============================================================================
class Track(str, Enum):
    RED = "RED"; YELLOW = "YELLOW"; GREEN = "GREEN"; OUT_OF_SCOPE = "OUT_OF_SCOPE"

class Facts(BaseModel):
    model_config = ConfigDict(frozen=True)
    op_id: int
    stage1_route: str
    p_oof: float = Field(ge=0, le=1)
    age: Optional[float] = None
    emergency: bool
    pcs: str = ""
    dx_hist: frozenset[str]
    dx_30d: frozenset[str]
    dx_90d: frozenset[str]
    dx_14d: frozenset[str] = frozenset()
    drugs_hist: frozenset[str]
    drugs_7d: frozenset[str]
    iv_route_known: bool
    iv_drugs_7d: frozenset[str]
    troponin_max_7d: Optional[float] = None
    bnp_30d: Optional[float] = None
    ntprobnp_30d: Optional[float] = None
    creat_latest: Optional[float] = None
    k_last_7d: Optional[float] = None

class Result(BaseModel):
    op_id: int
    track: Track
    priority: str
    destination: list[str]
    recommendations: list[str]
    red_flags: list[str]
    modifiers: list[str]
    rcri: Optional[int]
    rcri_items: list[str]
    risk_category: str
    biomarker_status: str
    pulmonary_flag: bool
    needs_confirmation: bool
    data_gaps: list[str]
    audit: list[str]

# ============================================================================
# 4. HELPERS
# ============================================================================
def code_in(codes, letter, lo, hi):
    return sorted(c for c in codes if len(c) == 3 and c[0] == letter and c[1:].isdigit() and lo <= int(c[1:]) <= hi)

def has(codes, *names): return sorted(set(codes) & set(names))
def drug(ds, key): return sorted(d for d in ds if re.search(DRUG[key], d))
def n_aht(ds): return len({re.search(AHT, d).group(0) for d in ds if re.search(AHT, d)})
def rec(rid): return f"{R[rid][3]} (Class {R[rid][2]}) [{rid}]" if R[rid][2] != "-" else f"{R[rid][3]} [{rid}]"

# ============================================================================
# 5. THE ENGINE (pure function: Facts -> Result)
# ============================================================================
def triage(f: Facts) -> Result:
    audit, recs, dest, red, mods, gaps = [], [], [], [], [], []
    def fire(rid, detail=""):
        audit.append(f"{rid}: {R[rid][1]}" + (f" | {detail}" if detail else ""))

    # --- pulmonary flag (parallel pathway, outside ACC/AHA)
    pulm = code_in(f.dx_hist, "J", 40, 47) + has(f.dx_hist, "J84", "J96", "C34")
    lung_resection = f.pcs[:3] in ("0BT", "0BB")
    pulm_flag = bool(pulm) or lung_resection
    if pulm_flag: fire("P1", ",".join(pulm) + (" lung resection" if lung_resection else ""))

    def out(track, priority, conf=False, rcri=None, items=(), cat="n/a", bio="n/a"):
        if pulm_flag:
            recs.append(rec("P1")); dest.append(R["P1"][3])
        # an open action item (pulmonary pathway / unconfirmed arrhythmia) can never be GREEN
        if track == Track.GREEN and priority == "routine" and (pulm_flag or any(x.startswith(("Y0:", "R4s:")) for x in audit)):
            track = Track.YELLOW
        if any(x.startswith(("R1b:", "R4t:")) for x in audit) and track in (Track.GREEN, Track.YELLOW) and not priority.startswith("EMERGENCY"):
            track, priority, conf = Track.YELLOW, "high", True
        d = list(dict.fromkeys(dest))
        if track in (Track.YELLOW, Track.RED):  # an open action item -> do not also say "proceed"
            d = [x for x in d if x != R["E0"][3]] or d
        return Result(op_id=f.op_id, track=track, priority=priority, destination=d,
                      recommendations=list(dict.fromkeys(recs)), red_flags=red, modifiers=mods, rcri=rcri,
                      rcri_items=list(items), risk_category=cat, biomarker_status=bio, pulmonary_flag=pulm_flag,
                      needs_confirmation=conf or bool(gaps), data_gaps=gaps, audit=audit)

    # --- S0 scope: cardiac surgery
    if f.pcs.startswith("02"):
        fire("S0", f.pcs); dest.append(R["S0"][3])
        return Result(op_id=f.op_id, track=Track.OUT_OF_SCOPE, priority="n/a", destination=dest, recommendations=[],
                      red_flags=[], modifiers=[], rcri=None, rcri_items=[], risk_category="n/a", biomarker_status="n/a",
                      pulmonary_flag=pulm_flag, needs_confirmation=False, data_gaps=[], audit=audit)

    # --- Step 1: CV risk factor / disease / symptom present?
    cv_dx = (code_in(f.dx_hist, "I", 10, 52) + code_in(f.dx_hist, "I", 60, 74) + code_in(f.dx_hist, "Q", 20, 26)
             + has(f.dx_hist, "R06", "R07", "R55", "Z95"))
    cv_drug = (n_aht(f.drugs_hist) >= 2) or bool(drug(f.drugs_hist, "p2y12") + drug(f.drugs_hist, "cv_other") + drug(f.drugs_hist, "amiodarone"))
    bio_abn = ((f.bnp_30d or 0) > CFG["bnp_abnormal"]) or ((f.ntprobnp_30d or 0) > CFG["ntprobnp_abnormal"]) \
              or ((f.troponin_max_7d or 0) > CFG["troponin_abnormal"])
    if not (cv_dx or cv_drug or bio_abn):
        if f.emergency:
            fire("N2", "no CV finding"); dest.append(R["N2"][3])
            return out(Track.GREEN, "EMERGENCY - alerts: none")
        if f.stage1_route == "MODEL_SAFETY_NET":
            fire("MS", f"p={f.p_oof:.2f}"); recs.append(rec("MS")); dest.append(R["MS"][3])
            fire("N1"); dest.append(R["N1"][3])
            return out(Track.GREEN, "routine", conf=True)
        fire("N1"); dest.append(R["N1"][3])
        return out(Track.GREEN, "routine")

    # --- Step 3 red conditions (evaluated BEFORE emergency so alerts are never hidden)
    acs = has(f.dx_30d, "I21", "I22", "I24")
    trop_high = (f.troponin_max_7d is not None) and f.troponin_max_7d > CFG["troponin_abnormal"]
    if acs: fire("R1", ",".join(acs)); red.append("ACS")
    elif trop_high and len(f.pcs) >= 2 and f.pcs[0] == "0" and f.pcs[1] in HIGH_RISK_PCS_SYSTEM | set("3"):
        fire("R1c", f"troponin={f.troponin_max_7d} pcs={f.pcs}"); red.append("Troponin + elevated-risk surgery")
    elif trop_high:
        fire("R1b", f"troponin={f.troponin_max_7d}")
        if not f.emergency: recs.append(rec("R1b")); dest.append(R["R1b"][3])
    arrest = has(f.dx_30d, "I46")
    if arrest: fire("R5", ",".join(arrest)); red.append("Recent cardiac arrest")

    arr = has(f.dx_30d, "I44", "I47", "I48", "I49")
    acute_tx = drug(f.drugs_7d, "acute_antiarrhythmic") + (drug(f.iv_drugs_7d, "amiodarone") + drug(f.iv_drugs_7d, "iv_capable_rate") if f.iv_route_known else [])
    k_bad = f.k_last_7d is not None and not (CFG["k_low"] <= f.k_last_7d <= CFG["k_high"])
    if arr and (acute_tx or k_bad): fire("R2", f"{arr} tx={acute_tx} K={f.k_last_7d}"); red.append("Unstable arrhythmia")
    elif arr:
        fire("Y0", ",".join(arr)); recs.append(rec("Y0")); dest.append(R["Y0"][3])
        if not f.iv_route_known: gaps.append("drug route missing: IV antiarrhythmic not detectable")

    hf30 = has(f.dx_30d, "I50")
    nat_hi = ((f.bnp_30d or 0) >= CFG["bnp_decompensated"]) or ((f.ntprobnp_30d or 0) >= CFG["ntprobnp_decompensated"])
    loop_iv = drug(f.iv_drugs_7d, "loop") if f.iv_route_known else []
    if hf30 and (nat_hi or loop_iv): fire("R3", f"BNP={f.bnp_30d} NT={f.ntprobnp_30d} loop={loop_iv}"); red.append("Decompensated HF")
    elif has(f.dx_30d, "J81") and has(f.dx_hist, "I50", "I11", "I13"):
        fire("R3b", "J81 + " + ",".join(has(f.dx_hist, "I50", "I11", "I13"))); red.append("Decompensated HF (pulmonary edema)")

    stroke = has(f.dx_90d, "I60", "I61", "I62", "I63", "I64", "G45")
    stroke_related = f.pcs[:2] in ("00", "03", "0N")   # CNS, cerebral/carotid arteries, skull
    cancer = code_in(f.dx_hist, "C", 0, 97)
    if stroke and not f.emergency:
        very_recent = has(f.dx_14d, "I60", "I61", "I62", "I63", "I64", "G45")
        if stroke_related and very_recent:
            fire("R4sR", f"{very_recent} pcs={f.pcs}"); red.append("Very recent stroke + stroke-related surgery")
        elif stroke_related:
            fire("R4s", f"{stroke} pcs={f.pcs}"); recs.append(rec("R4s")); dest.append(R["R4s"][3])
            gaps.append("stroke date from coding - confirm event date")
        elif cancer and not very_recent:   # adjudication: stroke 15-90 d + cancer surgery -> MDT timing (YELLOW-high); <=14 d stays RED
            fire("R4t", f"{stroke} cancer={cancer}"); recs.append(rec("R4t")); dest.append(R["R4t"][3])
        else:
            fire("R4", ",".join(stroke)); red.append("Recent stroke/TIA")

    # --- Step 2: emergency -> proceed (red alerts still reported)
    if f.emergency:
        fire("N2"); dest.append(R["N2"][3])
        alerts = red + (["Troponin elevated"] if trop_high and not acs else [])
        return out(Track.GREEN, "EMERGENCY - alerts: " + ("; ".join(alerts) if alerts else "none"), conf=bool(alerts))

    # --- LOCAL R4L: low-risk procedure exception (ACS/MI and stroke only; never arrest/arrhythmia/HF)
    lr = low_risk_procedure(f.pcs)
    exc_ok = {"ACS", "Recent stroke/TIA"}
    if red and lr and CFG["low_risk_exception_track"] and set(red) <= exc_ok:
        fire("R4L", f"{lr} | conditions={red}"); recs.append(rec("R4L")); dest.append(R["R4L"][3])
        track = Track.GREEN if CFG["low_risk_exception_track"] == "GREEN" else Track.YELLOW
        return out(track, "routine" if track == Track.GREEN else "standard", conf=True)

    if red:
        for rid in ("R1", "R1c", "R5", "R2", "R3", "R3b", "R4sR", "R4"):
            if any(a.startswith(rid + ":") for a in audit): recs.append(rec(rid)); dest.append(R[rid][3])
        return out(Track.RED, "urgent", conf=any(a.startswith("R4") for a in audit))

    # --- Risk estimation: RCRI (proxy)
    items = []
    if len(f.pcs) >= 2 and f.pcs[0] == "0" and f.pcs[1] in HIGH_RISK_PCS_SYSTEM: items.append("high-risk surgery")
    if code_in(f.dx_hist, "I", 20, 25): items.append("ischemic heart disease")
    if has(f.dx_hist, "I50"): items.append("heart failure")
    if has(f.dx_hist, "I60", "I61", "I62", "I63", "I64", "I69", "G45"): items.append("cerebrovascular disease")
    if has(f.dx_hist, "E10") or drug(f.drugs_hist, "insulin"): items.append("insulin-treated DM")
    elif has(f.dx_hist, "E11", "E13", "E14"): gaps.append("DM present, insulin use unknown (RCRI item may be under-counted)")
    if f.creat_latest is not None and f.creat_latest > CFG["creatinine_rcri"]: items.append("creatinine >2.0")
    if f.creat_latest is None: gaps.append("no pre-op creatinine")
    rcri = len(items); elevated = rcri >= CFG["rcri_elevated"]
    cat = "elevated" if elevated else "low"

    # --- Risk modifiers
    if v := code_in(f.dx_hist, "I", 34, 37) + has(f.dx_hist, "I05", "I06", "I08"): fire("M1", ",".join(v)); mods.append("valvular")
    if has(f.dx_hist, "I27"): fire("M2"); mods.append("pulmonary HTN")
    if q := code_in(f.dx_hist, "Q", 20, 26): fire("M3", ",".join(q)); mods.append("congenital HD")
    p2 = drug(f.drugs_hist, "p2y12")
    if has(f.dx_hist, "Z95") or (code_in(f.dx_hist, "I", 20, 25) and p2): fire("M4", f"Z95={bool(has(f.dx_hist,'Z95'))} P2Y12={p2}"); mods.append("stent/CABG")
    if has(f.dx_hist, "Z95"): fire("M5"); mods.append("possible CIED")
    for rid, m in (("M1","valvular"),("M2","pulmonary HTN"),("M3","congenital HD"),("M4","stent/CABG"),("M5","possible CIED")):
        if m in mods: recs.append(rec(rid)); dest.append(R[rid][3])
    if f.age is not None and f.age >= CFG["frailty_age"]: fire("F1", f"age={f.age:.0f}"); recs.append(rec("F1"))
    if cv_dx: fire("ECG"); recs.append(rec("ECG"))

    # --- Low risk, no modifiers -> GREEN
    if not elevated and not mods:
        fire("E0", f"RCRI={rcri}"); dest.append(R["E0"][3])
        return out(Track.GREEN, "routine", rcri=rcri, items=items, cat=cat, bio="not required")

    # --- Elevated risk and/or modifier: functional capacity unknown -> biomarkers
    if elevated: fire("E1", f"RCRI={rcri}: {items}"); recs.append(rec("E1"))
    fire("B0"); gaps.append("functional capacity (DASI/METs) not in dataset")
    if f.bnp_30d is None and f.ntprobnp_30d is None:
        recs.append(rec("B0")); dest.append(R["B0"][3])
        return out(Track.YELLOW, "standard", conf=True, rcri=rcri, items=items, cat=cat, bio="not available")
    if ((f.bnp_30d or 0) > CFG["bnp_abnormal"]) or ((f.ntprobnp_30d or 0) > CFG["ntprobnp_abnormal"]):
        fire("B1", f"BNP={f.bnp_30d} NT={f.ntprobnp_30d}"); recs.append(rec("B1")); dest.append(R["B1"][3])
        return out(Track.YELLOW, "high", conf=True, rcri=rcri, items=items, cat=cat, bio="abnormal")
    fire("B2"); dest.append(R["B2"][3])
    if mods:  # modifier-specific actions still required
        return out(Track.YELLOW, "standard", conf=True, rcri=rcri, items=items, cat=cat, bio="normal")
    return out(Track.GREEN, "routine", conf=True, rcri=rcri, items=items, cat=cat, bio="normal")

# ============================================================================
# 6. DATA LOADING -> Facts
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--stage1", default="outputs/stage1_final.csv")
    ap.add_argument("--out-dir", default="outputs")
    a = ap.parse_args()
    D = lambda f: os.path.join(a.data_dir, f)
    os.makedirs(a.out_dir, exist_ok=True)

    s1 = pd.read_csv(a.stage1); s1 = s1[s1.stage1_positive == 1]
    ops = pd.read_csv(D("operations_5k.csv.gz")).merge(s1[["op_id", "stage1_route", "p_oof"]], on="op_id")
    dx = pd.read_csv(D("diagnosis_5k.csv.gz")); lab = pd.read_csv(D("labs_cardiopulm_5k.csv.gz"))
    med = pd.read_csv(D("medications_cardiopulm_5k.csv.gz"))
    icd = next(c for c in dx.columns if "icd" in c.lower())
    dx["code"] = dx[icd].astype(str).str.replace(".", "", regex=False).str[:3].str.upper()
    med["d"] = med["drug_name"].astype(str).str.lower()
    route_col = next((c for c in med.columns if "route" in c.lower()), None)
    med["iv"] = med[route_col].astype(str).str.lower().str.contains("iv|intraven") if route_col else False
    lab["item"] = lab["item_name"].astype(str).str.lower(); lab["value"] = pd.to_numeric(lab["value"], errors="coerce")
    lab["kind"] = None
    for k, rx in LAB.items(): lab.loc[lab.kind.isna() & lab.item.str.contains(rx, regex=True), "kind"] = k

    has_time = "chart_time" in dx and "orin_time" in ops
    gdx, glab, gmed = dx.groupby("subject_id"), lab.dropna(subset=["kind", "value"]).groupby("subject_id"), med.groupby("subject_id")
    def sub(g, sid):
        try: return g.get_group(sid)
        except KeyError: return None

    results, errors, n_dx = [], 0, 0
    for o in ops.itertuples(index=False):
        t0 = getattr(o, "orin_time", np.nan)
        def win(df, days):
            if df is None: return df
            if not has_time or pd.isna(t0): return df if days is None else df.iloc[0:0]
            m = df.chart_time <= t0
            if days is not None: m &= df.chart_time >= t0 - days * DAY
            return df[m]
        d_all, l_all, m_all = sub(gdx, o.subject_id), sub(glab, o.subject_id), sub(gmed, o.subject_id)
        dset = lambda days: frozenset(win(d_all, days).code) if d_all is not None else frozenset()
        mset = lambda days, iv=False: frozenset((lambda w: w[w.iv].d if iv else w.d)(win(m_all, days))) if m_all is not None else frozenset()
        def labv(kind, days, how):
            if l_all is None: return None
            w = win(l_all, days); w = w[w.kind == kind]
            if w.empty: return None
            return float(w.value.max() if how == "max" else w.sort_values("chart_time").value.iloc[-1] if "chart_time" in w else w.value.iloc[-1])
        try:
            f = Facts(op_id=o.op_id, stage1_route=o.stage1_route, p_oof=float(o.p_oof),
                      age=None if pd.isna(o.age) else float(o.age), emergency=bool(getattr(o, "emop", 0) == 1),
                      pcs=str(getattr(o, "icd10_pcs", "") or "").upper().replace("NAN", ""),
                      dx_hist=dset(None), dx_30d=dset(CFG["acs_window_d"]), dx_90d=dset(CFG["stroke_window_d"]), dx_14d=dset(14),
                      drugs_hist=mset(None), drugs_7d=mset(CFG["acute_tx_window_d"]), iv_route_known=route_col is not None,
                      iv_drugs_7d=mset(CFG["acute_tx_window_d"], iv=True) if route_col else frozenset(),
                      troponin_max_7d=labv("troponin", CFG["lab_recent_d"], "max"),
                      bnp_30d=labv("bnp", CFG["biomarker_window_d"], "last"), ntprobnp_30d=labv("ntprobnp", CFG["biomarker_window_d"], "last"),
                      creat_latest=labv("creatinine", CFG["biomarker_window_d"], "last"), k_last_7d=labv("potassium", 3, "last"))
            n_dx += bool(f.dx_hist)
            results.append(triage(f))
        except Exception as e:  # type-safety: a bad record is reported, never silently triaged
            errors += 1
            if errors <= 5: print(f"VALIDATION ERROR op_id={o.op_id}: {e}")

    df = pd.DataFrame([r.model_dump() for r in results])
    df["track"] = df.track.map(lambda t: t.value if hasattr(t, "value") else t)
    for c in ["destination", "recommendations", "red_flags", "modifiers", "rcri_items", "data_gaps", "audit"]:
        df[c] = df[c].map(lambda x: " || ".join(x))
    df["rules_version"] = RULES_VERSION
    df = s1[["op_id", "subject_id", "stage1_route", "p_oof"]].merge(df, on="op_id")
    df.to_csv(f"{a.out_dir}/stage2_jev_triage.csv", index=False)
    pd.DataFrame([dict(rule_id=k, node=v[0], condition=v[1], acc_aha_class=v[2], action=v[3]) for k, v in R.items()]) \
      .to_csv(f"{a.out_dir}/jev_rules_v1.csv", index=False)

    lines = [f"Jev AI {RULES_VERSION}", f"Stage-1 positives triaged: {len(df)}   validation errors: {errors}", "",
             "TRACK x STAGE-1 ROUTE", pd.crosstab(df.track, df.stage1_route, margins=True).to_string(), "",
             "PRIORITY (top)", df.priority.str.slice(0, 40).value_counts().head(8).to_string(), "",
             f"needs_confirmation: {df.needs_confirmation.mean():.1%}   pulmonary_flag: {df.pulmonary_flag.mean():.1%}", "",
             "RED reasons", df.loc[df.track == "RED", "red_flags"].str.split(r" \|\| ").explode().value_counts().to_string(), "",
             "Modifiers", df.modifiers.replace("", np.nan).dropna().str.split(r" \|\| ").explode().value_counts().to_string(), "",
             "Biomarker status", df.biomarker_status.value_counts().to_string(), "",
             "DATA CHECK", f"lab items mapped: {lab.kind.value_counts().to_dict()}",
             f"drug route column: {route_col}", f"diagnosis/OR times usable: {has_time}",
             f"cases with >=1 diagnosis dated BEFORE surgery: {n_dx}/{len(results)} (if low, diagnoses are coded after OR -> tell Claude)"]
    txt = "\n".join(lines); print(txt); open(f"{a.out_dir}/stage2_summary.txt", "w").write(txt)
    print(f"\nsaved -> {a.out_dir}/stage2_jev_triage.csv, stage2_summary.txt, jev_rules_v1.csv")

if __name__ == "__main__":
    main()
