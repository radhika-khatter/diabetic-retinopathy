"""Step 7: Referral logic (human-in-the-loop)."""
GRADES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

def decide(grade, confidence, quality_ok, conf_thr=0.60):
    if not quality_ok:
        return "RECAPTURE", ["Image ungradeable - retake the fundus photo"]
    reasons = []
    if grade >= 2: reasons.append(f"Referable DR predicted ({GRADES[grade]})")
    if confidence < conf_thr: reasons.append(f"Low calibrated confidence ({confidence:.0%})")
    if reasons: return "REFER_TO_SPECIALIST", reasons
    if grade == 1: return "RESCREEN_12_MONTHS", ["Mild DR - monitor"]
    return "ROUTINE_SCREENING", ["No DR detected with adequate confidence"]
