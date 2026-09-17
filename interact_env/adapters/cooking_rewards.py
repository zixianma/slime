"""Versioned local smoke rewards, not a replacement for benchmark evaluation."""


def score(report, version="native_outcome_v1"):
    if report.get("aborted") or report.get("outcome") not in ("won", "wrong_serve", "burned", "timeout"):
        raise ValueError("incomplete/aborted native report is not a training example")
    success = float(report["outcome"] == "won")
    planned = int(report.get("errors_planned", 0))
    detected = sum(bool(e.get("flagged_valid") or e.get("prevented"))
                   for e in report.get("injections", []) if e.get("family") == "error")
    false_flags = len(report.get("false_positives", []))
    components = dict(task_success=success, errors_planned=planned,
                      errors_offered=report.get("errors_offered", 0),
                      detected_errors=detected, false_flags=false_flags)
    if version == "native_outcome_v1":
        reward = success
    elif version == "assistant_detection_v1":
        # Experimental: planned denominator prevents skipping errors to raise recall.
        # No credit for merely naming a mistake early without preventing it.
        if planned <= 0 or detected > planned:
            raise ValueError("detection reward requires a valid error-bearing composite")
        reward = 0.5 * success + 0.5 * detected / planned - min(0.1 * false_flags, 1.0)
    else:
        raise ValueError(f"unknown reward version: {version}")
    return float(reward), components
