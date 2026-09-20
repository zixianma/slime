"""Versioned local smoke rewards, not a replacement for benchmark evaluation."""


def score(report, version="native_outcome_v1", assistant_turns=None):
    if report.get("aborted") or report.get("outcome") not in ("won", "wrong_serve", "burned", "timeout"):
        raise ValueError("incomplete/aborted native report is not a training example")
    success = float(report["outcome"] == "won")
    planned = int(report.get("errors_planned", 0))
    detected = sum(bool(e.get("flagged_valid") or e.get("prevented"))
                   for e in report.get("injections", []) if e.get("family") == "error")
    prevented = sum(bool(e.get("prevented"))
                    for e in report.get("injections", []) if e.get("family") == "error")
    false_flags = len(report.get("false_positives", []))
    components = dict(task_success=success, errors_planned=planned,
                      errors_offered=report.get("errors_offered", 0),
                      detected_errors=detected, prevented_errors=prevented,
                      false_flags=false_flags)
    if version == "native_outcome_v1":
        reward = success
    elif version == "assistant_detection_v1":
        # Experimental: planned denominator prevents skipping errors to raise recall.
        # No credit for merely naming a mistake early without preventing it.
        if planned <= 0 or detected > planned:
            raise ValueError("detection reward requires a valid error-bearing composite")
        reward = 0.5 * success + 0.5 * detected / planned - min(0.1 * false_flags, 1.0)
    elif version == "cooking_prevention_turns_v1":
        if planned != 1:
            raise ValueError("prevention-turn reward requires exactly one planned error")
        if assistant_turns is None or not 0 <= int(assistant_turns) <= 200:
            raise ValueError("prevention-turn reward requires assistant_turns in [0, 200]")
        if prevented not in (0, 1):
            raise ValueError("single-error prevention must be binary")
        reward = (success + .30 * prevented - .02 * min(false_flags, 15)
                  - .05 * min(int(assistant_turns) / 200, 1))
        # Success remains strictly dominant for every valid component combination.
        epsilon = 1e-12
        assert (-.35-epsilon <= reward <= .30+epsilon) if not success else (
            .65-epsilon <= reward <= 1.30+epsilon)
    else:
        raise ValueError(f"unknown reward version: {version}")
    return float(reward), components
