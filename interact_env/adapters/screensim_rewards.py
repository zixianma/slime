"""Versioned native rewards; do not conflate detection with in-time success."""
import math

VERSIONS = {"screensim_f1_v1", "screensim_intime_success_v1"}


def score(report, version):
    if version not in VERSIONS:
        raise ValueError(f"unsupported ScreenSim reward: {version}")
    if version == "screensim_f1_v1" and not report["n_fired"]:
        raise ValueError("F1 reward requires at least one fired error opportunity")
    value = float(report["f1"] if version == "screensim_f1_v1" else report["success"])
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("invalid native ScreenSim reward")
    if version == "screensim_intime_success_v1" and value not in (0, 1):
        raise ValueError("native in-time success must be binary")
    return value
