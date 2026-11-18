"""Optional conservative routing for the verified machine-1/2 numeric layout."""

import re


def conservative_prediction(
    baseline, baseline_confidence, candidate, candidate_confidence, region, machine_id
):
    """Never insert punctuation: choose an observed prediction or flag review."""
    if str(machine_id) not in {"1", "2"}:
        raise ValueError(
            "Conservative numerical policy is only defined for machines 1/2"
        )
    if region not in {f"region{i}" for i in range(1, 7)}:
        raise ValueError("No verified numerical format for this region")
    pattern = r"[0-9]+\.[0-9]{2}" if region == "region6" else r"[+-]?[0-9]+"
    old_valid = re.fullmatch(pattern, baseline) is not None
    new_valid = re.fullmatch(pattern, candidate) is not None
    disagreement = baseline != candidate
    if old_valid and baseline_confidence >= 0.95:
        return {
            "text": baseline,
            "confidence": baseline_confidence,
            "source": "baseline_confident",
            "needs_review": disagreement,
        }
    if new_valid:
        return {
            "text": candidate,
            "confidence": candidate_confidence,
            "source": "candidate_valid",
            "needs_review": disagreement and old_valid,
        }
    if old_valid:
        return {
            "text": baseline,
            "confidence": baseline_confidence,
            "source": "baseline_after_invalid_candidate",
            "needs_review": True,
        }
    return {
        "text": baseline,
        "confidence": baseline_confidence,
        "source": "both_invalid",
        "needs_review": True,
    }
