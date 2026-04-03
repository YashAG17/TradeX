"""Baseline policy for the surveillance benchmark."""

from __future__ import annotations

from .models import SurveillanceObservation


def choose_surveillance_action(observation: SurveillanceObservation) -> str:
    """Risk-aware baseline that handles normal, bursty, and manipulative cases smoothly."""

    risk_score = (
        0.28 * observation.suspiciousness_score
        + 0.24 * observation.manipulation_score
        + 0.20 * observation.pattern_indicator
        + 0.16 * observation.burst_indicator
        + 0.07 * min(1.0, observation.recent_slippage_impact / 0.05)
        + 0.05 * min(1.0, observation.trade_frequency / 10.0)
    )

    clearly_normal = (
        observation.suspiciousness_score < 0.36
        and observation.manipulation_score < 0.30
        and observation.burst_indicator < 0.45
        and observation.pattern_indicator < 0.45
        and observation.recent_slippage_impact < 0.02
    )
    if clearly_normal:
        return "ALLOW"

    severe_manipulation = (
        observation.manipulation_score >= 0.78
        or (
            observation.pattern_indicator >= 0.72
            and observation.suspiciousness_score >= 0.70
            and observation.recent_slippage_impact >= 0.02
        )
        or (
            risk_score >= 0.76
            and observation.pattern_indicator >= 0.55
        )
    )
    if severe_manipulation:
        return "BLOCK"

    bursty_or_elevated = (
        observation.burst_indicator >= 0.70
        or observation.trade_frequency >= 7.5
        or risk_score >= 0.58
    )
    if bursty_or_elevated:
        return "FLAG"

    if observation.suspiciousness_score >= 0.46 or risk_score >= 0.42:
        return "MONITOR"

    return "ALLOW"
