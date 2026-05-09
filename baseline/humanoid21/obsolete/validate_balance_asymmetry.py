"""Sanity check for BalanceValueRewarder asymmetric safe-zone.

Directly feeds hand-crafted balance_output dicts into the scoring
function to verify:
  1. Inside the safe zone => score == 1.0 (modulo small velocity terms).
  2. Same-magnitude backward offset is penalized harder than forward.
  3. Moving toward center reduces penalty; moving away increases it.
"""

from __future__ import annotations

from pathlib import Path
import sys

COMBATBENCH_DIR = Path(__file__).resolve().parents[2]
if str(COMBATBENCH_DIR) not in sys.path:
    sys.path.insert(0, str(COMBATBENCH_DIR))

from baseline.humanoid21.rewards import (
    BALANCE_SAFE_BACK_MARGIN,
    BALANCE_SAFE_FRONT_MARGIN,
    _compute_balance_value_terms,
)


def make_balance_output(
    lateral_signed_distance: float,
    lateral_velocity: float = 0.0,
    axis_projection: float = 0.1,
    support_span: float = 0.2,
    axis_velocity: float = 0.0,
) -> dict:
    return {
        "ground_support_frame_defined": True,
        "support_span": float(support_span),
        "support_axis_projection_coordinate": float(axis_projection),
        "support_lateral_signed_distance": float(lateral_signed_distance),
        "center_of_mass_velocity_along_support_axis": float(axis_velocity),
        "center_of_mass_velocity_along_support_lateral_axis": float(lateral_velocity),
    }


def _fmt(value: float) -> str:
    return f"{value:+.4f}"


def _eval(label: str, lateral_signed_distance: float, lateral_velocity: float = 0.0) -> float:
    out = make_balance_output(lateral_signed_distance, lateral_velocity=lateral_velocity)
    terms = _compute_balance_value_terms(out)
    score = terms["absolute_score"]
    print(
        f"{label:<40s}"
        f"signed={_fmt(lateral_signed_distance)} "
        f"v_lat={_fmt(lateral_velocity)}  "
        f"=>  front_dist={terms['front_distance']:.3f} "
        f"back_dist={terms['back_distance']:.3f} "
        f"score={_fmt(score)}"
    )
    return score


def main() -> None:
    print(f"SAFE_FRONT_MARGIN = {BALANCE_SAFE_FRONT_MARGIN:.3f} m")
    print(f"SAFE_BACK_MARGIN  = {BALANCE_SAFE_BACK_MARGIN:.3f} m")

    print("\n--- Inside safe zone (expect score == 1.0) ---")
    _eval("centered on feet line", 0.0)
    _eval("natural forward (5cm)", 0.05)
    _eval("natural forward (at front margin)", BALANCE_SAFE_FRONT_MARGIN)
    _eval("at back margin edge", -BALANCE_SAFE_BACK_MARGIN)

    print("\n--- Outside safe zone, same 5cm offset forward vs backward ---")
    score_front_5 = _eval("forward 5cm beyond margin", BALANCE_SAFE_FRONT_MARGIN + 0.05)
    score_back_5 = _eval("backward 5cm beyond margin", -(BALANCE_SAFE_BACK_MARGIN + 0.05))
    asymmetry_position = score_front_5 - score_back_5
    print(f"  -> backward penalty exceeds forward by {asymmetry_position:+.4f}")

    print("\n--- Outside safe zone, same 10cm offset forward vs backward ---")
    score_front_10 = _eval("forward 10cm beyond margin", BALANCE_SAFE_FRONT_MARGIN + 0.10)
    score_back_10 = _eval("backward 10cm beyond margin", -(BALANCE_SAFE_BACK_MARGIN + 0.10))
    print(f"  -> backward penalty exceeds forward by {score_front_10 - score_back_10:+.4f}")

    print("\n--- Dynamic: backward tilt + still vs backward tilt + falling further backward ---")
    _eval("back 5cm, still",           -(BALANCE_SAFE_BACK_MARGIN + 0.05), lateral_velocity=0.0)
    _eval("back 5cm, falling back @1m/s", -(BALANCE_SAFE_BACK_MARGIN + 0.05), lateral_velocity=-1.0)
    _eval("back 5cm, recovering @1m/s",   -(BALANCE_SAFE_BACK_MARGIN + 0.05), lateral_velocity=+1.0)

    print("\n--- Invalid support (lost ground frame) ---")
    bad = {"ground_support_frame_defined": False}
    print(f"invalid score = {_compute_balance_value_terms(bad)['absolute_score']:+.4f}")

    print("\n--- Assertions ---")
    assert score_back_5 < score_front_5, "Backward should be penalized more than forward at same offset"
    assert score_back_10 < score_front_10, "Backward penalty must dominate at larger offsets too"
    print("OK: backward penalty > forward penalty at matched offsets.")


if __name__ == "__main__":
    main()
