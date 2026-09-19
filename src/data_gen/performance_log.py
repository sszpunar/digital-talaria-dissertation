"""
performance_log.py
------------------
Digital Talaria – Performance Log Schema

Defines the PerformanceLog dataclass representing a single synthetic runner's
daily training entry across the 20-week marathon simulation. Each log entry
captures the artifact's recommendation, the runner's simulated adherence
response, and constraint compliance flags for evaluation against RQ1–RQ4.

Marathon phase schedule — 20 weeks (Base 6 / Build 6 / Peak 4 / Taper 4):
    Base  : Weeks  1–6   (aerobic foundation, easy volume accumulation)
    Build : Weeks  7–12  (tempo and progression work, volume increases)
    Peak  : Weeks 13–16  (race-specific long runs, highest volume)
    Taper : Weeks 17–20  (volume reduction, intensity preserved, race prep)

Named MARATHON_PHASE_SCHEDULE to distinguish from other goal-distance
phase schedules (5K, 10K, half marathon) that will be defined in PlanContext
as the commercial platform expands beyond the dissertation study window.

References:
    Esteve-Lanao et al. (2007) — periodization framework
    Daniels (2005)             — Daniels' Running Formula (marathon training)
    Noakes (2003)              — Lore of Running (marathon periodization)
    Banister et al. (1975)     — training load and phase structure
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

# ---------------------------------------------------------------------------
# Marathon periodization phase schedule — 20 weeks
# Base 6 / Build 6 / Peak 4 / Taper 4
# ---------------------------------------------------------------------------

MARATHON_PHASE_SCHEDULE: dict[int, str] = {
    1:  "base",
    2:  "base",
    3:  "base",
    4:  "base",
    5:  "base",
    6:  "base",
    7:  "build",
    8:  "build",
    9:  "build",
    10: "build",
    11: "build",
    12: "build",
    13: "peak",
    14: "peak",
    15: "peak",
    16: "peak",
    17: "taper",
    18: "taper",
    19: "taper",
    20: "taper",
}

# Backward-compatible alias — existing code referencing PHASE_SCHEDULE
# will continue to work during the migration period
PHASE_SCHEDULE = MARATHON_PHASE_SCHEDULE

SIMULATION_WEEKS = 20   # Total weeks in the marathon training plan

# ---------------------------------------------------------------------------
# Valid activity types
# ---------------------------------------------------------------------------

VALID_ACTIVITY_TYPES = {
    # -- Aerobic base --------------------------------------------------------
    "easy_run",           # Conversational pace, aerobic base building
    "recovery_run",       # Very easy effort post hard workout
    "long_run",           # Weekly peak distance effort

    # -- Intensity work ------------------------------------------------------
    "tempo_run",          # Sustained comfortably hard effort, lactate threshold
    "interval",           # Structured repetitions at high intensity with rest
    "fartlek",            # Unstructured speed play mixing efforts within a run
    "speed_workout",      # Short, fast repetitions targeting neuromuscular dev
    "progression_run",    # Run that gradually increases pace from easy to tempo

    # -- Strength and terrain ------------------------------------------------
    "incline_repeat",     # Treadmill incline intervals simulating hill repeats

    # -- Cross-modal and recovery --------------------------------------------
    "cross_train",        # Non-running aerobic activity (cycling, swimming)
    "strength_train",     # Resistance or functional strength session

    # -- Rest ----------------------------------------------------------------
    "rest",               # Full rest day, no activity
}

VALID_ADHERENCE_STATUSES = {"full", "partial", "skipped", "injury"}


# ---------------------------------------------------------------------------
# Performance log entry
# ---------------------------------------------------------------------------

@dataclass
class PerformanceLog:
    """
    A single daily training log entry for one synthetic runner.

    Recommendation fields capture what the artifact prescribed.
    Response fields capture the simulated runner's actual output.
    Compliance fields support RQ1–RQ4 evaluation.
    """

    # -- Metadata ------------------------------------------------------------
    runner_id:      str
    week_number:    int    # 1–20
    day_number:     int    # 1–7
    training_phase: str    # base | build | peak | taper
    log_id:         str    = field(default_factory=lambda: str(uuid.uuid4()))
    created_at:     str    = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )

    # -- Recommendation fields (artifact output) -----------------------------
    prescribed_activity_type:    str   = "easy_run"
    prescribed_distance_mi:      float = 0.0
    prescribed_duration_min:     float = 0.0
    prescribed_rpe_target:       int   = 5
    prescribed_weekly_volume_mi: float = 0.0

    # -- Simulated runner response fields ------------------------------------
    actual_distance_mi:  float = 0.0
    actual_duration_min: float = 0.0
    actual_rpe_reported: int   = 5
    adherence_status:    str   = "full"
    adherence_score:     float = 1.0

    # -- Constraint compliance fields (RQ1–RQ4) ------------------------------
    progressive_overload_compliant: bool  = True
    periodization_compliant:        bool  = True
    ten_percent_rule_compliant:     bool  = True
    weekly_volume_at_entry_mi:      float = 0.0

    # -- Disruption tracking -------------------------------------------------
    disruption_injury:     bool = False
    disruption_abstention: bool = False
    disruption_partial:    bool = False

    # -- Preference compliance -----------------------------------------------
    preference_compliant:         bool  = True   # plan respects runner preferences
    preference_alignment_score:   float = 1.0    # 0.0–1.0

    # -- Optional notes ------------------------------------------------------
    adaptation_notes: Optional[str] = None

    # -- Violation tracking --------------------------------------------------
    ten_percent_violation_count: int = 0

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        errors: List[str] = []

        if not (1 <= self.week_number <= SIMULATION_WEEKS):
            errors.append(
                f"week_number {self.week_number} outside valid range "
                f"[1, {SIMULATION_WEEKS}]."
            )

        if not (1 <= self.day_number <= 7):
            errors.append(
                f"day_number {self.day_number} outside valid range [1, 7]."
            )

        expected_phase = MARATHON_PHASE_SCHEDULE.get(self.week_number)
        if expected_phase and self.training_phase != expected_phase:
            errors.append(
                f"training_phase '{self.training_phase}' does not match "
                f"expected phase '{expected_phase}' for week {self.week_number}."
            )

        if self.prescribed_activity_type not in VALID_ACTIVITY_TYPES:
            errors.append(
                f"prescribed_activity_type '{self.prescribed_activity_type}' "
                f"is not a valid activity."
            )

        if self.adherence_status not in VALID_ADHERENCE_STATUSES:
            errors.append(
                f"adherence_status '{self.adherence_status}' is not valid. "
                f"Must be one of {VALID_ADHERENCE_STATUSES}."
            )

        if not (0.0 <= self.adherence_score <= 1.0):
            errors.append(
                f"adherence_score {self.adherence_score} outside [0.0, 1.0]."
            )

        if not (1 <= self.prescribed_rpe_target <= 10):
            errors.append(
                f"prescribed_rpe_target {self.prescribed_rpe_target} "
                f"outside valid range [1, 10]."
            )

        if not (1 <= self.actual_rpe_reported <= 10):
            errors.append(
                f"actual_rpe_reported {self.actual_rpe_reported} "
                f"outside valid range [1, 10]."
            )

        if errors:
            raise ValueError(
                f"PerformanceLog validation failed for runner "
                f"'{self.runner_id}' (Week {self.week_number}, "
                f"Day {self.day_number}):\n  " + "\n  ".join(errors)
            )

    # -- Convenience constructors --------------------------------------------

    @classmethod
    def rest_day(
        cls,
        runner_id:   str,
        week_number: int,
        day_number:  int,
    ) -> "PerformanceLog":
        """Create a validated rest day log entry."""
        return cls(
            runner_id=runner_id,
            week_number=week_number,
            day_number=day_number,
            training_phase=MARATHON_PHASE_SCHEDULE[week_number],
            prescribed_activity_type="rest",
            prescribed_distance_mi=0.0,
            prescribed_duration_min=0.0,
            prescribed_rpe_target=1,
            actual_distance_mi=0.0,
            actual_duration_min=0.0,
            actual_rpe_reported=1,
            adherence_status="full",
            adherence_score=1.0,
        )

    @classmethod
    def skipped_day(
        cls,
        runner_id:                   str,
        week_number:                 int,
        day_number:                  int,
        prescribed_activity_type:    str   = "easy_run",
        prescribed_distance_mi:      float = 0.0,
        prescribed_duration_min:     float = 0.0,
        prescribed_rpe_target:       int   = 5,
        prescribed_weekly_volume_mi: float = 0.0,
        weekly_volume_at_entry_mi:   float = 0.0,
    ) -> "PerformanceLog":
        """Create a validated skipped workout log entry."""
        return cls(
            runner_id=runner_id,
            week_number=week_number,
            day_number=day_number,
            training_phase=MARATHON_PHASE_SCHEDULE[week_number],
            prescribed_activity_type=prescribed_activity_type,
            prescribed_distance_mi=prescribed_distance_mi,
            prescribed_duration_min=prescribed_duration_min,
            prescribed_rpe_target=prescribed_rpe_target,
            prescribed_weekly_volume_mi=prescribed_weekly_volume_mi,
            actual_distance_mi=0.0,
            actual_duration_min=0.0,
            actual_rpe_reported=1,
            adherence_status="skipped",
            adherence_score=0.0,
            weekly_volume_at_entry_mi=weekly_volume_at_entry_mi,
        )