"""
profile_generator.py
--------------------
Digital Talaria – Synthetic Runner Profile Generator
200-Runner Marathon Cohort with Full Preference Schema

Generates 200 synthetic runner profiles for a full marathon training study
spanning 20 weeks (Base 6 / Build 6 / Peak 4 / Taper 4).

Cohort design:
    200 runners total
    5 profile types × 5 age divisions × 2 genders × 4 runners each
    = 5 × 5 × 2 × 4 = 200

    Profile types: PA, UW, OW, HS, SD
    Age divisions: 20-29, 30-39, 40-49, 50-59, 60-69
    Genders: F, M
    Runners per cell: 4 (numbered 01-04)

Runner ID format:
    "{age}-{gender}-{type}-{number:02d}"
    e.g. "24-F-PA-01", "31-M-HS-03"

Key changes from 50-runner 10K study:
    - 200 runners (was 50)
    - Marathon goal fields replace 10K goal fields
    - Full preference schema added (rest days, exclusions, intensity, schedule, goal)
    - Weekly mileage baselines scaled for marathon training loads
    - Training age ranges extended for marathon-capable runners
    - All runners generated programmatically (not hand-coded)

Performance context schema (marathon-oriented):
    prior_5k_time_min             : Optional[float] — nullable; not every runner has raced
    prior_10k_time_min            : Optional[float] — nullable; not every runner has raced
    prior_half_marathon_time_min  : Optional[float] — nullable; not every runner has raced
    prior_marathon_time_min       : Optional[float] — nullable; not every runner has raced
    goal_marathon_time_min        : float            — always set; physiologically plausible
                                                       finish target based on profile and VO2

    Population probability of having a prior race result varies by profile type:
        PA  — experienced; likely has full race history
        UW  — moderate history; may lack longer-distance results
        OW  — newer to racing; sparse history common
        HS  — moderate history; stress may have interrupted race seasons
        SD  — newer or intermittent; sparse history common

    Riegel power-law consistency is enforced only when both values in a pair
    are non-null. goal_marathon_time_min is not required to be faster than
    prior_marathon_time_min — aging, injury, and first-time marathoners make
    that constraint unrealistic.

Preference schema:
    Rest day preferences:
        preferred_rest_days_per_week  : int 0-3
        preferred_rest_day_slots      : List[int] subset of [1-7]
        flexible_rest_days            : bool

    Activity exclusions:
        excluded_activity_types       : List[str]
        excluded_reason               : str

    Intensity preferences:
        preferred_max_rpe             : int 5-10
        prefers_low_impact            : bool

    Schedule flexibility:
        available_days_per_week       : int 3-7
        preferred_long_run_day        : int 1-7
        preferred_workout_time        : str morning|midday|evening

    Goal context:
        goal_race_distance            : str (full vocabulary)
        target_finish_time            : float (minutes)
        experience_level              : str novice|intermediate|advanced

References:
    Tanaka et al. (2001)       — HRmax = 208 - 0.7 × age
    Riegel (1981)              — race time prediction
    Banister et al. (1975)     — training load
    Noakes (2003)              — Lore of Running (marathon training loads)
    Daniels (2005)             — Daniels' Running Formula
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Race distance vocabulary
# ---------------------------------------------------------------------------

RACE_DISTANCE_VOCAB = [
    # Track events
    "100m", "200m", "400m", "800m", "1500m", "mile",
    "3000m", "steeplechase", "5000m", "10000m",
    # Road events
    "5k", "8k", "10k", "15k", "10_mile", "20k",
    "half_marathon", "25k", "30k", "marathon",
    # Ultra
    "50k", "50_mile", "100k", "100_mile", "ultra_open",
    # Multi-sport
    "sprint_triathlon", "olympic_triathlon",
    "half_ironman", "ironman",
]

# ---------------------------------------------------------------------------
# Activity vocabulary (must match plan catalogue)
# ---------------------------------------------------------------------------

ACTIVITY_VOCAB = [
    "easy_run", "recovery_run", "long_run", "tempo_run",
    "interval", "fartlek", "speed_workout", "progression_run",
    "incline_repeat", "cross_train", "strength_train", "rest",
]

HIGH_IMPACT_ACTIVITIES = {"interval", "speed_workout", "incline_repeat"}

# ---------------------------------------------------------------------------
# Race history population probabilities by profile type
#
# Probability that a runner of this type has a recorded prior result at each
# distance. Longer distances are conditional on shorter ones being present —
# a runner cannot have a prior half time without a prior 10K time.
# ---------------------------------------------------------------------------

_RACE_HISTORY_PROBS: Dict[str, Dict[str, float]] = {
    "PA": {"5k": 0.95, "10k": 0.90, "half": 0.80, "full": 0.60},
    "UW": {"5k": 0.75, "10k": 0.65, "half": 0.45, "full": 0.25},
    "OW": {"5k": 0.55, "10k": 0.40, "half": 0.20, "full": 0.10},
    "HS": {"5k": 0.70, "10k": 0.60, "half": 0.40, "full": 0.20},
    "SD": {"5k": 0.55, "10k": 0.45, "half": 0.25, "full": 0.10},
}

# ---------------------------------------------------------------------------
# Riegel power-law race time prediction
# T2 = T1 × (D2 / D1) ^ 1.06
# ---------------------------------------------------------------------------

def _riegel(t1_min: float, d1_km: float, d2_km: float) -> float:
    """Predict race time at d2 given a known time t1 at d1 (Riegel, 1981)."""
    return round(t1_min * (d2_km / d1_km) ** 1.06, 1)


# ---------------------------------------------------------------------------
# Supporting dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MacroRatio:
    carb_pct:    int
    protein_pct: int
    fat_pct:     int

    def __post_init__(self) -> None:
        total = self.carb_pct + self.protein_pct + self.fat_pct
        if total != 100:
            raise ValueError(f"MacroRatio must sum to 100, got {total}")

    def to_dict(self) -> dict:
        return {
            "carb_pct":    self.carb_pct,
            "protein_pct": self.protein_pct,
            "fat_pct":     self.fat_pct,
        }


@dataclass
class RunnerPreferences:
    """
    Personal training preferences embedded in the runner profile.

    These preferences constrain which plans TFN and all other models
    can recommend. The plan catalogue is filtered against these preferences
    before any model sees the candidates — ensuring fair comparison across
    all thirteen models.
    """
    # Rest day preferences
    preferred_rest_days_per_week: int        # 0–3
    preferred_rest_day_slots:     List[int]  # subset of [1–7] (Mon=1, Sun=7)
    flexible_rest_days:           bool       # can rest day shift ±1?

    # Activity exclusions
    excluded_activity_types:      List[str]  # from ACTIVITY_VOCAB
    excluded_reason:              str        # injury_history|no_equipment|preference|medical

    # Intensity preferences
    preferred_max_rpe:            int        # 5–10 hard ceiling
    prefers_low_impact:           bool       # exclude interval, speed, incline

    # Schedule flexibility
    available_days_per_week:      int        # 3–7
    preferred_long_run_day:       int        # 1–7
    preferred_workout_time:       str        # morning|midday|evening

    # Goal context
    goal_race_distance:           str        # from RACE_DISTANCE_VOCAB
    target_finish_time:           float      # minutes (marathon finish target)
    experience_level:             str        # novice|intermediate|advanced

    def to_dict(self) -> dict:
        return {
            "preferred_rest_days_per_week": self.preferred_rest_days_per_week,
            "preferred_rest_day_slots":     self.preferred_rest_day_slots,
            "flexible_rest_days":           self.flexible_rest_days,
            "excluded_activity_types":      self.excluded_activity_types,
            "excluded_reason":              self.excluded_reason,
            "preferred_max_rpe":            self.preferred_max_rpe,
            "prefers_low_impact":           self.prefers_low_impact,
            "available_days_per_week":      self.available_days_per_week,
            "preferred_long_run_day":       self.preferred_long_run_day,
            "preferred_workout_time":       self.preferred_workout_time,
            "goal_race_distance":           self.goal_race_distance,
            "target_finish_time":           self.target_finish_time,
            "experience_level":             self.experience_level,
        }

    def effective_excluded_activities(self) -> List[str]:
        """
        Return the full set of excluded activities, combining explicit
        exclusions with low-impact preference.
        """
        excluded = set(self.excluded_activity_types)
        if self.prefers_low_impact:
            excluded |= HIGH_IMPACT_ACTIVITIES
        return sorted(excluded)


# ---------------------------------------------------------------------------
# RunnerProfile dataclass
# ---------------------------------------------------------------------------

@dataclass
class RunnerProfile:
    # -- Core identity -------------------------------------------------------
    runner_id:    str
    age:          int
    gender:       str
    profile_type: str

    # -- Anthropometric ------------------------------------------------------
    height_in:  int
    weight_lbs: float
    bmi:        float

    # -- Physiological -------------------------------------------------------
    resting_hr_bpm: int
    vo2_proxy:      float

    # -- Training history ----------------------------------------------------
    training_age_years:         float
    weekly_mileage_baseline_mi: float
    long_run_baseline_mi:       float

    # -- Performance context (marathon-oriented) -----------------------------
    # All four prior-time fields are nullable: not every runner has raced every
    # distance. goal_marathon_time_min is always set.
    prior_5k_time_min:            Optional[float]  # None if runner has not raced
    prior_10k_time_min:           Optional[float]  # None if runner has not raced
    prior_half_marathon_time_min: Optional[float]  # None if runner has not raced
    prior_marathon_time_min:      Optional[float]  # None if runner has not raced
    goal_marathon_time_min:       float            # always set

    # -- Lifestyle -----------------------------------------------------------
    sleep_hours_avg:     float
    sleep_quality_1to5:  int
    stress_level_1to5:   int
    hydration_oz_per_day: int
    macro_ratio:          MacroRatio

    # -- Biomechanical -------------------------------------------------------
    biomech_strength_1to5: int
    flexibility_1to5:      int
    joint_mobility_1to5:   int

    # -- Contextual ----------------------------------------------------------
    terrain_preference:     str
    schedule_constraints:   str
    device_use:             str
    injury_history:         List[str]
    adherence_risk_factors: List[str]
    motivation_notes:       str

    # -- Preference schema ---------------------------------------------------
    preferences: RunnerPreferences

    # -- Derived (computed in __post_init__) ---------------------------------
    est_hr_max_bpm:    int = field(init=False)
    est_hr_max_method: str = field(init=False, default="Tanaka")
    age_division:      str = field(init=False)
    is_pilot:          bool = False

    # -- Wearable device source ----------------------------------------------
    # None = no device; runner relies on manual log entries.
    # Must be a value from wearable_data.SUPPORTED_SOURCES or None.
    wearable_source: Optional[str] = None

    def __post_init__(self) -> None:
        self.est_hr_max_bpm    = round(208 - 0.7 * self.age)
        self.est_hr_max_method = "Tanaka"
        self.age_division      = self._resolve_age_division()
        self._validate()

    def _resolve_age_division(self) -> str:
        for div in ["20-29", "30-39", "40-49", "50-59", "60-69"]:
            lo, hi = map(int, div.split("-"))
            if lo <= self.age <= hi:
                return div
        raise ValueError(f"Age {self.age} outside supported divisions.")

    def _validate(self) -> None:
        errors: List[str] = []

        # BMI range by profile type
        bmi_ranges = {
            "PA": (19.0, 24.5), "UW": (16.0, 18.9),
            "OW": (27.0, 34.0), "HS": (19.0, 28.0), "SD": (19.0, 27.0),
        }
        lo, hi = bmi_ranges.get(self.profile_type, (15.0, 40.0))
        if not (lo <= self.bmi <= hi):
            errors.append(
                f"BMI {self.bmi} out of range [{lo}, {hi}] "
                f"for profile type {self.profile_type}"
            )

        # Long run should be 20–35% of weekly volume
        ratio = self.long_run_baseline_mi / max(self.weekly_mileage_baseline_mi, 1.0)
        if not (0.18 <= ratio <= 0.38):
            errors.append(
                f"Long run {self.long_run_baseline_mi}mi is {ratio:.0%} "
                f"of weekly {self.weekly_mileage_baseline_mi}mi "
                f"(expected 20–35%)"
            )

        # Riegel consistency — only validated when both values in a pair are non-null.
        # goal_marathon_time_min is explicitly NOT validated against prior_marathon_time_min:
        # aging, injury, and first-time marathoners make improvement unrealistic to require.
        _riegel_pairs: List[Tuple[Optional[float], Optional[float], float, float, str]] = [
            (self.prior_5k_time_min,  self.prior_10k_time_min,           5.0,  10.0, "5K→10K"),
            (self.prior_10k_time_min, self.prior_half_marathon_time_min, 10.0, 21.1, "10K→half"),
            (self.prior_half_marathon_time_min, self.prior_marathon_time_min, 21.1, 42.2, "half→marathon"),
        ]
        for t1, t2, d1, d2, label in _riegel_pairs:
            if t1 is not None and t2 is not None:
                predicted = _riegel(t1, d1, d2)
                tolerance = predicted * 0.15  # ±15% for real-world variation
                if not (predicted - tolerance <= t2 <= predicted + tolerance):
                    errors.append(
                        f"Riegel {label}: recorded {t2:.1f} min inconsistent with "
                        f"predicted {predicted:.1f} min (±15%) from {t1:.1f} min"
                    )

        # Profile-specific lifestyle checks
        if self.profile_type == "HS" and self.stress_level_1to5 < 4:
            errors.append(
                f"HS runner stress={self.stress_level_1to5} (expected ≥4)"
            )
        if self.profile_type == "SD":
            if self.sleep_hours_avg > 5.5:
                errors.append(
                    f"SD runner sleep={self.sleep_hours_avg}h (expected ≤5.5)"
                )
            if self.sleep_quality_1to5 > 2:
                errors.append(
                    f"SD runner sleep_quality={self.sleep_quality_1to5} (expected ≤2)"
                )

        # Resting HR — floor 38 bpm to accommodate trained endurance athletes
        if not (38 <= self.resting_hr_bpm <= 90):
            errors.append(
                f"resting_hr_bpm={self.resting_hr_bpm} outside [38, 90]"
            )

        # Preference validation
        if not (5 <= self.preferences.preferred_max_rpe <= 10):
            errors.append(
                f"preferred_max_rpe={self.preferences.preferred_max_rpe} outside [5, 10]"
            )
        if not (3 <= self.preferences.available_days_per_week <= 7):
            errors.append(
                f"available_days_per_week={self.preferences.available_days_per_week} "
                f"outside [3, 7]"
            )

        if errors:
            raise ValueError(
                f"RunnerProfile validation failed for '{self.runner_id}':\n"
                + "\n".join(f"  {e}" for e in errors)
            )

    def to_dict(self) -> dict:
        return {
            "runner_id":                    self.runner_id,
            "age":                          self.age,
            "gender":                       self.gender,
            "profile_type":                 self.profile_type,
            "age_division":                 self.age_division,
            "height_in":                    self.height_in,
            "weight_lbs":                   self.weight_lbs,
            "bmi":                          self.bmi,
            "resting_hr_bpm":               self.resting_hr_bpm,
            "est_hr_max_bpm":               self.est_hr_max_bpm,
            "est_hr_max_method":            self.est_hr_max_method,
            "vo2_proxy":                    self.vo2_proxy,
            "training_age_years":           self.training_age_years,
            "weekly_mileage_baseline_mi":   self.weekly_mileage_baseline_mi,
            "long_run_baseline_mi":         self.long_run_baseline_mi,
            "prior_5k_time_min":            self.prior_5k_time_min,
            "prior_10k_time_min":           self.prior_10k_time_min,
            "prior_half_marathon_time_min": self.prior_half_marathon_time_min,
            "prior_marathon_time_min":      self.prior_marathon_time_min,
            "goal_marathon_time_min":       self.goal_marathon_time_min,
            "sleep_hours_avg":              self.sleep_hours_avg,
            "sleep_quality_1to5":           self.sleep_quality_1to5,
            "stress_level_1to5":            self.stress_level_1to5,
            "hydration_oz_per_day":         self.hydration_oz_per_day,
            "macro_ratio":                  self.macro_ratio.to_dict(),
            "biomech_strength_1to5":        self.biomech_strength_1to5,
            "flexibility_1to5":             self.flexibility_1to5,
            "joint_mobility_1to5":          self.joint_mobility_1to5,
            "terrain_preference":           self.terrain_preference,
            "schedule_constraints":         self.schedule_constraints,
            "device_use":                   self.device_use,
            "injury_history":               self.injury_history,
            "adherence_risk_factors":       self.adherence_risk_factors,
            "motivation_notes":             self.motivation_notes,
            "preferences":                  self.preferences.to_dict(),
            "is_pilot":                     self.is_pilot,
            "wearable_source":              self.wearable_source,
        }


# ---------------------------------------------------------------------------
# Profile generation parameters by type
# ---------------------------------------------------------------------------

_PROFILE_PARAMS: Dict[str, dict] = {
    "PA": {
        "bmi":                          (19.5, 24.0),
        "resting_hr_bpm":               (44,   56),
        "vo2_proxy":                    (52.0, 65.0),
        "training_age_years":           (3.0,  12.0),
        "weekly_mileage_baseline_mi":   (30.0, 55.0),
        "sleep_hours_avg":              (7.0,  8.5),
        "sleep_quality_1to5":           (3,    5),
        "stress_level_1to5":            (1,    3),
        "biomech_strength_1to5":        (4,    5),
        "flexibility_1to5":             (3,    5),
        "joint_mobility_1to5":          (3,    5),
        "preferred_max_rpe":            (7,    10),
        "available_days_per_week":      (5,    7),
        "experience_level":             ["intermediate", "advanced"],
        "excluded_reason":              "preference",
        "preferred_rest_days_per_week": (1,    2),
    },
    "UW": {
        "bmi":                          (16.5, 18.8),
        "resting_hr_bpm":               (54,   70),
        "vo2_proxy":                    (40.0, 52.0),
        "training_age_years":           (1.0,  6.0),
        "weekly_mileage_baseline_mi":   (18.0, 32.0),
        "sleep_hours_avg":              (6.5,  8.0),
        "sleep_quality_1to5":           (2,    4),
        "stress_level_1to5":            (2,    4),
        "biomech_strength_1to5":        (2,    4),
        "flexibility_1to5":             (3,    5),
        "joint_mobility_1to5":          (2,    4),
        "preferred_max_rpe":            (6,    8),
        "available_days_per_week":      (4,    6),
        "experience_level":             ["novice", "intermediate"],
        "excluded_reason":              "medical",
        "preferred_rest_days_per_week": (1,    2),
    },
    "OW": {
        "bmi":                          (27.0, 33.5),
        "resting_hr_bpm":               (65,   82),
        "vo2_proxy":                    (28.0, 42.0),
        "training_age_years":           (0.5,  5.0),
        "weekly_mileage_baseline_mi":   (12.0, 25.0),
        "sleep_hours_avg":              (6.0,  7.5),
        "sleep_quality_1to5":           (2,    4),
        "stress_level_1to5":            (2,    4),
        "biomech_strength_1to5":        (2,    3),
        "flexibility_1to5":             (2,    3),
        "joint_mobility_1to5":          (2,    3),
        "preferred_max_rpe":            (5,    7),
        "available_days_per_week":      (3,    5),
        "experience_level":             ["novice", "intermediate"],
        "excluded_reason":              "medical",
        "preferred_rest_days_per_week": (2,    3),
    },
    "HS": {
        "bmi":                          (20.0, 27.5),
        "resting_hr_bpm":               (60,   78),
        "vo2_proxy":                    (38.0, 52.0),
        "training_age_years":           (2.0,  8.0),
        "weekly_mileage_baseline_mi":   (15.0, 30.0),
        "sleep_hours_avg":              (5.5,  7.0),
        "sleep_quality_1to5":           (1,    3),
        "stress_level_1to5":            (4,    5),
        "biomech_strength_1to5":        (2,    4),
        "flexibility_1to5":             (2,    4),
        "joint_mobility_1to5":          (2,    4),
        "preferred_max_rpe":            (5,    8),
        "available_days_per_week":      (3,    5),
        "experience_level":             ["novice", "intermediate"],
        "excluded_reason":              "preference",
        "preferred_rest_days_per_week": (1,    3),
    },
    "SD": {
        "bmi":                          (20.0, 26.5),
        "resting_hr_bpm":               (62,   78),
        "vo2_proxy":                    (32.0, 46.0),
        "training_age_years":           (0.5,  5.0),
        "weekly_mileage_baseline_mi":   (10.0, 22.0),
        "sleep_hours_avg":              (3.5,  5.0),
        "sleep_quality_1to5":           (1,    2),
        "stress_level_1to5":            (2,    4),
        "biomech_strength_1to5":        (2,    3),
        "flexibility_1to5":             (2,    3),
        "joint_mobility_1to5":          (2,    3),
        "preferred_max_rpe":            (5,    7),
        "available_days_per_week":      (3,    5),
        "experience_level":             ["novice", "intermediate"],
        "excluded_reason":              "preference",
        "preferred_rest_days_per_week": (1,    3),
    },
}

# Age-based adjustments applied on top of profile type parameters
_AGE_ADJUSTMENTS: Dict[str, Dict[str, float]] = {
    "20-29": {"vo2_adj": +5.0, "hr_adj": -2, "strength_adj": +1},
    "30-39": {"vo2_adj": +2.0, "hr_adj":  0, "strength_adj":  0},
    "40-49": {"vo2_adj":  0.0, "hr_adj": +2, "strength_adj":  0},
    "50-59": {"vo2_adj": -3.0, "hr_adj": +3, "strength_adj": -1},
    "60-69": {"vo2_adj": -6.0, "hr_adj": +5, "strength_adj": -1},
}

# Gender-based adjustments
_GENDER_MILEAGE_OFFSETS: Dict[str, float] = {"F": -2.0, "M": +2.0}
_GENDER_VO2_OFFSETS:     Dict[str, float] = {"F": -3.0, "M": +3.0}

# Terrain preferences by profile type
_TERRAIN_OPTIONS: Dict[str, List[str]] = {
    "PA": ["road", "trail", "mixed"],
    "UW": ["road", "mixed"],
    "OW": ["road"],
    "HS": ["road", "mixed"],
    "SD": ["road"],
}

# Device use by profile type
_DEVICE_OPTIONS: Dict[str, List[str]] = {
    "PA": ["watch", "watch", "hr_strap"],
    "UW": ["watch", "hr_strap", "none"],
    "OW": ["watch", "hr_strap", "none"],
    "HS": ["watch", "watch", "hr_strap"],
    "SD": ["hr_strap", "none", "watch"],
}

# Wearable device source — distribution reflects 2026 market share
_WEARABLE_OPTIONS: Dict[str, List[Optional[str]]] = {
    "PA": ["garmin_connect", "garmin_connect", "apple_healthkit",
           "apple_healthkit", "whoop", None],
    "UW": ["apple_healthkit", "garmin_connect", "oura_ring",
           "apple_healthkit", None, None],
    "OW": ["apple_healthkit", "apple_healthkit", "oura_ring",
           "google_health_connect", None, None],
    "HS": ["whoop", "whoop", "apple_healthkit",
           "garmin_connect", "oura_ring", None],
    "SD": ["oura_ring", "oura_ring", "apple_healthkit",
           "whoop", None, None],
}

# Activity exclusions by profile type (cycled by runner number)
_EXCLUSION_SETS: Dict[str, List[List[str]]] = {
    "PA": [[], [], ["strength_train"], ["cross_train"]],
    "UW": [["incline_repeat"], ["speed_workout"], ["interval", "speed_workout"], []],
    "OW": [["interval", "speed_workout", "incline_repeat"],
           ["speed_workout", "incline_repeat"],
           ["interval", "incline_repeat"],
           ["speed_workout"]],
    "HS": [["speed_workout"], ["interval"], ["incline_repeat", "speed_workout"], []],
    "SD": [["speed_workout", "interval"],
           ["incline_repeat"],
           ["speed_workout"],
           ["interval", "speed_workout", "incline_repeat"]],
}

# Marathon finish time targets by experience level (minutes)
_MARATHON_TARGETS: Dict[str, Dict[str, float]] = {
    "advanced":     {"F": 210.0, "M": 190.0},  # ~3:30 / ~3:10
    "intermediate": {"F": 255.0, "M": 235.0},  # ~4:15 / ~3:55
    "novice":       {"F": 320.0, "M": 295.0},  # ~5:20 / ~4:55
}

# Narrative templates
_SCHEDULE_TEMPLATES: Dict[str, List[str]] = {
    "PA": [
        "Full-time professional; trains 6am before work.",
        "Graduate student with flexible morning schedule.",
        "Remote worker; trains midday.",
        "Teacher; trains immediately after school.",
    ],
    "UW": [
        "Works irregular hours; trains when energy allows.",
        "Freelancer; trains midday most days.",
        "College student; trains between classes.",
        "Part-time worker; trains mornings.",
    ],
    "OW": [
        "Desk job; walks at lunch and runs evenings.",
        "Physician-recommended exercise; trains mornings.",
        "Family commitments; trains early before household wakes.",
        "Shift worker; trains on days off.",
    ],
    "HS": [
        "Demanding professional role; runs at lunch.",
        "Business owner; trains before 6am.",
        "New parent; trains during nap windows.",
        "Caregiver role; trains when coverage available.",
    ],
    "SD": [
        "Parent of young children; trains in fragmented windows.",
        "Night shift nurse; trains post-shift before sleep.",
        "Startup employee; deprioritizes sleep for work.",
        "Graduate student; stays up late writing.",
    ],
}

_MOTIVATION_TEMPLATES: Dict[str, List[str]] = {
    "PA": [
        "Training to qualify for Boston Marathon.",
        "First marathon after several half marathons.",
        "Chasing a personal best at the city marathon.",
        "Supporting a charity through marathon fundraising.",
    ],
    "UW": [
        "Marathon is a strength-building milestone.",
        "Completing a marathon to prove endurance capacity.",
        "Training with a nutrition coach toward first marathon.",
        "Building aerobic base before marathon attempt.",
    ],
    "OW": [
        "Marathon goal set with physician as health milestone.",
        "Completing a marathon for weight management.",
        "First marathon to prove ability after weight loss journey.",
        "Running the marathon with a family member.",
    ],
    "HS": [
        "Marathon is a stress management anchor.",
        "Running keeps them grounded during a high-demand period.",
        "Marathon training structured around demanding career.",
        "Using marathon prep as a reason to protect recovery time.",
    ],
    "SD": [
        "Marathon goal to reclaim personal identity.",
        "Completing a marathon before next life stage.",
        "Running the marathon for a cause close to them.",
        "First marathon as a symbol of resilience.",
    ],
}

_INJURY_TEMPLATES: Dict[str, List[List[str]]] = {
    "PA": [[], ["mild shin splints resolved"], ["IT band tightness managed"]],
    "UW": [["stress fracture history managed"], ["mild anemia treated"], []],
    "OW": [["knee osteoarthritis mild"], ["plantar fasciitis managed"],
           ["lower back pain ongoing"]],
    "HS": [["plantar fasciitis managed"], ["Achilles tendinopathy"],
           ["neck tension from stress"]],
    "SD": [["shin splints intermittent"], [], ["hip flexor tightness"]],
}

_ADHERENCE_RISK_TEMPLATES: Dict[str, List[List[str]]] = {
    "PA": [["high travel frequency"], ["competition anxiety"],
           ["occasional overtraining risk"]],
    "UW": [["low energy availability risk"], ["nutrition deficit risk"],
           ["bone stress risk"]],
    "OW": [["joint load management"], ["heat sensitivity"],
           ["pacing consistency"]],
    "HS": [["cortisol-driven fatigue"], ["schedule disruption"],
           ["emotional stress load"]],
    "SD": [["sleep fragmentation"], ["energy variability"],
           ["recovery deficit"]],
}


# ---------------------------------------------------------------------------
# Preference generator
# ---------------------------------------------------------------------------

def _generate_preferences(
    profile_type:     str,
    gender:           str,
    experience_level: str,
    rng:              random.Random,
    number:           int,
) -> RunnerPreferences:
    """
    Generate a RunnerPreferences object for a given runner.

    Preferences are systematically varied by profile type, with controlled
    randomisation to ensure population-level diversity within each type.

    Args:
        profile_type:     PA | UW | OW | HS | SD
        gender:           F | M
        experience_level: novice | intermediate | advanced
        rng:              Seeded random generator
        number:           Runner number within the cell (1–4)

    Returns:
        RunnerPreferences object.
    """
    params = _PROFILE_PARAMS[profile_type]

    # Rest day preferences
    rest_min, rest_max = params["preferred_rest_days_per_week"]
    rest_days = rng.randint(rest_min, rest_max)
    rest_slots_pool = [1, 3, 5, 7]
    rest_slots = sorted(
        rng.sample(rest_slots_pool, min(rest_days, len(rest_slots_pool)))
    )
    if not rest_slots:
        rest_slots = [7]
    flexible_rest = profile_type in ("PA", "HS") and rng.random() > 0.4

    # Activity exclusions
    excluded = list(_EXCLUSION_SETS[profile_type][(number - 1) % 4])
    low_impact = profile_type in ("OW", "UW") or (
        profile_type == "SD" and rng.random() > 0.6
    )
    if low_impact:
        excluded = sorted(set(excluded) | HIGH_IMPACT_ACTIVITIES)
    excluded_reason = params["excluded_reason"]

    # Intensity preferences
    rpe_min, rpe_max = params["preferred_max_rpe"]
    max_rpe = rng.randint(rpe_min, rpe_max)

    # Schedule flexibility
    days_min, days_max = params["available_days_per_week"]
    avail_days = rng.randint(days_min, days_max)

    # Long run day — PA/UW prefer weekend, others more variable
    if profile_type in ("PA", "UW"):
        long_run_day = rng.choice([6, 7])
    else:
        long_run_day = rng.choice([5, 6, 7])

    # Workout time by profile type
    _time_prefs: Dict[str, List[str]] = {
        "PA": ["morning", "morning", "midday"],
        "UW": ["morning", "midday", "morning"],
        "OW": ["morning", "evening", "midday"],
        "HS": ["morning", "morning", "midday"],
        "SD": ["midday", "evening", "morning"],
    }
    workout_time = rng.choice(_time_prefs[profile_type])

    # Target finish time: population-level anchor ± individual variation
    base_target = _MARATHON_TARGETS[experience_level][gender]
    target_time = round(base_target + rng.uniform(-15.0, 15.0), 1)

    return RunnerPreferences(
        preferred_rest_days_per_week=rest_days,
        preferred_rest_day_slots=rest_slots,
        flexible_rest_days=flexible_rest,
        excluded_activity_types=sorted(excluded),
        excluded_reason=excluded_reason,
        preferred_max_rpe=max_rpe,
        prefers_low_impact=low_impact,
        available_days_per_week=avail_days,
        preferred_long_run_day=long_run_day,
        preferred_workout_time=workout_time,
        goal_race_distance="marathon",
        target_finish_time=target_time,
        experience_level=experience_level,
    )


# ---------------------------------------------------------------------------
# Profile generator
# ---------------------------------------------------------------------------

def _generate_profile(
    profile_type: str,
    age_division: str,
    gender:       str,
    number:       int,
    rng:          random.Random,
) -> RunnerProfile:
    """
    Generate a single RunnerProfile using systematic parameterisation
    with controlled randomisation.

    Args:
        profile_type: PA | UW | OW | HS | SD
        age_division: 20-29 | 30-39 | 40-49 | 50-59 | 60-69
        gender:       F | M
        number:       Runner number within the cell (1–4)
        rng:          Seeded random generator for reproducibility

    Returns:
        Validated RunnerProfile object.
    """
    params  = _PROFILE_PARAMS[profile_type]
    age_adj = _AGE_ADJUSTMENTS[age_division]

    # Age — spread across the decade by runner number
    lo_age, hi_age = map(int, age_division.split("-"))
    age = min(lo_age + (number - 1) * 2 + rng.randint(0, 1), hi_age)

    # BMI and anthropometrics
    bmi_lo, bmi_hi = params["bmi"]
    bmi = round(rng.uniform(bmi_lo, bmi_hi), 1)
    height_in  = rng.randint(62, 66) if gender == "F" else rng.randint(66, 72)
    height_m   = height_in * 0.0254
    weight_kg  = bmi * (height_m ** 2)
    weight_lbs = round(weight_kg * 2.20462, 1)
    bmi        = round(weight_kg / (height_m ** 2), 1)  # recompute for accuracy

    # Resting HR
    hr_lo, hr_hi = params["resting_hr_bpm"]
    hr_lo    = max(44, hr_lo + age_adj["hr_adj"])
    hr_hi    = min(82, hr_hi + age_adj["hr_adj"])
    resting_hr = rng.randint(hr_lo, hr_hi)

    # VO2 proxy
    vo2_lo, vo2_hi = params["vo2_proxy"]
    vo2_lo += age_adj["vo2_adj"] + _GENDER_VO2_OFFSETS[gender]
    vo2_hi += age_adj["vo2_adj"] + _GENDER_VO2_OFFSETS[gender]
    vo2 = round(rng.uniform(max(18.0, vo2_lo), max(20.0, vo2_hi)), 1)

    # Training age
    ta_lo, ta_hi = params["training_age_years"]
    training_age = round(rng.uniform(ta_lo, ta_hi), 1)

    # Weekly and long run mileage
    mi_lo, mi_hi = params["weekly_mileage_baseline_mi"]
    mi_lo   += _GENDER_MILEAGE_OFFSETS[gender]
    mi_hi   += _GENDER_MILEAGE_OFFSETS[gender]
    weekly_mi   = round(rng.uniform(mi_lo, mi_hi), 1)
    long_run_mi = round(weekly_mi * rng.uniform(0.22, 0.30), 1)

    # Experience level derived from training age and VO2
    if training_age >= 5.0 and vo2 >= 50.0:
        experience = "advanced"
    elif training_age >= 2.0 and vo2 >= 38.0:
        experience = "intermediate"
    else:
        experience = "novice"
    # Cap experience ceiling by profile type
    if profile_type in ("OW", "UW", "SD") and experience == "advanced":
        experience = "intermediate"

    # Goal marathon time — physiologically plausible anchor ± individual variation
    goal_marathon_time_min = round(
        _MARATHON_TARGETS[experience][gender] + rng.uniform(-20.0, 20.0), 1
    )

    # ---------------------------------------------------------------------------
    # Race history — nullable prior times with population probabilities
    #
    # Each distance is conditional on the shorter distance being present.
    # Times are anchored to a VO2-derived 5K proxy and propagated via Riegel.
    # Small noise is added at each distance to reflect real-world variation.
    # ---------------------------------------------------------------------------
    probs = _RACE_HISTORY_PROBS[profile_type]

    # VO2-derived 5K anchor (used even if prior_5k_time_min ends up null,
    # to propagate consistent Riegel estimates for longer distances)
    _base_5k = round(max(17.0, 60.0 - vo2 * 0.70 + rng.uniform(-1.5, 1.5)), 1)

    prior_5k_time_min: Optional[float] = (
        _base_5k
        if rng.random() < probs["5k"]
        else None
    )
    prior_10k_time_min: Optional[float] = (
        round(_riegel(_base_5k, 5.0, 10.0) + rng.uniform(-1.0, 1.0), 1)
        if prior_5k_time_min is not None and rng.random() < probs["10k"]
        else None
    )
    prior_half_marathon_time_min: Optional[float] = (
        round(_riegel(_base_5k, 5.0, 21.1) + rng.uniform(-3.0, 3.0), 1)
        if prior_10k_time_min is not None and rng.random() < probs["half"]
        else None
    )
    prior_marathon_time_min: Optional[float] = (
        round(_riegel(_base_5k, 5.0, 42.2) + rng.uniform(-5.0, 5.0), 1)
        if prior_half_marathon_time_min is not None and rng.random() < probs["full"]
        else None
    )

    # Lifestyle
    sl_lo, sl_hi = params["sleep_hours_avg"]
    sleep_hours  = round(rng.uniform(sl_lo, sl_hi), 1)
    sq_lo, sq_hi = params["sleep_quality_1to5"]
    sleep_quality = rng.randint(sq_lo, sq_hi)
    st_lo, st_hi  = params["stress_level_1to5"]
    stress        = rng.randint(st_lo, st_hi)
    hydration     = rng.randint(60, 100)

    # Macro ratio
    if profile_type == "PA":
        carb  = rng.choice([55, 58, 60])
        macro = MacroRatio(carb, 25, 100 - carb - 25)
    elif profile_type == "UW":
        prot  = rng.choice([20, 22, 23])
        macro = MacroRatio(50, prot, 100 - 50 - prot)
    else:
        macro = MacroRatio(50, 25, 25)

    # Biomechanical
    bio_lo, bio_hi = params["biomech_strength_1to5"]
    bio_str = int(min(5, max(1, rng.randint(bio_lo, bio_hi) + age_adj["strength_adj"])))
    flex_lo, flex_hi = params["flexibility_1to5"]
    flex = rng.randint(flex_lo, flex_hi)
    mob_lo, mob_hi = params["joint_mobility_1to5"]
    mob  = rng.randint(mob_lo, mob_hi)

    # Contextual
    terrain         = rng.choice(_TERRAIN_OPTIONS[profile_type])
    device          = rng.choice(_DEVICE_OPTIONS[profile_type])
    wearable_source = rng.choice(_WEARABLE_OPTIONS[profile_type])
    schedule        = _SCHEDULE_TEMPLATES[profile_type][(number - 1) % 4]
    motivate        = _MOTIVATION_TEMPLATES[profile_type][(number - 1) % 4]
    injury          = rng.choice(_INJURY_TEMPLATES[profile_type])
    adh_risk        = rng.choice(_ADHERENCE_RISK_TEMPLATES[profile_type])

    runner_id = f"{age}-{gender}-{profile_type}-{number:02d}"
    prefs     = _generate_preferences(profile_type, gender, experience, rng, number)

    return RunnerProfile(
        runner_id=runner_id,
        age=age,
        gender=gender,
        profile_type=profile_type,
        height_in=height_in,
        weight_lbs=weight_lbs,
        bmi=bmi,
        resting_hr_bpm=resting_hr,
        vo2_proxy=vo2,
        training_age_years=training_age,
        weekly_mileage_baseline_mi=weekly_mi,
        long_run_baseline_mi=long_run_mi,
        prior_5k_time_min=prior_5k_time_min,
        prior_10k_time_min=prior_10k_time_min,
        prior_half_marathon_time_min=prior_half_marathon_time_min,
        prior_marathon_time_min=prior_marathon_time_min,
        goal_marathon_time_min=goal_marathon_time_min,
        sleep_hours_avg=sleep_hours,
        sleep_quality_1to5=sleep_quality,
        stress_level_1to5=stress,
        hydration_oz_per_day=hydration,
        macro_ratio=macro,
        biomech_strength_1to5=bio_str,
        flexibility_1to5=flex,
        joint_mobility_1to5=mob,
        terrain_preference=terrain,
        schedule_constraints=schedule,
        device_use=device,
        wearable_source=wearable_source,
        injury_history=list(injury),
        adherence_risk_factors=list(adh_risk),
        motivation_notes=motivate,
        preferences=prefs,
    )


# ---------------------------------------------------------------------------
# Build all 200 profiles
# ---------------------------------------------------------------------------

PROFILE_TYPES    = ["PA", "UW", "OW", "HS", "SD"]
AGE_DIVISIONS    = ["20-29", "30-39", "40-49", "50-59", "60-69"]
GENDERS          = ["F", "M"]
RUNNERS_PER_CELL = 4   # 5 types × 5 divisions × 2 genders × 4 = 200
GENERATION_SEED  = 42


def build_profiles() -> List[RunnerProfile]:
    """
    Generate all 200 RunnerProfile objects.

    Uses a seeded random generator for full reproducibility.
    Ordering: age_division × gender × profile_type × number

    Returns:
        List of 200 validated RunnerProfile objects.
    """
    profiles: List[RunnerProfile] = []
    rng = random.Random(GENERATION_SEED)

    for age_div in AGE_DIVISIONS:
        for gender in GENDERS:
            for profile_type in PROFILE_TYPES:
                for number in range(1, RUNNERS_PER_CELL + 1):
                    profiles.append(_generate_profile(
                        profile_type=profile_type,
                        age_division=age_div,
                        gender=gender,
                        number=number,
                        rng=rng,
                    ))

    assert len(profiles) == 200, f"Expected 200 profiles, got {len(profiles)}"
    return profiles


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = os.path.join("data", "profiles")
DEFAULT_JSON_PATH  = os.path.join(DEFAULT_OUTPUT_DIR, "profiles.json")
DEFAULT_CSV_PATH   = os.path.join(DEFAULT_OUTPUT_DIR, "profiles.csv")


def save_profiles(
    profiles:  List[RunnerProfile],
    json_path: str = DEFAULT_JSON_PATH,
    csv_path:  str = DEFAULT_CSV_PATH,
) -> None:
    """Save profiles to JSON and CSV."""
    import csv

    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([p.to_dict() for p in profiles], f, indent=2)
    print(f"[OK] Profiles JSON saved -> {json_path} ({len(profiles)} profiles)")

    if profiles:
        flat_rows = []
        for p in profiles:
            d     = p.to_dict()
            prefs = d.pop("preferences")
            macro = d.pop("macro_ratio")
            row   = {
                **d,
                **{f"macro_{k}": v for k, v in macro.items()},
                **{f"pref_{k}":  v for k, v in prefs.items()},
            }
            for key in list(row.keys()):
                if isinstance(row[key], list):
                    row[key] = "|".join(str(x) for x in row[key])
            flat_rows.append(row)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=flat_rows[0].keys())
            writer.writeheader()
            writer.writerows(flat_rows)
        print(f"[OK] Profiles CSV saved  -> {csv_path}")


# ---------------------------------------------------------------------------
# Profile loader
# ---------------------------------------------------------------------------

def load_profiles(json_path: str = DEFAULT_JSON_PATH) -> List[RunnerProfile]:
    """
    Load profiles from JSON and reconstruct RunnerProfile objects.

    Includes migration for profiles saved under the pre-marathon field names:
        prior_half_marathon_time  → prior_half_marathon_time_min
        goal_marathon_time        → goal_marathon_time_min
    Missing nullable fields default to None.

    Args:
        json_path: Path to profiles.json

    Returns:
        List of validated RunnerProfile objects.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"profiles.json not found at '{json_path}'. "
            "Run profile_generator.py first."
        )

    with open(json_path, encoding="utf-8") as f:
        raw = json.load(f)

    profiles = []
    for d in raw:
        prefs_d = d.pop("preferences")
        macro_d = d.pop("macro_ratio")

        # Migrate legacy field names from pre-marathon schema
        if "prior_half_marathon_time" in d:
            d["prior_half_marathon_time_min"] = d.pop("prior_half_marathon_time")
        if "goal_marathon_time" in d:
            d["goal_marathon_time_min"] = d.pop("goal_marathon_time")

        # Default nullable fields absent from older saved files
        d.setdefault("prior_10k_time_min",           None)
        d.setdefault("prior_marathon_time_min",       None)
        d.setdefault("prior_half_marathon_time_min",  None)

        # Remove derived fields — recomputed in __post_init__
        d.pop("est_hr_max_bpm",    None)
        d.pop("est_hr_max_method", None)
        d.pop("age_division",      None)

        prefs = RunnerPreferences(
            preferred_rest_days_per_week=prefs_d["preferred_rest_days_per_week"],
            preferred_rest_day_slots=    prefs_d["preferred_rest_day_slots"],
            flexible_rest_days=          prefs_d["flexible_rest_days"],
            excluded_activity_types=     prefs_d["excluded_activity_types"],
            excluded_reason=             prefs_d["excluded_reason"],
            preferred_max_rpe=           prefs_d["preferred_max_rpe"],
            prefers_low_impact=          prefs_d["prefers_low_impact"],
            available_days_per_week=     prefs_d["available_days_per_week"],
            preferred_long_run_day=      prefs_d["preferred_long_run_day"],
            preferred_workout_time=      prefs_d["preferred_workout_time"],
            goal_race_distance=          prefs_d["goal_race_distance"],
            target_finish_time=          prefs_d["target_finish_time"],
            experience_level=            prefs_d["experience_level"],
        )
        macro = MacroRatio(
            carb_pct=    macro_d["carb_pct"],
            protein_pct= macro_d["protein_pct"],
            fat_pct=     macro_d["fat_pct"],
        )

        profiles.append(RunnerProfile(**d, macro_ratio=macro, preferences=prefs))

    return profiles


def get_pilot_profiles(
    profiles: Optional[List[RunnerProfile]] = None,
) -> List[RunnerProfile]:
    """
    Return pilot profiles (is_pilot=True) from the loaded profile set.

    Args:
        profiles: Pre-loaded profiles. If None, loads from default path.

    Returns:
        List of pilot RunnerProfile objects.
    """
    if profiles is None:
        profiles = load_profiles()
    return [p for p in profiles if p.is_pilot]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from collections import Counter

    print("Generating 200 runner profiles (marathon cohort)...")
    profiles = build_profiles()

    type_counts = Counter(p.profile_type            for p in profiles)
    div_counts  = Counter(p.age_division             for p in profiles)
    gen_counts  = Counter(p.gender                   for p in profiles)
    exp_counts  = Counter(p.preferences.experience_level for p in profiles)

    # Nullable field coverage — spot-check population probabilities
    n = len(profiles)
    null_5k   = sum(1 for p in profiles if p.prior_5k_time_min           is None)
    null_10k  = sum(1 for p in profiles if p.prior_10k_time_min          is None)
    null_half = sum(1 for p in profiles if p.prior_half_marathon_time_min is None)
    null_full = sum(1 for p in profiles if p.prior_marathon_time_min      is None)

    print(f"\nTotal profiles      : {n}")
    print(f"By profile type     : {dict(sorted(type_counts.items()))}")
    print(f"By age division     : {dict(sorted(div_counts.items()))}")
    print(f"By gender           : {dict(gen_counts.items())}")
    print(f"By experience level : {dict(sorted(exp_counts.items()))}")
    print(f"\nNullable field coverage (None count / {n}):")
    print(f"  prior_5k_time_min            : {null_5k:3d} null  ({null_5k/n:.0%})")
    print(f"  prior_10k_time_min           : {null_10k:3d} null  ({null_10k/n:.0%})")
    print(f"  prior_half_marathon_time_min : {null_half:3d} null  ({null_half/n:.0%})")
    print(f"  prior_marathon_time_min      : {null_full:3d} null  ({null_full/n:.0%})")

    save_profiles(profiles)