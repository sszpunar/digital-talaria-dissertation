"""
wearable_data.py
----------------
Digital Talaria – Wearable Data Input Schema
Device-Agnostic Normalized Layer (2026 Edition)

Defines the WearableData dataclass — the normalized representation of
metrics ingested from consumer wearable devices. Device-specific API
adapters (Apple HealthKit, Garmin Connect, Whoop API, Oura Cloud API,
Google Health Connect) are responsible for fetching, unit-converting,
and populating this schema. TFN components consume only WearableData
and have no awareness of the source device.

Design principles:
    - Device-agnostic: all fields use SI or standard sports-science units
    - Optional with explicit null: None = not provided by device or not synced
    - Grouped by functional role: recovery, sleep, training load, biometrics
    - Forward-compatible: new device metrics are additive, never breaking
    - Simulation-compatible: WearableSimulator generates synthetic readings
      consistent with RunnerProfile physiological parameters

Metric coverage by device (2026):
    Apple HealthKit   : HR, HRV (spot), SpO2, skin temp, sleep (basic),
                        active calories, VO2 max estimate, ECG (flag only)
    Garmin Connect    : HR, HRV, SpO2, training load, training status,
                        body battery, training effect, VO2 max estimate,
                        sleep (basic), respiration rate, stress score
    Whoop API         : HRV, resting HR, recovery score, strain score,
                        sleep performance, respiratory rate, skin temp
    Oura Cloud API    : HRV, resting HR, readiness score, sleep stages
                        (high accuracy), body temp deviation, SpO2,
                        activity score, respiratory rate
    Google Health     : HR, steps, sleep sessions, skin temp, SpO2,
    Connect (Android)   exercise routes (replaces Google Fit, deprecated 2026)

References:
    Dial et al. (2025)      — Nocturnal HRV validation, Oura CCC=0.99
    Schyvens et al. (2025)  — Sleep staging accuracy comparison
    Chin et al. (2023)      — Oura Ring 3 polysomnography validation
    WHOOP Research (2023)   — Physiological monitoring for recovery
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import List, Optional


# ---------------------------------------------------------------------------
# Source device registry
# ---------------------------------------------------------------------------

SUPPORTED_SOURCES = [
    "apple_healthkit",
    "garmin_connect",
    "whoop",
    "oura_ring",
    "google_health_connect",
    "manual",        # runner-entered values (no device)
    "simulated",     # synthetic data for study purposes
]


# ---------------------------------------------------------------------------
# Sub-schemas (grouped by functional role)
# ---------------------------------------------------------------------------

@dataclass
class RecoveryMetrics:
    """
    Recovery and readiness indicators.

    These metrics feed directly into the Banister encoder as real-time
    supplements to TRIMP-derived fitness and fatigue estimates. Device
    composite scores (recovery_score, readiness_score, body_battery)
    are normalized to [0, 1] regardless of source scale.

    Device coverage:
        hrv_rmssd_ms        : Whoop, Oura (high accuracy), Garmin, Apple (spot)
        hrv_sdnn_ms         : Garmin, Oura
        recovery_score      : Whoop (0–100 → normalized 0–1)
        readiness_score     : Oura (0–100 → normalized 0–1)
        body_battery        : Garmin (0–100 → normalized 0–1)
        resting_hr_bpm      : All devices
        respiratory_rate    : Whoop, Oura, Garmin
        skin_temp_deviation : Whoop (°C from personal baseline)
        body_temp_deviation : Oura (°C from personal baseline)
        spo2_pct            : Apple, Garmin, Oura, Whoop
        stress_score        : Garmin (0–100 → normalized 0–1)
    """
    # -- HRV (primary recovery signal) ---------------------------------------
    hrv_rmssd_ms:         Optional[float] = None  # ms; gold standard metric
    hrv_sdnn_ms:          Optional[float] = None  # ms; alternative HRV measure

    # -- Composite device scores (all normalized to [0, 1]) ------------------
    recovery_score:       Optional[float] = None  # Whoop recovery (0–1)
    readiness_score:      Optional[float] = None  # Oura readiness (0–1)
    body_battery:         Optional[float] = None  # Garmin body battery (0–1)
    stress_score:         Optional[float] = None  # Garmin stress (0–1, inv.)

    # -- Physiological vitals ------------------------------------------------
    resting_hr_bpm:       Optional[float] = None  # bpm; replaces profile field
    respiratory_rate:     Optional[float] = None  # breaths/min
    spo2_pct:             Optional[float] = None  # % blood oxygen saturation

    # -- Temperature deviation (from personal baseline) ----------------------
    skin_temp_deviation_c:  Optional[float] = None  # °C; Whoop
    body_temp_deviation_c:  Optional[float] = None  # °C; Oura

    # -- Data quality --------------------------------------------------------
    measurement_window_hrs: Optional[float] = None  # hours of valid data
    confidence:             Optional[float] = None  # 0–1, device-reported


@dataclass
class SleepMetrics:
    """
    Sleep quantity and quality metrics.

    Replaces the static sleep_hours_avg and sleep_quality_1to5 profile
    fields with dynamic nightly measurements. Sleep stage data enables
    the Banister encoder to weight fatigue decay by sleep quality rather
    than using population averages.

    Device coverage:
        total_sleep_hrs     : All devices
        sleep_efficiency_pct: Oura (high accuracy), Whoop, Garmin
        rem_pct             : Oura, Whoop, Garmin
        deep_pct            : Oura, Whoop, Garmin
        light_pct           : Oura, Whoop, Garmin
        awake_pct           : Oura, Whoop, Garmin
        sleep_score         : Oura, Whoop, Garmin (normalized 0–1)
        sleep_latency_min   : Oura, Whoop
        hrv_during_sleep_ms : Oura, Whoop (most accurate context for HRV)
        respiratory_disturbance_idx : Oura
    """
    # -- Quantity ------------------------------------------------------------
    total_sleep_hrs:        Optional[float] = None  # hours
    sleep_latency_min:      Optional[float] = None  # minutes to fall asleep

    # -- Stage breakdown (fractions, sum to 1.0) -----------------------------
    rem_pct:                Optional[float] = None  # 0–1
    deep_pct:               Optional[float] = None  # 0–1 (slow-wave)
    light_pct:              Optional[float] = None  # 0–1
    awake_pct:              Optional[float] = None  # 0–1

    # -- Quality scores (normalized to [0, 1]) -------------------------------
    sleep_efficiency_pct:   Optional[float] = None  # time asleep / time in bed
    sleep_score:            Optional[float] = None  # composite device score
    hrv_during_sleep_ms:    Optional[float] = None  # ms; most accurate HRV ctx

    # -- Disturbance indicators ----------------------------------------------
    respiratory_disturbance_idx: Optional[float] = None  # events/hour; Oura


@dataclass
class TrainingLoadMetrics:
    """
    Acute training load metrics from the preceding workout session.

    These metrics complement PerformanceLog entries with device-measured
    actuals, enabling the Banister encoder to compute TRIMP from real
    heart rate zone data rather than estimated values. Training effect
    scores provide periodization-aware context for the constraint layer.

    Device coverage:
        active_hr_avg_bpm   : All devices (workout HR)
        active_hr_max_bpm   : All devices
        hr_zone_*_min       : Garmin, Whoop, Apple (5-zone model)
        trimp_estimate      : Garmin (direct); computed from zones otherwise
        strain_score        : Whoop (0–21 scale → normalized 0–1)
        training_load_score : Garmin training load (normalized 0–1)
        training_effect_aerobic   : Garmin (0–5 → normalized 0–1)
        training_effect_anaerobic : Garmin (0–5 → normalized 0–1)
        training_status     : Garmin categorical
        vo2_max_estimate    : Garmin, Apple (ml/kg/min)
        active_calories     : All devices
        steps               : All devices
        distance_km         : Garmin, Apple, Whoop
        pace_min_per_km     : Garmin, Apple
        cadence_spm         : Garmin, Apple (steps per minute)
        ground_contact_ms   : Garmin (biomechanical; future use)
        vertical_oscillation_cm : Garmin (biomechanical; future use)
    """
    # -- Heart rate during activity ------------------------------------------
    active_hr_avg_bpm:      Optional[float] = None
    active_hr_max_bpm:      Optional[float] = None

    # -- HR zone distribution (minutes in each zone) -------------------------
    hr_zone_1_min:          Optional[float] = None  # very light (<57% HRmax)
    hr_zone_2_min:          Optional[float] = None  # light (57–63%)
    hr_zone_3_min:          Optional[float] = None  # aerobic (64–76%)
    hr_zone_4_min:          Optional[float] = None  # threshold (77–95%)
    hr_zone_5_min:          Optional[float] = None  # max (>95%)

    # -- Load scores (all normalized to [0, 1]) ------------------------------
    trimp_estimate:         Optional[float] = None  # training impulse
    strain_score:           Optional[float] = None  # Whoop (0–21 → 0–1)
    training_load_score:    Optional[float] = None  # Garmin normalized

    # -- Training effect (Garmin, normalized to [0, 1]) ----------------------
    training_effect_aerobic:   Optional[float] = None
    training_effect_anaerobic: Optional[float] = None
    training_status:           Optional[str]   = None  # e.g. "productive"

    # -- Performance metrics -------------------------------------------------
    vo2_max_estimate:       Optional[float] = None  # ml/kg/min
    active_calories:        Optional[int]   = None
    steps:                  Optional[int]   = None
    distance_km:            Optional[float] = None
    pace_min_per_km:        Optional[float] = None
    cadence_spm:            Optional[float] = None

    # -- Biomechanical (Garmin; Phase 4 / future use) ------------------------
    ground_contact_ms:          Optional[float] = None
    vertical_oscillation_cm:    Optional[float] = None


@dataclass
class CumulativeLoadMetrics:
    """
    Rolling window load summaries for trend detection.

    These metrics are computed by the device platform over 7-day and
    28-day windows and give the constraint layer and anomaly detection
    layer a multi-week view without requiring raw log reconstruction.
    All window metrics are computed by the source device; TFN uses them
    as-is without re-aggregation.

    Device coverage:
        acute_load_7d       : Garmin, Whoop
        chronic_load_28d    : Garmin, Whoop
        acwr                : Acute:chronic workload ratio (computed)
        resting_hr_trend    : Garmin, Oura (7-day rolling)
        hrv_trend           : Garmin, Oura, Whoop (7-day rolling)
    """
    acute_load_7d:          Optional[float] = None  # normalized 0–1
    chronic_load_28d:       Optional[float] = None  # normalized 0–1
    acwr:                   Optional[float] = None  # acute:chronic ratio
    resting_hr_trend_7d:    Optional[float] = None  # bpm delta from baseline
    hrv_trend_7d:           Optional[float] = None  # ms delta from baseline


# ---------------------------------------------------------------------------
# Top-level WearableData schema
# ---------------------------------------------------------------------------

@dataclass
class WearableData:
    """
    Normalized wearable data record for a single runner on a single date.

    This is the canonical input object consumed by TFN's Banister encoder,
    feature extraction pipeline, adherence model, and anomaly detection
    layer. It is populated by device-specific adapter classes and stored
    per runner per day in the training log pipeline.

    All sub-schemas are optional as a group — a runner with no wearable
    device produces WearableData with all sub-schemas set to None.
    Individual fields within a sub-schema may be None when the source
    device does not provide that metric.

    Usage:
        # From device adapter
        wearable = WearableData(
            runner_id="20-F-PA-01",
            record_date=date.today(),
            source_device="oura_ring",
            recovery=RecoveryMetrics(hrv_rmssd_ms=62.4, readiness_score=0.81),
            sleep=SleepMetrics(total_sleep_hrs=7.8, deep_pct=0.22),
        )

        # From simulator
        wearable = WearableSimulator.generate(profile, week_number=3, day_number=2)
    """

    # -- Identity ------------------------------------------------------------
    runner_id:          str
    record_date:        date
    source_device:      str                  # must be in SUPPORTED_SOURCES
    synced_at:          Optional[datetime] = None

    # -- Sub-schemas ---------------------------------------------------------
    recovery:           Optional[RecoveryMetrics]        = None
    sleep:              Optional[SleepMetrics]           = None
    training_load:      Optional[TrainingLoadMetrics]    = None
    cumulative_load:    Optional[CumulativeLoadMetrics]  = None

    # -- Data quality flags --------------------------------------------------
    sync_complete:      bool  = True   # False = partial sync, use with caution
    manually_entered:   bool  = False  # True = runner typed values, not device
    flagged_anomaly:    bool  = False  # True = anomaly detection triggered

    def has_recovery_data(self) -> bool:
        """True if any recovery metric is present."""
        return self.recovery is not None and any(
            v is not None for v in vars(self.recovery).values()
            if not isinstance(v, str)
        )

    def has_sleep_data(self) -> bool:
        """True if any sleep metric is present."""
        return self.sleep is not None and any(
            v is not None for v in vars(self.sleep).values()
        )

    def has_training_load_data(self) -> bool:
        """True if any training load metric is present."""
        return self.training_load is not None and any(
            v is not None for v in vars(self.training_load).values()
            if not isinstance(v, str)
        )

    def primary_hrv(self) -> Optional[float]:
        """
        Return the best available HRV estimate.

        Priority: sleep HRV (most accurate context) → RMSSD → SDNN.
        Returns None if no HRV data is available.
        """
        if self.sleep and self.sleep.hrv_during_sleep_ms is not None:
            return self.sleep.hrv_during_sleep_ms
        if self.recovery:
            if self.recovery.hrv_rmssd_ms is not None:
                return self.recovery.hrv_rmssd_ms
            if self.recovery.hrv_sdnn_ms is not None:
                return self.recovery.hrv_sdnn_ms
        return None

    def primary_readiness(self) -> Optional[float]:
        """
        Return the best available readiness score (0–1).

        Priority: Oura readiness → Whoop recovery → Garmin body battery.
        Returns None if no readiness score is available.
        """
        if self.recovery:
            if self.recovery.readiness_score is not None:
                return self.recovery.readiness_score
            if self.recovery.recovery_score is not None:
                return self.recovery.recovery_score
            if self.recovery.body_battery is not None:
                return self.recovery.body_battery
        return None

    def primary_sleep_hours(self) -> Optional[float]:
        """Return total sleep hours if available."""
        if self.sleep and self.sleep.total_sleep_hrs is not None:
            return self.sleep.total_sleep_hrs
        return None

    def to_banister_inputs(self) -> dict:
        """
        Extract fields relevant to Banister encoder augmentation.

        Returns a dict consumed by the BanisterEncoder when computing
        fitness, fatigue, and readiness. Fields absent from the device
        record return None and the encoder falls back to TRIMP-only
        estimates for those components.
        """
        return {
            "hrv_ms":            self.primary_hrv(),
            "readiness_score":   self.primary_readiness(),
            "resting_hr_bpm":    (self.recovery.resting_hr_bpm
                                  if self.recovery else None),
            "sleep_hrs":         self.primary_sleep_hours(),
            "sleep_efficiency":  (self.sleep.sleep_efficiency_pct
                                  if self.sleep else None),
            "hrv_trend_7d":      (self.cumulative_load.hrv_trend_7d
                                  if self.cumulative_load else None),
            "acwr":              (self.cumulative_load.acwr
                                  if self.cumulative_load else None),
            "strain_score":      (self.training_load.strain_score
                                  if self.training_load else None),
            "trimp_actual":      (self.training_load.trimp_estimate
                                  if self.training_load else None),
        }

    def to_feature_vector_supplement(self) -> dict:
        """
        Extract fields that supplement the runner feature vector.

        These replace or augment the static profile fields:
            resting_hr_bpm      → replaces profile.resting_hr_bpm
            vo2_max_estimate    → replaces profile.vo2_proxy
            sleep_hours_avg     → replaces profile.sleep_hours_avg
            sleep_quality       → derived from sleep efficiency + stages
        """
        sleep_quality_derived = None
        if self.sleep:
            eff   = self.sleep.sleep_efficiency_pct or 0.0
            deep  = self.sleep.deep_pct or 0.0
            rem   = self.sleep.rem_pct or 0.0
            # Weighted composite: efficiency (50%) + deep (30%) + REM (20%)
            # Scaled to [0, 1]; maps to profile sleep_quality_1to5 scale
            sleep_quality_derived = round(eff * 0.5 + deep * 0.3 + rem * 0.2, 4)

        return {
            "resting_hr_bpm":       (self.recovery.resting_hr_bpm
                                     if self.recovery else None),
            "vo2_max_estimate":     (self.training_load.vo2_max_estimate
                                     if self.training_load else None),
            "sleep_hours_dynamic":  self.primary_sleep_hours(),
            "sleep_quality_dynamic": sleep_quality_derived,
            "hrv_rmssd_ms":         self.primary_hrv(),
            "readiness_score":      self.primary_readiness(),
            "respiratory_rate":     (self.recovery.respiratory_rate
                                     if self.recovery else None),
        }


# ---------------------------------------------------------------------------
# Wearable simulator (for dissertation study)
# ---------------------------------------------------------------------------

class WearableSimulator:
    """
    Generates synthetic WearableData consistent with RunnerProfile
    physiological parameters for use in the Digital Talaria study.

    Simulation strategy:
        - Base values derived from runner profile fields
        - Week/day-specific variation using the same seeded RNG as
          the adherence model (ensures deterministic, cross-model
          consistency)
        - Profile-type-specific noise distributions:
          PA: low variance, high HRV baseline
          SD: high variance, degraded sleep metrics
          HS: elevated resting HR, suppressed HRV
          UW: lower strain tolerance, faster recovery
          OW: elevated resting HR, reduced VO2 estimate

    Usage:
        wearable = WearableSimulator.generate(profile, week=3, day=2)
    """

    # Profile-type base parameters
    _PROFILE_PARAMS = {
        "PA": {
            "hrv_base": 65.0, "hrv_var": 8.0,
            "rhr_base": 50.0, "rhr_var": 4.0,
            "readiness_base": 0.82, "readiness_var": 0.08,
            "sleep_hrs_base": 7.8, "sleep_hrs_var": 0.6,
            "deep_pct_base": 0.22, "rem_pct_base": 0.23,
        },
        "UW": {
            "hrv_base": 52.0, "hrv_var": 9.0,
            "rhr_base": 62.0, "rhr_var": 5.0,
            "readiness_base": 0.72, "readiness_var": 0.10,
            "sleep_hrs_base": 6.5, "sleep_hrs_var": 0.8,
            "deep_pct_base": 0.18, "rem_pct_base": 0.20,
        },
        "OW": {
            "hrv_base": 44.0, "hrv_var": 10.0,
            "rhr_base": 74.0, "rhr_var": 6.0,
            "readiness_base": 0.65, "readiness_var": 0.12,
            "sleep_hrs_base": 6.8, "sleep_hrs_var": 0.9,
            "deep_pct_base": 0.16, "rem_pct_base": 0.20,
        },
        "HS": {
            "hrv_base": 46.0, "hrv_var": 12.0,
            "rhr_base": 70.0, "rhr_var": 7.0,
            "readiness_base": 0.62, "readiness_var": 0.14,
            "sleep_hrs_base": 6.4, "sleep_hrs_var": 1.0,
            "deep_pct_base": 0.15, "rem_pct_base": 0.19,
        },
        "SD": {
            "hrv_base": 38.0, "hrv_var": 14.0,
            "rhr_base": 68.0, "rhr_var": 8.0,
            "readiness_base": 0.55, "readiness_var": 0.16,
            "sleep_hrs_base": 4.8, "sleep_hrs_var": 1.2,
            "deep_pct_base": 0.10, "rem_pct_base": 0.16,
        },
    }

    @classmethod
    def generate(
        cls,
        profile,
        week_number:  int,
        day_number:   int,
        record_date:  Optional[date] = None,
    ) -> "WearableData":
        """
        Generate a synthetic WearableData record for a runner-day.

        Uses the same seeding strategy as the adherence model:
        hash(runner_id + week + day) for full reproducibility.

        Args:
            profile:      RunnerProfile instance.
            week_number:  Current week (1–20).
            day_number:   Current day (1–7).
            record_date:  Date for the record (defaults to None).

        Returns:
            WearableData with simulated recovery, sleep, and
            training load metrics.
        """
        import numpy as np

        seed = abs(hash(
            f"wearable_{profile.runner_id}_{week_number}_{day_number}"
        )) % (2 ** 31)
        rng = np.random.default_rng(seed)

        pt     = profile.profile_type
        params = cls._PROFILE_PARAMS.get(pt, cls._PROFILE_PARAMS["PA"])

        # -- Apply individual profile modifiers ------------------------------
        # SD: sleep quality degrades with poor sleep score
        sleep_mod = 1.0
        if pt == "SD":
            sleep_mod = profile.sleep_quality_1to5 / 5.0

        # HS: HRV suppressed by stress level
        hrv_mod = 1.0
        if pt == "HS":
            hrv_mod = max(0.7, 1.0 - (profile.stress_level_1to5 - 1) / 4.0 * 0.3)

        # -- Generate recovery metrics ---------------------------------------
        hrv = float(np.clip(
            rng.normal(params["hrv_base"] * hrv_mod, params["hrv_var"]),
            20.0, 120.0
        ))
        rhr = float(np.clip(
            rng.normal(params["rhr_base"], params["rhr_var"]),
            35.0, 100.0
        ))
        readiness = float(np.clip(
            rng.normal(params["readiness_base"], params["readiness_var"]),
            0.0, 1.0
        ))
        resp_rate = float(np.clip(rng.normal(14.5, 1.5), 10.0, 22.0))
        spo2      = float(np.clip(rng.normal(97.5, 0.8), 92.0, 100.0))
        temp_dev  = float(np.clip(rng.normal(0.0, 0.15), -1.0, 1.0))

        recovery = RecoveryMetrics(
            hrv_rmssd_ms=round(hrv, 2),
            resting_hr_bpm=round(rhr, 1),
            readiness_score=round(readiness, 4),
            respiratory_rate=round(resp_rate, 1),
            spo2_pct=round(spo2, 1),
            body_temp_deviation_c=round(temp_dev, 3),
            measurement_window_hrs=8.0,
            confidence=round(float(rng.uniform(0.85, 0.99)), 3),
        )

        # -- Generate sleep metrics ------------------------------------------
        sleep_hrs = float(np.clip(
            rng.normal(params["sleep_hrs_base"] * sleep_mod,
                       params["sleep_hrs_var"]),
            2.0, 10.0
        ))
        deep_pct   = float(np.clip(
            rng.normal(params["deep_pct_base"] * sleep_mod, 0.05),
            0.05, 0.35
        ))
        rem_pct    = float(np.clip(
            rng.normal(params["rem_pct_base"] * sleep_mod, 0.05),
            0.05, 0.35
        ))
        awake_pct  = float(np.clip(rng.normal(0.06, 0.02), 0.01, 0.20))
        light_pct  = max(0.0, 1.0 - deep_pct - rem_pct - awake_pct)
        efficiency = float(np.clip(rng.normal(0.85 * sleep_mod, 0.06),
                                   0.50, 0.98))

        sleep = SleepMetrics(
            total_sleep_hrs=round(sleep_hrs, 2),
            sleep_latency_min=round(float(np.clip(
                rng.normal(14.0, 6.0), 2.0, 60.0
            )), 1),
            deep_pct=round(deep_pct, 4),
            rem_pct=round(rem_pct, 4),
            light_pct=round(light_pct, 4),
            awake_pct=round(awake_pct, 4),
            sleep_efficiency_pct=round(efficiency, 4),
            sleep_score=round(float(np.clip(
                efficiency * 0.5 + deep_pct * 0.3 + rem_pct * 0.2, 0.0, 1.0
            )), 4),
            hrv_during_sleep_ms=round(hrv * float(rng.uniform(1.0, 1.15)), 2),
        )

        # -- Generate training load (if active day) --------------------------
        training_load = None
        if day_number != 7:  # day 7 = typical rest day
            # Weekly volume baseline scaled by week progression
            vol_factor   = 1.0 + (week_number - 1) * 0.03
            base_volume  = profile.weekly_mileage_baseline_mi * vol_factor
            # Daily distance: typical easy run is 25–30% of weekly volume
            daily_dist   = base_volume * 0.20  # conservative single session
            duration_min = float(np.clip(
                rng.normal(daily_dist * 10.0, 5.0), 20.0, 120.0
            ))

            hr_avg  = float(np.clip(
                rng.normal(profile.resting_hr_bpm * 1.55, 8.0),
                80.0, 190.0
            ))
            hr_max  = float(np.clip(hr_avg + rng.normal(20.0, 5.0),
                                    hr_avg, 200.0))
            # TRIMP mirrors _compute_trimp() in banister_encoder.py:
            #   TRIMP = distance × (rpe/10) × (duration/60)
            rpe_norm   = 5.0 / 10.0  # easy run baseline RPE
            dur_norm   = duration_min / 60.0
            trimp      = round(daily_dist * rpe_norm * dur_norm, 4)
            strain_raw = float(np.clip(trimp / 3.0, 0.0, 1.0))

            training_load = TrainingLoadMetrics(
                active_hr_avg_bpm=round(hr_avg, 1),
                active_hr_max_bpm=round(hr_max, 1),
                hr_zone_2_min=round(duration_min * 0.40, 1),
                hr_zone_3_min=round(duration_min * 0.35, 1),
                hr_zone_4_min=round(duration_min * 0.15, 1),
                hr_zone_1_min=round(duration_min * 0.10, 1),
                trimp_estimate=trimp,
                strain_score=round(strain_raw, 4),
                vo2_max_estimate=round(float(profile.vo2_proxy), 1),
                active_calories=int(daily_dist * 100),
                distance_km=round(daily_dist * 1.609, 2),
                cadence_spm=round(float(rng.normal(175.0, 5.0)), 1),
            )

        # -- Cumulative load (rolling windows) -------------------------------
        acwr = None
        if week_number >= 4:
            acute  = float(np.clip(rng.normal(0.85, 0.10), 0.4, 1.5))
            chronic = float(np.clip(rng.normal(0.80, 0.06), 0.4, 1.3))
            acwr   = round(acute / chronic if chronic > 0 else 1.0, 3)

        cumulative_load = CumulativeLoadMetrics(
            acwr=acwr,
            resting_hr_trend_7d=round(float(rng.normal(0.0, 1.5)), 2),
            hrv_trend_7d=round(float(rng.normal(0.0, 3.0)), 2),
        )

        return WearableData(
            runner_id=profile.runner_id,
            record_date=record_date or date.today(),
            source_device="simulated",
            recovery=recovery,
            sleep=sleep,
            training_load=training_load,
            cumulative_load=cumulative_load,
            sync_complete=True,
            manually_entered=False,
            flagged_anomaly=False,
        )