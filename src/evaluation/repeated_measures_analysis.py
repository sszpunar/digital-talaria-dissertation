"""
repeated_measures_analysis.py
-----------------------------
Digital Talaria - Repeated-Measures Statistical Analysis (Study 1 and Study 2)

Produces the inferential statistics for RQ1 through RQ5 from the combined
evaluation dataset written by comparison_evaluator.py.

Design rationale
----------------
Every runner in the evaluation cohort is assessed under every model and every
ablation configuration, with identical disruption draws supplied by the shared
adherence simulator. Both studies are therefore fully crossed repeated-measures
designs in which the runner is the blocking factor and the model or
configuration is the within-block condition.

The Kruskal-Wallis H-test assumes independent samples and is not appropriate
for this structure. Applying it treats repeated observations on the same runner
as independent, which is pseudoreplication and inflates the test statistic.
This module instead applies:

    Omnibus     Friedman test across conditions, with the runner as the block
                (Friedman, 1937).
    Effect size Kendall's coefficient of concordance, W = chi2 / (n(k-1)),
                bounded on [0, 1] (Kendall & Babington Smith, 1939).
    Post hoc    Pairwise Wilcoxon signed-rank tests with Bonferroni correction
                (Wilcoxon, 1945).

Each runner contributes one value per condition, computed as that runner's mean
adherence score across the simulated weeks within the cell being tested.
Blocking at the runner level keeps blocks mutually independent, which blocking
at the runner-week level would not.

Analyses performed
------------------
    Study 1 (RQ1-RQ4)   six recommendation models, compared overall and within
                        each profile type, training phase, and fitness tier.
                        Post hoc across all 15 model pairs per cell.

    Study 2 (RQ5)       eight additive ablation configurations, compared
                        overall. Post hoc contrasts each configuration against
                        the complete TFN architecture, Bonferroni corrected
                        across the 7 contrasts.

Outputs
-------
    data/comparison/rm_omnibus.csv     one row per analysis cell
    data/comparison/rm_pairwise.csv    one row per contrast per cell
    data/comparison/rm_report.txt      narrative summary

Usage
-----
    python -m src.evaluation.repeated_measures_analysis

References
----------
    Friedman, M. (1937). The use of ranks to avoid the assumption of normality
        implicit in the analysis of variance. Journal of the American
        Statistical Association, 32(200), 675-701.
    Kendall, M. G., & Babington Smith, B. (1939). The problem of m rankings.
        The Annals of Mathematical Statistics, 10(3), 275-287.
    Wilcoxon, F. (1945). Individual comparisons by ranking methods.
        Biometrics Bulletin, 1(6), 80-83.
"""

from __future__ import annotations

import csv
import itertools
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

COMBINED_SPSS = os.path.join("data", "comparison", "combined_spss.csv")
OUTPUT_DIR = os.path.join("data", "comparison")
OMNIBUS_PATH = os.path.join(OUTPUT_DIR, "rm_omnibus.csv")
PAIRWISE_PATH = os.path.join(OUTPUT_DIR, "rm_pairwise.csv")
REPORT_PATH = os.path.join(OUTPUT_DIR, "rm_report.txt")

# ---------------------------------------------------------------------------
# Study definitions
# ---------------------------------------------------------------------------

# Six architecturally distinct models compared in Study 1. The complete TFN
# also terminates the Study 2 additive staircase.
STUDY1_MODELS: List[str] = [
    "logreg", "xgb", "sasrec", "transformer", "two_tower", "TFN",
]

# Eight additive ablation configurations in staircase order.
STUDY2_CONFIGS: List[str] = [
    "TFN-B", "TFN-B-G", "TFN-BG-C", "TFN-BGC-Uq",
    "TFN-BGCUq-W", "TFN-BGCUqW-Mh", "TFN-BGCUqWMh-Mt", "TFN",
]

REFERENCE_CONFIG = "TFN"

LABELS: Dict[str, str] = {
    "logreg": "Logistic Regression",
    "xgb": "XGBoost",
    "sasrec": "SASRec",
    "transformer": "Transformer",
    "two_tower": "Two-Tower",
    "TFN": "TFN",
    "TFN-B": "TFN-B",
    "TFN-B-G": "TFN-B-G",
    "TFN-BG-C": "TFN-BG-C",
    "TFN-BGC-Uq": "TFN-BGC-Uq",
    "TFN-BGCUq-W": "TFN-BGCUq-W",
    "TFN-BGCUqW-Mh": "TFN-BGCUqW-Mh",
    "TFN-BGCUqWMh-Mt": "TFN-BGCUqWMh-Mt",
}

PROFILE_TYPES: List[str] = ["PA", "UW", "OW", "HS", "SD"]
TRAINING_PHASES: List[str] = ["base", "build", "peak", "taper"]
FITNESS_TIERS: List[str] = ["low", "mid", "high"]

# ---------------------------------------------------------------------------
# Analysis constants
# ---------------------------------------------------------------------------

ALPHA = 0.05

# Kendall's W is bounded on [0, 1] and expresses the proportion of rank
# variance attributable to systematic differences between conditions.
W_NEGLIGIBLE = 0.10
W_SMALL = 0.30
W_MODERATE = 0.50

BLOCK_COLUMN = "runner_id"
VALUE_COLUMN = "avg_adherence_score"
MODEL_COLUMN = "model"

MIN_BLOCKS = 5
MIN_CONDITIONS = 3


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class OmnibusResult:
    """Friedman omnibus result for a single analysis cell."""
    study: str
    research_question: str
    dimension: str
    cell: str
    chi_square: float
    df: int
    p_value: float
    kendalls_w: float
    n_blocks: int
    k_conditions: int

    @property
    def significant(self) -> bool:
        return self.p_value < ALPHA

    @property
    def effect_magnitude(self) -> str:
        if self.kendalls_w < W_NEGLIGIBLE:
            return "negligible"
        if self.kendalls_w < W_SMALL:
            return "small"
        if self.kendalls_w < W_MODERATE:
            return "moderate"
        return "large"

    @property
    def meaningful(self) -> bool:
        """
        Dual criterion: a difference is meaningful when it is statistically
        significant and the effect size exceeds the negligible threshold.
        """
        return self.significant and self.kendalls_w >= W_NEGLIGIBLE

    def to_row(self) -> dict:
        return {
            "study": self.study,
            "research_question": self.research_question,
            "dimension": self.dimension,
            "cell": self.cell,
            "test": "Friedman",
            "chi_square": round(self.chi_square, 4),
            "df": self.df,
            "p_value": f"{self.p_value:.3e}",
            "kendalls_w": round(self.kendalls_w, 4),
            "effect_magnitude": self.effect_magnitude,
            "n_blocks": self.n_blocks,
            "k_conditions": self.k_conditions,
            "significant": self.significant,
            "meaningful": self.meaningful,
        }


@dataclass
class ContrastResult:
    """Wilcoxon signed-rank result for one condition pair within a cell."""
    study: str
    cell: str
    condition_a: str
    condition_b: str
    statistic: float
    p_raw: float
    p_bonferroni: float
    median_difference: float
    n_pairs: int
    correction_divisor: int

    @property
    def significant(self) -> bool:
        return self.p_bonferroni < ALPHA

    def to_row(self) -> dict:
        return {
            "study": self.study,
            "cell": self.cell,
            "condition_a": self.condition_a,
            "condition_b": self.condition_b,
            "test": "Wilcoxon signed-rank",
            "statistic": round(self.statistic, 2),
            "p_raw": f"{self.p_raw:.3e}",
            "p_bonferroni": f"{self.p_bonferroni:.3e}",
            "correction_divisor": self.correction_divisor,
            "median_difference": round(self.median_difference, 5),
            "n_pairs": self.n_pairs,
            "significant": self.significant,
        }


# ---------------------------------------------------------------------------
# Core statistics
# ---------------------------------------------------------------------------

def build_block_matrix(
    frame: pd.DataFrame,
    conditions: Sequence[str],
) -> pd.DataFrame:
    """
    Reduce a subset of the combined dataset to a complete block matrix.

    Rows are blocks (runners) and columns are conditions. Each cell holds that
    runner's mean adherence score under that condition across the rows present
    in the subset. Blocks missing any condition are dropped so that the design
    passed to the Friedman test is complete.

    Args:
        frame: Subset of the combined dataset for one analysis cell.
        conditions: Condition labels, in the order they should appear.

    Returns:
        Complete block matrix, one row per runner and one column per condition.
    """
    matrix = frame.pivot_table(
        index=BLOCK_COLUMN,
        columns=MODEL_COLUMN,
        values=VALUE_COLUMN,
        aggfunc="mean",
    )
    present = [c for c in conditions if c in matrix.columns]
    return matrix[present].dropna()


def friedman_omnibus(
    matrix: pd.DataFrame,
    study: str,
    research_question: str,
    dimension: str,
    cell: str,
) -> Optional[OmnibusResult]:
    """
    Run the Friedman omnibus test on a complete block matrix.

    Kendall's W normalizes the Friedman statistic onto [0, 1] as
    chi2 / (n * (k - 1)).

    Returns:
        OmnibusResult, or None when the matrix is too small to test.
    """
    n_blocks, k_conditions = matrix.shape
    if n_blocks < MIN_BLOCKS or k_conditions < MIN_CONDITIONS:
        return None

    chi_square, p_value = stats.friedmanchisquare(
        *[matrix[column].values for column in matrix.columns]
    )
    return OmnibusResult(
        study=study,
        research_question=research_question,
        dimension=dimension,
        cell=cell,
        chi_square=float(chi_square),
        df=k_conditions - 1,
        p_value=float(p_value),
        kendalls_w=float(chi_square / (n_blocks * (k_conditions - 1))),
        n_blocks=n_blocks,
        k_conditions=k_conditions,
    )


def _wilcoxon_contrast(
    matrix: pd.DataFrame,
    condition_a: str,
    condition_b: str,
    study: str,
    cell: str,
    divisor: int,
) -> Optional[ContrastResult]:
    """Run one Wilcoxon signed-rank contrast, or None if the pair is identical."""
    a, b = matrix[condition_a], matrix[condition_b]
    difference = a - b
    if difference.abs().sum() == 0:
        return None  # identical configurations carry no rank information
    statistic, p_raw = stats.wilcoxon(a, b)
    return ContrastResult(
        study=study,
        cell=cell,
        condition_a=condition_a,
        condition_b=condition_b,
        statistic=float(statistic),
        p_raw=float(p_raw),
        p_bonferroni=min(float(p_raw) * divisor, 1.0),
        median_difference=float(difference.median()),
        n_pairs=len(a),
        correction_divisor=divisor,
    )


def all_pairs_contrasts(
    matrix: pd.DataFrame,
    study: str,
    cell: str,
) -> List[ContrastResult]:
    """
    Contrast every condition pair, Bonferroni corrected across all pairs.

    Used for Study 1, where no single condition serves as a reference.
    """
    pairs = list(itertools.combinations(matrix.columns, 2))
    divisor = len(pairs)
    results = [
        _wilcoxon_contrast(matrix, a, b, study, cell, divisor)
        for a, b in pairs
    ]
    return [r for r in results if r is not None]


def reference_contrasts(
    matrix: pd.DataFrame,
    reference: str,
    study: str,
    cell: str,
) -> List[ContrastResult]:
    """
    Contrast every condition against a single reference condition.

    Used for Study 2, where each ablation configuration is compared against
    the complete architecture. Bonferroni correction is applied across the
    number of contrasts performed rather than across all possible pairs.
    """
    others = [c for c in matrix.columns if c != reference]
    divisor = len(others)
    results = [
        _wilcoxon_contrast(matrix, c, reference, study, cell, divisor)
        for c in others
    ]
    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# Study drivers
# ---------------------------------------------------------------------------

def analyse_study1(
    frame: pd.DataFrame,
) -> Tuple[List[OmnibusResult], List[ContrastResult]]:
    """Run the Study 1 analysis across all RQ1-RQ4 cells."""
    subset = frame[frame[MODEL_COLUMN].isin(STUDY1_MODELS)]

    cells: List[Tuple[str, str, str, pd.DataFrame]] = [
        ("RQ1", "overall", "overall", subset),
    ]
    cells += [("RQ2", "profile_type", f"profile_type={t}",
               subset[subset["profile_type"] == t]) for t in PROFILE_TYPES]
    cells += [("RQ3", "training_phase", f"training_phase={p}",
               subset[subset["training_phase"] == p]) for p in TRAINING_PHASES]
    cells += [("RQ4", "fitness_tier", f"fitness_tier={t}",
               subset[subset["fitness_tier"] == t]) for t in FITNESS_TIERS]

    omnibus: List[OmnibusResult] = []
    contrasts: List[ContrastResult] = []
    for question, dimension, cell, cell_frame in cells:
        matrix = build_block_matrix(cell_frame, STUDY1_MODELS)
        result = friedman_omnibus(matrix, "Study 1", question, dimension, cell)
        if result is None:
            print(f"  [SKIP] {cell}: insufficient blocks")
            continue
        omnibus.append(result)
        contrasts.extend(all_pairs_contrasts(matrix, "Study 1", cell))
    return omnibus, contrasts


def analyse_study2(
    frame: pd.DataFrame,
) -> Tuple[List[OmnibusResult], List[ContrastResult]]:
    """Run the Study 2 ablation analysis for RQ5."""
    subset = frame[frame[MODEL_COLUMN].isin(STUDY2_CONFIGS)]
    matrix = build_block_matrix(subset, STUDY2_CONFIGS)

    result = friedman_omnibus(
        matrix, "Study 2", "RQ5", "ablation", "additive_staircase")
    if result is None:
        print("  [SKIP] additive_staircase: insufficient blocks")
        return [], []

    contrasts = reference_contrasts(
        matrix, REFERENCE_CONFIG, "Study 2", "additive_staircase")
    return [result], contrasts


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def export_results(
    omnibus: Sequence[OmnibusResult],
    contrasts: Sequence[ContrastResult],
) -> None:
    """Write both result sets to CSV."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(OMNIBUS_PATH, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(omnibus[0].to_row().keys()))
        writer.writeheader()
        writer.writerows(r.to_row() for r in omnibus)
    print(f"[OK] Omnibus results  -> {OMNIBUS_PATH} ({len(omnibus)} cells)")

    with open(PAIRWISE_PATH, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(contrasts[0].to_row().keys()))
        writer.writeheader()
        writer.writerows(r.to_row() for r in contrasts)
    print(f"[OK] Contrast results -> {PAIRWISE_PATH} "
          f"({len(contrasts)} contrasts)")


def _omnibus_table(results: Sequence[OmnibusResult]) -> List[str]:
    """Format an omnibus result table."""
    lines = [f"  {'cell':<26}{'chi2':>10}{'df':>4}{'p':>12}"
             f"{'W':>9}{'magnitude':>12}{'n':>6}"]
    for r in results:
        lines.append(
            f"  {r.cell:<26}{r.chi_square:>10.2f}{r.df:>4}"
            f"{r.p_value:>12.2e}{r.kendalls_w:>9.4f}"
            f"{r.effect_magnitude:>12}{r.n_blocks:>6}")
    return lines


def write_report(
    omnibus: Sequence[OmnibusResult],
    contrasts: Sequence[ContrastResult],
) -> None:
    """Write the narrative summary report."""
    lines: List[str] = []
    add = lines.append

    add("=" * 78)
    add("  DIGITAL TALARIA - Repeated-Measures Analysis (Study 1 and Study 2)")
    add("=" * 78)
    add("")
    add("  Design      : fully crossed repeated measures")
    add("  Blocks      : evaluation runners")
    add("  Omnibus     : Friedman test")
    add("  Effect size : Kendall's W")
    add("  Post hoc    : pairwise Wilcoxon signed-rank, Bonferroni corrected")
    add(f"  Alpha       : {ALPHA}")
    add("")
    add("  A difference is reported as meaningful when it is statistically")
    add(f"  significant and Kendall's W is at least {W_NEGLIGIBLE}.")
    add("")

    for question in ["RQ1", "RQ2", "RQ3", "RQ4", "RQ5"]:
        cells = [r for r in omnibus if r.research_question == question]
        if not cells:
            continue
        add("-" * 78)
        add(f"  {question}  ({cells[0].study})")
        add("-" * 78)
        lines.extend(_omnibus_table(cells))
        meaningful = [r.cell for r in cells if r.meaningful]
        add("")
        add(f"  Meaningful differences: "
            f"{', '.join(meaningful) if meaningful else 'none'}")
        add("")

    add("-" * 78)
    add("  STUDY 2 - EACH CONFIGURATION AGAINST THE COMPLETE TFN")
    add("-" * 78)
    s2 = [c for c in contrasts if c.study == "Study 2"]
    if s2:
        add(f"  {'configuration':<22}{'median diff':>14}{'p_adj':>12}"
            f"   verdict")
        for c in s2:
            if not c.significant:
                verdict = "not significant"
            elif c.median_difference < 0:
                verdict = "significantly below complete TFN"
            else:
                verdict = "significantly above complete TFN"
            add(f"  {LABELS.get(c.condition_a, c.condition_a):<22}"
                f"{c.median_difference:>14.5f}{c.p_bonferroni:>12.2e}"
                f"   {verdict}")
        add("")

    add("-" * 78)
    add("  STUDY 1 - SIGNIFICANT PAIRWISE COMPARISONS (Bonferroni corrected)")
    add("-" * 78)
    s1 = [c for c in contrasts if c.study == "Study 1"]
    for cell in dict.fromkeys(c.cell for c in s1):
        significant = [c for c in s1 if c.cell == cell and c.significant]
        if not significant:
            add(f"  {cell}: no significant pairwise differences")
            add("")
            continue
        add(f"  {cell}")
        for c in sorted(significant, key=lambda r: r.p_bonferroni):
            direction = "<" if c.median_difference < 0 else ">"
            add(f"    {LABELS.get(c.condition_a, c.condition_a):<22}"
                f"{direction} {LABELS.get(c.condition_b, c.condition_b):<22}"
                f"p={c.p_bonferroni:.2e}  median diff="
                f"{c.median_difference:+.5f}")
        add("")

    add("=" * 78)

    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"[OK] Report           -> {REPORT_PATH}")
    print()
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 78)
    print("  Digital Talaria - Repeated-Measures Analysis")
    print("=" * 78)
    print()

    if not os.path.exists(COMBINED_SPSS):
        print(f"[ERROR] Combined dataset not found at '{COMBINED_SPSS}'.")
        print("        Run: python -m src.evaluation.comparison_evaluator")
        return 1

    frame = pd.read_csv(COMBINED_SPSS)
    print(f"Loaded {len(frame):,} rows | "
          f"{frame[MODEL_COLUMN].nunique()} models | "
          f"{frame[BLOCK_COLUMN].nunique()} runners\n")

    omnibus_1, contrasts_1 = analyse_study1(frame)
    omnibus_2, contrasts_2 = analyse_study2(frame)

    omnibus = omnibus_1 + omnibus_2
    contrasts = contrasts_1 + contrasts_2

    if not omnibus:
        print("[ERROR] No cells could be analysed.")
        return 1

    export_results(omnibus, contrasts)
    write_report(omnibus, contrasts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
