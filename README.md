# Digital Talaria — Dissertation Artifacts

**Digital Talaria: Developing and Evaluating a Physiologically Grounded Machine Learning Framework for Adaptive Running Plan Generation**

Sean Eben Aloysius Szpunar
Doctor of Philosophy — School of Technology and Engineering
National University, San Diego, California
September 2026

---

## Scope of this repository

This repository contains the **data generation and statistical analysis artifacts** for the Digital Talaria dissertation: the seeded synthetic cohort generator, the evaluation dataset, the Chapter 4 analysis notebook, and all 44 published figures.

The **Talaria Fusion Network implementation is maintained separately and is not included here.** Neither are the five baseline model implementations, the constraint layer, the simulation harness, or the test suites. This repository exists so that the study's synthetic cohort can be independently regenerated and its published statistics independently recomputed.

---

## Study overview

Digital Talaria is a constraint-aware machine learning recommender system that generates adaptive, personalized 20-week marathon training plans. The study compares six architecturally distinct recommendation models under a shared physiological constraint layer, and decomposes the contribution of each architectural component of the novel model through an ablation analysis.

**Models compared (Study 1):** Logistic Regression, XGBoost, SASRec, an attention-based Transformer, a Two-Tower Neural Network, and the Talaria Fusion Network.

**The Talaria Fusion Network** combines eight components: a Banister physiological state encoder, a graph attention encoder, a cross-attention scorer, Monte Carlo Dropout uncertainty quantification, wearable device integration, multi-head attention, multi-task scoring, and a clinical safety gate.

**Ablation (Study 2):** Eight additive configurations in staircase order, from a Banister-only baseline to the complete architecture, plus two subtractive configurations that remove a single feature signal from the complete model.

**Evaluation:** Every runner is evaluated under every configuration with identical constraint conditions and identical disruption draws, producing a paired counterfactual design across 160 evaluation runners and 20 simulated weeks.

---

## Key results

### Study 1 — cross-model comparison

Kruskal-Wallis omnibus test: **H(5) = 185.11, p < .001, ε² = 0.0042**

| Rank | Model | Mean adherence |
|---|---|---|
| 1 | **Talaria Fusion Network** | **0.8141** |
| 2 | XGBoost | 0.8117 |
| 2 | Two-Tower Neural Network | 0.8117 |
| 4 | SASRec | 0.8100 |
| 5 | Logistic Regression | 0.8041 |
| 6 | Transformer | 0.8015 |

XGBoost (0.811720) and Two-Tower (0.811702) are tied at four decimal places. Dunn post hoc contrasts with Bonferroni correction identified the Transformer and Logistic Regression as significantly below the leading models; the TFN, Two-Tower, and XGBoost did not differ significantly from one another.

The small effect size reflects the shared physiological constraint layer, which standardizes plan quality across architectures. This is the study's primary interpretive finding: **constraint design, not learning paradigm, is the dominant driver of plan quality.**

### Per-runner best-fit wins

Each of the 160 evaluation runners is assigned to the model producing that runner's highest mean adherence score.

| Model | Wins | Share |
|---|---|---|
| **Talaria Fusion Network** | **47** | **29.4%** |
| Two-Tower Neural Network | 35 | 21.9% |
| SASRec | 23 | 14.4% |
| Logistic Regression | 23 | 14.4% |
| XGBoost | 22 | 13.8% |
| Transformer | 10 | 6.3% |

Chi-square goodness-of-fit against a uniform distribution: **χ²(5) = 30.35, p < .001**

Wins by runner profile type show the TFN's advantage concentrating among runners whose body composition deviates most from normative ranges — 14 of 32 Overweight runners and 12 of 32 Underweight runners.

### Study 2 — component ablation

Additive staircase, from a Banister-only baseline (0.7982) to the complete architecture (0.8141), cumulative gain **+0.0159**:

| Component added | Step delta |
|---|---|
| Graph attention encoder | +0.0000 |
| Cross-attention scorer | +0.0040 |
| MC Dropout uncertainty quantification | +0.0019 |
| **Wearable device integration** | **+0.0073** |
| Multi-head attention | +0.0000 |
| Multi-task scoring | +0.0026 |
| Clinical safety gate | +0.0001 |

Zero-delta components are structural prerequisites or joint contributors rather than independent adherence drivers.

Subtractive configurations, each removing one feature signal from the complete model:

| Configuration | Mean adherence | Delta vs. complete TFN |
|---|---|---|
| Without physiological encoder | 0.8052 | **−0.0089** |
| Without fitness-level features | 0.8167 | +0.0026 |

The physiological encoder produces the largest delta observed in the study. The fitness-level result is directionally opposite to prediction and is attributed to simulation-level averaging rather than improved plan quality; see the dissertation's Study 2 discussion.

---

## Synthetic cohort generation

All 200 runner profiles are produced programmatically. No profile is hand-authored, and no profile in the study cohort was generated by a large language model.

**Design.** A balanced factorial frame of five profile types × five age divisions × two genders × four runners per cell:

- **Profile types:** Peak Athlete, Underweight, Overweight, High-Stress, Sleep-Deprived
- **Age divisions:** 20–29, 30–39, 40–49, 50–59, 60–69
- **Genders:** female, male (100 each, balanced by construction)

**Sampling.** Every attribute is drawn from a parameter range specific to the runner's profile type and adjusted for age division and gender. Parameter ranges are anchored to published literature: Tanaka et al. (2001) for maximum heart rate, Riegel (1981) for race-time consistency, and Noakes (2003) and Daniels (2005) for training volume.

**Validation.** Each candidate profile must pass programmatic coherence gates before admission to the cohort — BMI ranges by profile type, long-run-to-weekly-volume ratio within 0.18–0.38, Riegel race-time consistency within ±15% across non-null prior-time pairs, resting heart rate within 38–90 bpm, and profile-type-specific lifestyle constraints. Violating profiles are rejected and resampled.

**Determinism.** Generation is seeded with `GENERATION_SEED = 42`. Regenerating reproduces all 200 profiles identically.

---

## Repository contents

```
digital-talaria-dissertation/
├── src/data_gen/
│   ├── profile_generator.py      # seeded cohort generator
│   ├── profile_loader.py         # validated load interface
│   ├── performance_log.py        # daily log schema, phase schedule
│   └── wearable_data.py          # wearable schema and simulator
├── notebooks/
│   ├── chapter4_analysis.ipynb   # all Chapter 4 statistics and figures
│   └── figures/                  # 44 PNGs at 300 DPI
├── data/
│   ├── profiles/
│   │   ├── profiles.json         # 200 generated profiles
│   │   └── profiles.csv          # flattened equivalent
│   ├── comparison/
│   │   └── combined_spss.csv     # 41,600 rows — 13 models × 160 runners × 20 weeks
│   └── subtractive/
│       ├── tfn_no_physio_spss_export.csv    # 3,200 rows
│       └── tfn_no_fitness_spss_export.csv   # 3,200 rows
├── requirements.txt
└── README.md
```

---

## Reproduction

### Regenerate the synthetic cohort

```bash
pip install -r requirements.txt
python -m src.data_gen.profile_generator
```

Writes `data/profiles/profiles.json` and `profiles.csv`. Output is byte-identical to the committed files under seed 42. The generator depends only on the Python standard library.

### Recompute the published statistics and figures

```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace chapter4_analysis.ipynb
```

Regenerates all 44 figures into `notebooks/figures/` and prints every statistic reported in Chapter 4, including the omnibus tests, Dunn contrasts, ablation deltas, and the per-runner win analysis.

---

## Dataset schema

### `data/comparison/combined_spss.csv`

One row per runner per simulated week per model (41,600 rows).

| Field | Description |
|---|---|
| `runner_id` | Format `{age}-{gender}-{type}-{nn}` |
| `model` | Model identifier |
| `age`, `gender`, `profile_type`, `age_division` | Runner attributes |
| `week_number`, `training_phase` | Simulation position (1–20; base/build/peak/taper) |
| `avg_adherence_score` | Primary dependent variable (0.0–1.0) |
| `adherence_rate` | Proportion of fully adherent sessions |
| `rmse_distance`, `rmse_rpe`, `rmse_weekly_volume` | Plan accuracy metrics |
| `precision`, `recall` | Plan recommendation metrics |
| `progressive_overload_compliance_rate` | Constraint compliance |
| `periodization_compliance_rate` | Constraint compliance |
| `ten_percent_compliance_rate`, `ten_percent_violation_count` | Constraint compliance |
| `adaptation_success` | Adaptation outcome flag |
| `disruption_injury`, `disruption_abstention`, `disruption_partial` | Disruption counts |
| `weekly_volume_actual_mi` | Actual weekly volume |
| `entries_logged` | Log entries contributing to the row |

### `data/profiles/profiles.json`

One object per runner. Physiological attributes (BMI, resting heart rate, VO₂ proxy, estimated maximum heart rate), training history (training age, weekly mileage baseline, long run baseline), nullable prior race times (5K, 10K, half marathon, marathon), lifestyle attributes (sleep hours, sleep quality, stress, hydration, macronutrient ratio), biomechanical ratings, contextual narrative fields, and a nested `preferences` object covering rest days, activity exclusions, intensity ceilings, schedule availability, and goal context.

---

## Citation

```bibtex
@phdthesis{szpunar2026talaria,
  author      = {Szpunar, Sean Eben Aloysius},
  title       = {Digital Talaria: Developing and Evaluating a Physiologically
                 Grounded Machine Learning Framework for Adaptive Running
                 Plan Generation},
  school      = {National University},
  address     = {San Diego, California},
  year        = {2026},
  type        = {PhD dissertation}
}
```

---

## Primary references

Banister, E. W., Calvert, T. W., Savage, M. V., & Bach, T. (1975). A systems model of training for athletic performance. *Australian Journal of Sports Medicine, 7*(3), 57–61.

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.

Daniels, J. (2005). *Daniels' running formula* (2nd ed.). Human Kinetics.

Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. *Proceedings of the 33rd International Conference on Machine Learning*, 1050–1059.

Kang, W.-C., & McAuley, J. (2018). Self-attentive sequential recommendation. *Proceedings of the IEEE International Conference on Data Mining*, 197–206.

Kruskal, W. H., & Wallis, W. A. (1952). Use of ranks in one-criterion variance analysis. *Journal of the American Statistical Association, 47*(260), 583–621.

Noakes, T. (2003). *Lore of running* (4th ed.). Human Kinetics.

Riegel, P. S. (1981). Athletic records and human endurance. *American Scientist, 69*(3), 285–290.

Tanaka, H., Monahan, K. D., & Seals, D. R. (2001). Age-predicted maximal heart rate revisited. *Journal of the American College of Cardiology, 37*(1), 153–156.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems, 30*.

Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph attention networks. *International Conference on Learning Representations*.
