# Feature Ablation and Reparameterization Plan

## Branch

- Working branch: `exp/feature-ablation-reparam`

## Objective

This branch is dedicated to one question:

> Can the current `log(psi)` mainline be reduced from the present 21 numeric features to a smaller, more physically coherent feature set without giving back the current performance advantage?

The target is not "as few features as possible at any cost". The target is:

1. keep the current training target definition:
   - `target_mode: psi_over_npl`
   - `target_transform: log`
2. reduce feature redundancy
3. improve feature interpretability
4. preserve or improve `CV composite objective`
5. preserve regime stability, especially on:
   - `scale_npl`
   - `section_family`
   - `slenderness_state`
   - `eccentricity_severity`

## Fixed Baseline

The current frozen baseline for all comparisons is the validated `log(psi)` mainline on `main`:

- training target: `log(psi)`
- reported target: `Nexp (kN)`
- current numeric feature count: `21`
- current baseline holdout metrics:
  - `RMSE = 229.11 kN`
  - `R2 = 0.9904`
  - `COV = 0.0905`
- current baseline CV selection metrics:
  - `composite_objective = 1.5051`
  - `R2 = 0.9839`
  - `COV = 0.1022`

Reference artifacts:

- `output/psi_over_npl_log_original_200/`
- `config/config.yaml`
- `config/experiments/log_original_metric.yaml`

## Scope

Included in this branch:

- feature ablation on the current numeric feature set
- feature reparameterization
- small supporting code changes when needed
- experiment configs / experiment matrix support
- result reporting and documentation sync

Explicitly out of scope for this branch:

- `Ref.No.` grouped split
- axial / eccentric dual-model training framework
- changing the default sample-level split protocol
- replacing XGBoost with a different model family

## Current Feature State

The current training pipeline uses 21 numeric features:

1. `R (%)`
2. `fy (MPa)`
3. `fc (MPa)`
4. `e1 (mm)`
5. `e2 (mm)`
6. `r0/h`
7. `b/t`
8. `Ac (mm^2)`
9. `As (mm^2)`
10. `Re (mm)`
11. `te (mm)`
12. `ke`
13. `xi`
14. `sigma_re (MPa)`
15. `lambda_bar`
16. `e/h`
17. `e1/e2`
18. `e_bar`
19. `Npl (kN)`
20. `b/h`
21. `L/h`

Two additional derived columns currently exist for diagnostics but do not enter the model by default:

- `axial_flag`
- `section_family`

## Working Hypotheses

### Hypothesis A: the current 21-feature set is wider than necessary

The present feature importance ranking is strongly concentrated at the top, which suggests there is room to compress the feature set.

### Hypothesis B: `psi = Nexp / Npl` makes some absolute-scale features less necessary

Once the target is normalized by `Npl`, the model may benefit more from relative composition variables than from keeping every absolute geometry / material magnitude in raw form.

### Hypothesis C: the present eccentricity description is over-parameterized

The current eccentricity family contains:

- `e1 (mm)`
- `e2 (mm)`
- `e/h`
- `e1/e2`
- `e_bar`

This is likely redundant. A more compact representation may preserve performance while improving interpretability.

### Hypothesis D: physically meaningful ratios may outperform some raw inputs

Likely useful reparameterized candidates:

- `axial_indicator = 1 if e_bar == 0 else 0`
- `steel_area_ratio = As / Ac`
- `strength_ratio = fy / fc`
- `steel_capacity_share = As * fy / (As * fy + Ac * fc)`

These are better aligned with the new `psi / Npl` target definition than simply keeping every raw absolute quantity.

## Success Criteria

This branch should not select a winner based on test metrics alone.

Primary selection basis:

- `CV composite objective`

Secondary gates:

- no meaningful deterioration in `CV R2`
- no meaningful deterioration in `CV COV`
- no clear collapse in worst-group regime metrics

Tie-break rule:

- if two candidates are effectively comparable, prefer the one with fewer features and clearer physical meaning

## Experimental Strategy

The branch will use a three-stage process.

### Stage 0: Freeze and Reproduce Baseline

Purpose:

- keep one stable reference point before touching features

Actions:

1. freeze the current 21-feature baseline as the comparison anchor
2. keep the current `log(psi)` training target and current selection objective unchanged
3. keep sample-level split, CV, and regime schema unchanged

### Stage 1: Delete-Only Ablation

Purpose:

- answer the narrow question: which of the three new mainline features are actually earning their place?

Historical 18-feature base:

- current 21-feature set minus:
  - `Npl (kN)`
  - `b/h`
  - `L/h`

Delete-only matrix:

1. `D0`: historical `18`
2. `D1`: historical `18` + `Npl (kN)`
3. `D2`: historical `18` + `b/h`
4. `D3`: historical `18` + `L/h`
5. `D4`: historical `18` + `Npl (kN)` + `b/h`
6. `D5`: historical `18` + `Npl (kN)` + `L/h`
7. `D6`: historical `18` + `b/h` + `L/h`
8. `D7`: full current `21` features

Promotion rule:

- promote the best `2-3` candidates by `CV composite objective`
- reject candidates that create obvious regime instability even if their global score looks acceptable

Expected outcome:

- establish whether the real baseline should remain at `21`, or naturally shrink to `18-20`

### Stage 2: Reparameterization Experiments

Purpose:

- replace correlated raw inputs with more physically coherent compact representations

This stage will start from the best Stage 1 candidate, not necessarily from the full current `21`.

#### Family R1: Eccentricity Compression

Current family:

- `e1 (mm)`
- `e2 (mm)`
- `e/h`
- `e1/e2`
- `e_bar`

Candidate compact family:

- `e/h`
- `e1/e2`
- `e_bar`
- `axial_indicator`

Key test:

- can `e1 (mm)` and `e2 (mm)` be removed once normalized eccentricity information is retained?

#### Family R2: Material and Composition Reparameterization

Current family:

- `fy (MPa)`
- `fc (MPa)`
- `Ac (mm^2)`
- `As (mm^2)`
- `Npl (kN)`

Candidate derived variables:

- `strength_ratio = fy / fc`
- `steel_area_ratio = As / Ac`
- `steel_capacity_share = As * fy / (As * fy + Ac * fc)`

Key tests:

1. keep `Npl (kN)` and replace part of the raw material / area family with ratio-style features
2. test whether `Ac (mm^2)` and `As (mm^2)` can be removed after the share variables are introduced
3. test whether separate `fy (MPa)` and `fc (MPa)` remain necessary after `strength_ratio` is introduced

#### Family R3: Geometry and Shape Compaction

Current geometry / shape family:

- `r0/h`
- `b/t`
- `Re (mm)`
- `te (mm)`
- `b/h`

Candidate strategy:

1. keep the dimensionless descriptors as the main shape representation:
   - `r0/h`
   - `b/t`
   - `b/h`
2. evaluate whether `Re (mm)` and `te (mm)` still earn their place after `Npl` and composition features are retained
3. if necessary, add numeric encoding for `section_family` later in the branch, but only after the simpler numeric-only tests are exhausted

#### Compact Reparameterized Target Candidate

One explicit compact target set should be tested, with an expected size around `14-16`:

1. `R (%)`
2. `strength_ratio`
3. `steel_capacity_share`
4. `ke`
5. `xi`
6. `sigma_re (MPa)`
7. `lambda_bar`
8. `L/h`
9. `r0/h`
10. `b/h`
11. `b/t`
12. `Npl (kN)`
13. `e/h`
14. `e1/e2`
15. `axial_indicator`

Optional companion variant:

- remove `Npl (kN)` from the compact set and compare directly

This is the main hypothesis-driven compact model for the branch.

### Stage 3: Narrowed Hyperparameter Search

Purpose:

- avoid spending full 200-trial Optuna on obviously weak feature sets

Screening protocol:

1. Stage 1 and Stage 2 candidates first run with the current validated `log(psi)` baseline hyperparameters
2. select the top `2-3` candidates by `CV composite objective`
3. run moderate retuning on finalists
4. only the final winner and one runner-up receive the full `200`-trial search

This keeps the comparison fair while avoiding needless search cost.

## Implementation Tasks Anticipated

Likely code changes required on this branch:

1. add new domain-derived numeric columns:
   - `axial_indicator`
   - `steel_area_ratio`
   - `strength_ratio`
   - `steel_capacity_share`
2. allow experiment configs to define explicit feature include lists or feature-set presets
3. add or update tests for the new derived columns and feature-set selection behavior
4. keep predictor compatibility when the winning feature set changes
5. export experiment summaries in a way that makes feature-set comparison easy

## Selection and Reporting Rules

For every candidate feature set, record:

1. feature count
2. exact feature list
3. CV:
   - composite objective
   - RMSE
   - R2
   - COV
4. holdout:
   - RMSE
   - R2
   - COV
5. regime summary:
   - worst `scale_npl`
   - worst `section_family`
   - worst `slenderness_state`
   - worst `eccentricity_severity`

Decision rule:

- do not select a feature set just because it gives a slightly better global score if it clearly worsens a known hard regime

## Expected Outcome

This branch is expected to converge to one of three outcomes:

1. the current `21` features remain the most robust option
2. a cleaned delete-only set in the `18-20` range wins
3. a compact reparameterized set in the `14-16` range wins

The third outcome is the preferred target, but it is not assumed in advance.

## Deliverables

Before merging anything back:

1. feature experiment configs or presets
2. updated tests
3. a comparison report covering all finalists
4. synced docs if the winning feature definition changes
