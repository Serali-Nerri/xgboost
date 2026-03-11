# Eccentricity Reparameterization Round

Date: 2026-03-11

## Objective

This round targeted only the eccentricity-related feature family.

The goal was not to change the whole feature system again, but to answer one narrower question:

> Can the current eccentricity descriptors be replaced by a more structured, physically coherent encoding while preserving or improving the current `log(psi)` mainline?

## Fixed Anchors

### Current default mainline

- anchor: `D7_current21`
- feature count: `21`
- eccentricity family:
  - `e1 (mm)`
  - `e2 (mm)`
  - `e/h`
  - `e1/e2`
  - `e_bar`

Metrics:

- `CV J = 1.5051`
- `CV R2 = 0.9839`
- `CV COV = 0.1022`
- `Test RMSE = 229.11 kN`
- `Test R2 = 0.9904`
- `Test COV = 0.0905`

Reference:

- `output/psi_over_npl_log_original_200/evaluation_report.json`

### Best compact baseline from the prior round

- anchor: `D6_hist18_plus_b_over_h_l_over_h`
- feature count: `20`

Metrics:

- `CV J = 1.3797`
- `CV R2 = 0.9845`
- `CV COV = 0.1019`
- `Test RMSE = 251.33 kN`
- `Test R2 = 0.9884`
- `Test COV = 0.0898`

Reference:

- `output/feature_ablation/finalists/D6_hist18_plus_b_over_h_l_over_h/evaluation_report.json`

## New Eccentricity-Derived Features

This round used the following structured eccentricity descriptors:

- `e_min/h`
- `end_asymmetry_ratio`
- `single_curvature_e/h`
- `double_curvature_e/h`
- `reverse_curvature_flag`

Interpretation:

- `e_min/h`: weaker-end normalized eccentricity magnitude
- `end_asymmetry_ratio`: ratio of weaker-end to stronger-end eccentricity
- `single_curvature_e/h`: same-sign / same-bending component
- `double_curvature_e/h`: reverse-curvature component
- `reverse_curvature_flag`: whether end eccentricities have opposite signs

## Screening Matrix

### D7-based variants

1. `E1_d7_mode_balance_keep_ebar`
2. `E2_d7_asymmetry_flag_keep_ebar`
3. `E3_d7_mode_balance_no_ebar`

### D6-based variants

1. `E4_d6_mode_balance_keep_ebar`
2. `E5_d6_asymmetry_flag_keep_ebar`

## Screening Results

| ID | n_features | CV J | CV R2 | CV COV | Test RMSE | Test R2 | Test COV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E1_d7_mode_balance_keep_ebar | 21 | 1.5412 | 0.9837 | 0.1023 | 223.33 | 0.9909 | 0.0880 |
| E2_d7_asymmetry_flag_keep_ebar | 21 | 1.6276 | 0.9834 | 0.1022 | 234.99 | 0.9899 | 0.0889 |
| E3_d7_mode_balance_no_ebar | 21 | 1.6954 | 0.9829 | 0.1018 | 242.20 | 0.9893 | 0.0878 |
| E5_d6_asymmetry_flag_keep_ebar | 20 | 1.6482 | 0.9831 | 0.1022 | 221.33 | 0.9910 | 0.0883 |
| E4_d6_mode_balance_keep_ebar | 20 | 1.6648 | 0.9832 | 0.1030 | 220.36 | 0.9911 | 0.0910 |

## Best Candidate

The best candidate to continue was:

- `E1_d7_mode_balance_keep_ebar`

Selected feature list:

- `R (%)`
- `fy (MPa)`
- `fc (MPa)`
- `r0/h`
- `b/t`
- `Ac (mm^2)`
- `As (mm^2)`
- `Re (mm)`
- `te (mm)`
- `ke`
- `xi`
- `sigma_re (MPa)`
- `lambda_bar`
- `e_bar`
- `Npl (kN)`
- `b/h`
- `L/h`
- `e_min/h`
- `end_asymmetry_ratio`
- `single_curvature_e/h`
- `double_curvature_e/h`

Why it was promoted:

- among the `D7`-based variants, it gave the strongest holdout result
- it preserved `e_bar`
- it replaced raw end eccentricities with a compact curvature-pattern representation

## Final Evaluation Using Best-So-Far Optuna Parameters

The original `200`-trial finalist run was interrupted before artifact save.

To close the loop cleanly, the best parameter set from the completed Optuna study was extracted after `103` completed trials and used for a fresh final evaluation run.

Best-so-far Optuna status:

- completed trials: `103`
- best `CV J`: `1.5092`

Best-so-far parameters:

- `max_depth = 6`
- `learning_rate = 0.031955818004798305`
- `n_estimators = 4452`
- `subsample = 0.7866734105919365`
- `colsample_bytree = 0.4591557378871465`
- `min_child_weight = 4`
- `reg_alpha = 0.0031543096015525514`
- `reg_lambda = 0.05198884625845396`
- `gamma = 0.00013407670223416032`

Final evaluation metrics for `E1_d7_mode_balance_keep_ebar`:

- `CV J = 1.5092`
- `CV R2 = 0.9840`
- `CV COV = 0.1026`
- `Test RMSE = 214.56 kN`
- `Test R2 = 0.9916`
- `Test COV = 0.0866`

Reference:

- `output/eccentricity_reparameterization/final_eval_103trials/E1_d7_mode_balance_keep_ebar/evaluation_report.json`

## Interpretation

This round gives a much stronger answer than the previous broad reparameterization attempts.

### What clearly worked

1. Eccentricity can be reparameterized more intelligently.
2. The combination below is materially better than the raw `e1 / e2 / e1-e2 ratio` bundle on the current holdout:
   - `e_bar`
   - `e_min/h`
   - `end_asymmetry_ratio`
   - `single_curvature_e/h`
   - `double_curvature_e/h`
3. The resulting model clearly satisfies the practical target:
   - `R2 > 0.99`
   - `COV < 0.10`

### What did not happen

1. The best-so-far tuned candidate did **not** clearly beat the current mainline on the CV composite objective.
2. The CV improvement target was nearly reached, but not crossed:
   - mainline `D7`: `CV J = 1.5051`
   - best-so-far `E1`: `CV J = 1.5092`

### What improved anyway

Compared with the current default mainline:

- `Test RMSE`: `229.11 -> 214.56 kN`
- `Test R2`: `0.9904 -> 0.9916`
- `Test COV`: `0.0905 -> 0.0866`

This is a meaningful holdout improvement.

## Conclusion

The answer to the original question is now clearer:

> Yes, the eccentricity feature family was compressible, but only after changing the representation, not by naive deletion.

Current recommendation:

1. Keep the current mainline unchanged on `main` for now, because the new candidate has not yet produced a clearly better `CV J`.
2. Treat `E1_d7_mode_balance_keep_ebar` as the strongest eccentricity-reparameterized challenger discovered so far.
3. If a next step is taken, it should be a repeated-split / repeated-CV comparison between:
   - current `D7_current21`
   - `E1_d7_mode_balance_keep_ebar`

This round is therefore a positive result:

- not a failed compression attempt
- not a final mainline replacement yet
- but a genuine improvement in how eccentricity should be encoded
