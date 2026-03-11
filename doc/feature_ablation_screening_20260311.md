# Feature Ablation and Reparameterization Screening

Date: 2026-03-11

## Baseline

- Base config: `config/config.yaml`
- Training target: `log(psi)`
- Selection basis: CV composite objective in original `Nexp` space

## Stage 1 Delete-Only Ablation

| ID | n_features | CV J | CV R2 | CV COV | Test R2 | Test COV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D6_hist18_plus_b_over_h_l_over_h | 20 | 1.4846 | 0.9839 | 0.1020 | 0.9895 | 0.0899 |
| D7_current21 | 21 | 1.5051 | 0.9839 | 0.1022 | 0.9904 | 0.0905 |
| D5_hist18_plus_npl_l_over_h | 20 | 1.6075 | 0.9833 | 0.1028 | 0.9894 | 0.0892 |
| D3_hist18_plus_l_over_h | 19 | 1.6648 | 0.9831 | 0.1035 | 0.9911 | 0.0914 |
| D4_hist18_plus_npl_b_over_h | 20 | 1.7925 | 0.9826 | 0.1042 | 0.9899 | 0.0907 |
| D2_hist18_plus_b_over_h | 19 | 2.0028 | 0.9815 | 0.1048 | 0.9887 | 0.0934 |
| D1_hist18_plus_npl | 19 | 2.0429 | 0.9813 | 0.1052 | 0.9897 | 0.0932 |
| D0_hist18 | 18 | 2.2254 | 0.9805 | 0.1053 | 0.9885 | 0.0944 |

## Stage 2 Reparameterization Screening

| ID | n_features | CV J | CV R2 | CV COV | Test R2 | Test COV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R4_geometry_compact_v1 | 18 | 1.5130 | 0.9838 | 0.1031 | 0.9891 | 0.0913 |
| R1_eccentricity_compact_v1 | 19 | 1.6622 | 0.9831 | 0.1034 | 0.9904 | 0.0906 |
| R5_compact_reparam_v1 | 15 | 2.4408 | 0.9798 | 0.1109 | 0.9847 | 0.0953 |
| R2_material_ratios_v1 | 18 | 2.6177 | 0.9788 | 0.1082 | 0.9876 | 0.0935 |
| R3_material_ratios_v2 | 19 | 2.6270 | 0.9787 | 0.1078 | 0.9864 | 0.0935 |
| R6_compact_reparam_v1_no_npl | 14 | 2.9172 | 0.9777 | 0.1127 | 0.9831 | 0.0987 |

## Finalist 200-Trial Check

Only one new finalist required a full rerun: `D6_hist18_plus_b_over_h_l_over_h`.

`D7_current21` already had an existing validated `200`-trial mainline result in `output/psi_over_npl_log_original_200/`, so it was used as the comparison anchor instead of burning another redundant `200`-trial run.

| ID | n_features | CV J | CV RMSE | CV R2 | CV COV | Test RMSE | Test R2 | Test COV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D6_hist18_plus_b_over_h_l_over_h | 20 | 1.3797 | 327.80 | 0.9845 | 0.1019 | 251.33 | 0.9884 | 0.0898 |
| D7_current21 mainline | 21 | 1.5051 | 333.37 | 0.9839 | 0.1022 | 229.11 | 0.9904 | 0.0905 |

## Conclusions

1. The best delete-only ablation was `D6 = Hist18 + b/h + L/h`, not the current 21-feature default.
2. `Npl (kN)` was not a necessary winner under this branch's sample-level protocol. The strongest compact screening candidate removed it.
3. Reparameterization did not beat delete-only ablation in this round. The material-ratio and compact reparameterized sets were clearly worse.
4. The `200`-trial rerun confirmed that `D6` improves the CV composite objective substantially, but it did **not** improve final holdout `RMSE` / `R2` versus the existing `D7_current21` mainline.
5. `D6` did slightly improve holdout `COV` (`0.0898` vs `0.0905`), but the loss in holdout `RMSE` and `R2` is too large to justify replacing the current mainline.

## Decision

- Keep `D7_current21` as the default mainline for now.
- Keep `D6_hist18_plus_b_over_h_l_over_h` as the most credible compact alternative discovered on this branch.
- If this branch is continued later, the next serious follow-up should be repeated-split or repeated-CV comparison between `D6` and `D7`, because the current evidence is:
  - `D6` wins on CV selection objective
  - `D7` wins on the present holdout split
