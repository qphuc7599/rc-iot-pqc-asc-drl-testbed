# Batch 11: A1 no history - seed 657123

Same registered 20 scenarios and 2000 steps. Three A0, three A2, three A3 and both previous A1 checkpoints, and eight baseline panels were reused, not rerun.

| Method | Late TPS | Zero service (%) | Failure (%) | Max exposure | Violation (%) | Guard f=6 |
|---|---:|---:|---:|---:|---:|---|
| A3 no security critic / 657142 | 324.27 | 11.78 | 11.77 | 2 | 0.000 | PASS |
| A1 no history / 657122 | 323.03 | 11.18 | 11.17 | 1 | 0.000 | PASS |
| A0 full / 657111 | 312.73 | 11.65 | 11.65 | 2 | 0.000 | PASS |
| A0 full / 657112 | 312.63 | 12.46 | 12.46 | 2 | 0.000 | PASS |
| A1 no history / 657123 | 309.41 | 10.07 | 10.08 | 0 | 0.000 | PASS |
| A2 no admission / 657132 | 309.15 | 15.43 | 15.43 | 3 | 0.000 | PASS |
| A2 no admission / 657133 | 307.84 | 15.76 | 15.75 | 2 | 0.000 | PASS |
| A3 no security critic / 657143 | 307.39 | 14.12 | 14.12 | 3 | 0.000 | PASS |
| A2 no admission / 657131 | 300.38 | 18.00 | 18.00 | 3 | 0.000 | PASS |
| A1 no history / 657121 | 296.50 | 14.18 | 14.18 | 2 | 0.000 | PASS |
| A0 full / 657113 | 284.44 | 16.57 | 16.56 | 3 | 0.000 | PASS |
| A3 no security critic / 657141 | 282.76 | 16.46 | 16.47 | 2 | 0.000 | PASS |
| CMAB | 252.91 | 23.74 | 23.74 | 4 | 0.000 | PASS |
| GBA | 247.64 | 21.20 | 21.20 | 6 | 0.000 | PASS |
| SA | 232.28 | 25.01 | 25.01 | 7 | 0.010 | FAIL |
| Random | 196.23 | 34.95 | 34.96 | 10 | 7.060 | FAIL |
| FMG | 154.52 | 40.66 | 40.66 | 6 | 0.000 | PASS |
| OCD-inspired | 151.86 | 35.74 | 35.75 | 4 | 0.000 | PASS |
| Risk-Prior | 150.51 | 45.70 | 45.70 | 2 | 0.000 | PASS |
| Static-with-valid-refill | 141.56 | 47.64 | 47.64 | 7 | 0.025 | FAIL |

## Conditional paired scenario comparisons

Fixed checkpoints; 95% paired t intervals over20shared scenarios, unadjusted. Intervals are conditional on fixed checkpoints, not population training-seed intervals.

- A1/657123 minus A0 full / 657111: -3.31 TPS, 95% CI [-18.11, 11.48].
- A1/657123 minus A0 full / 657112: -3.22 TPS, 95% CI [-17.82, 11.38].
- A1/657123 minus A0 full / 657113: 24.97 TPS, 95% CI [11.49, 38.46].
- A1/657123 minus A2 no admission / 657131: 9.03 TPS, 95% CI [-6.38, 24.45].
- A1/657123 minus A2 no admission / 657132: 0.26 TPS, 95% CI [-14.94, 15.47].
- A1/657123 minus A2 no admission / 657133: 1.57 TPS, 95% CI [-13.62, 16.76].
- A1/657123 minus A3 no security critic / 657141: 26.65 TPS, 95% CI [12.31, 40.99].
- A1/657123 minus A3 no security critic / 657142: -14.86 TPS, 95% CI [-29.66, -0.06].
- A1/657123 minus A3 no security critic / 657143: 2.02 TPS, 95% CI [-20.47, 24.52].
- A1/657123 minus A1 no history / 657121: 12.91 TPS, 95% CI [1.46, 24.36].
- A1/657123 minus A1 no history / 657122: -13.62 TPS, 95% CI [-29.30, 2.06].
- A1/657123 minus CMAB: 56.50 TPS, 95% CI [35.88, 77.12].

## Equal-weight descriptive means across available training checkpoints

| Arm | Training seeds | Late TPS | Zero service (%) | Failure (%) | Maximum actual exposure | Violation (%) |
|---|---|---:|---:|---:|---:|---:|
| A1 no history | 657121,657122,657123 | 309.65 | 11.81 | 11.81 | 2 | 0.000 |
| A0 full | 657111,657112,657113 | 303.26 | 13.56 | 13.56 | 3 | 0.000 |
| A2 no admission | 657131,657132,657133 | 305.79 | 16.39 | 16.39 | 3 | 0.000 |
| A3 no security critic | 657141,657142,657143 | 304.81 | 14.12 | 14.12 | 3 | 0.000 |

All four arms have THREE independent training checkpoints each. The registered extension training/evaluation campaign is complete only after independent evidence and visual delivery validation. This extension is exploratory and uses an already observed shared panel; no unseen-panel or population-significance claim. The same 20 scenarios are paired outcomes, not independent training replicates; arm means are descriptive and paired intervals condition on fixed checkpoints.

Validation PASS establishes evidence integrity, not security-guard success. Preserve negative outcomes. Training step/s is not held-out disaster TPS.

## Registered exploratory three-checkpoint analysis

Hierarchy:3 independent training checkpoints per arm,20 shared paired environment scenarios. Bootstrap10,000 replicates resamples training checkpoints within arms and shared scenarios; percentile95 intervals. Exact two-sided sign flip enumerates all2^20 assignments on training-seed-averaged scenario differences; Holm adjusts ONLY the three primary lateTPS contrasts. Scenario p-values condition on these checkpoints, not a population of training seeds. Extension was planned after observing the original panel; no unseen-panel qualification or guaranteed/population significance claim.

| Contrast | Difference TPS | Hierarchical95% interval | Conditional scenario p | Holm p |
|---|---:|---:|---:|---:|
| A1_no_history minus A0_full | 6.382 | [-15.114, 28.226] | 0.049677 | 0.149031 |
| A2_no_admission minus A0_full | 2.524 | [-13.260, 22.175] | 0.442329 | 0.884659 |
| A3_no_security_critic minus A0_full | 1.541 | [-24.462, 28.650] | 0.704039 | 0.884659 |

Between-training-seed variability (SD of checkpoint mean lateTPS, n=3):
- A0_full: mean303.265 TPS; between-seed SD16.305.
- A1_no_history: mean309.647 TPS; between-seed SD13.267.
- A2_no_admission: mean305.789 TPS; between-seed SD4.731.
- A3_no_security_critic: mean304.806 TPS; between-seed SD20.873.
