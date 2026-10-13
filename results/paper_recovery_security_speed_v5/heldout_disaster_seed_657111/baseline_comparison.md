# Security-speed v5 held-out disaster comparison

Single training seed; held-out diagnostic only, not confirmatory paper evidence.

| Method | Late TPS | Damaged TPS | Post failure | Mean actual exposure | Max exposure | Guard > f rate |
|---|---:|---:|---:|---:|---:|---:|
| PPO deterministic primary | 312.73 | 262.68 | 11.65% | 0.006 | 2 | 0.0000% |
| PPO stochastic audit | 287.42 | 256.69 | 13.98% | 0.000 | 1 | 0.0000% |
| Risk-Prior | 150.51 | 123.90 | 45.70% | 0.014 | 2 | 0.0000% |
| FMG | 154.52 | 138.17 | 40.66% | 0.343 | 6 | 0.0000% |
| CMAB | 252.91 | 196.67 | 23.74% | 0.068 | 4 | 0.0000% |
| SA | 232.28 | 192.42 | 25.01% | 0.309 | 7 | 0.0100% |
| GBA | 247.64 | 230.55 | 21.20% | 0.530 | 6 | 0.0000% |
| OCD-inspired | 151.86 | 142.42 | 35.75% | 0.438 | 4 | 0.0000% |
| Random | 196.23 | 151.70 | 34.96% | 4.198 | 10 | 7.0600% |
| Static-with-valid-refill | 141.56 | 127.30 | 47.64% | 1.548 | 7 | 0.0250% |

Best late-window baseline: **CMAB** (252.91 TPS).

Strict security guard means no held-out post-event step selected more than f=6 compromised committee members. Hidden identities were used only for this offline audit.
