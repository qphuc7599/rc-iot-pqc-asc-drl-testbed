# Minor revision: updated seven-claim verification

Verified 12 October 2026. This ledger derives from sealed local evidence and primary sources; it does not execute an environment, training run, formal audit, or blockchain transaction. The previous 5 October ledger is preserved, including its historical wording. New CSV provenance is in `../../05_completion_20261012/derived_export_validation.json`.

| Claim | Result | Required interpretation |
|---|---|---|
| A: FMG/CMAB/Risk-Prior fairness | PASS, qualified | Same delayed public-observation and physical-validity contract, not an identical feature subset. FMG uses fixed public features, CMAB public node context, Risk-Prior fixed coefficients. Comparator differences are not isolated component effects. |
| B: nine policies and markers | PASS | New `heldout_all_policies.csv` retains all nine policies and raw zero-TPS counts. PPO star, CMAB square, FMG/GBA/OCD/Risk-Prior neutral circles; SA/Random/Static warning markers. Guard failures and raw throughput are both retained. |
| C: 602 records | PASS | Two executed updates contain 310 and 292 records. Total 602; mean 301, not 301 in each update. |
| D: radio configuration | PASS, qualified | UDP/IPv4, 802.11g ERP-OFDM 6 Mb/s, tested frames below RTS/CTS threshold, CWmin 15. Nine- and twenty-microsecond slots are analytical sensitivity cases, not two measured radio traces. |
| E: actual generation | PASS, qualified | 7,421.829806 / 10,000 = 74.21829806% over the 60-s collection window. Producer logs specify 45-s generation + 15-s drain; this is not a 26% active-generator shortfall. Logged production averages approximately 9,896/s during its active window. |
| F: verifier gas | PASS, qualified | Approximately 8.1M gas is the cited non-packed implementation's ML-DSA-44 benchmark. It is not universal or current-best. The current pinned README also reports a cheaper experimental packed implementation. No external verifier benchmark was executed locally. |
| G: compact versus ABI | PASS after text correction | Compact header is 32 + 8 + 32 = 72 B; ML-DSA-44 signature is 2,420 B; total 2,492 B. Fourteen-signature fallback is 34,066 B. Actual local adapter ABI update is 34,244 B. These are distinct objects, not contradictory measurements. |

## Exact evidence and negative outcomes

Source files: `drl/paper_recovery_hardcore_v5/baselines.py`, `drl/paper_recovery/env.py`, `experiments/paper_recovery_hardcore_v5/protocol.json`, `ns3/scratch/committee-response-trace-v12.cc`, `results/revision_system_factorial/factorial_summary.json`, and `results/e2e_settlement_v1/20260730T151005/validation.json`. Their old sealed hashes remain in the historical ledger; new export validation binds current inputs and output hashes. The executed Solidity adapter uses a padded uint64 sequence, extra digest fields and a dynamic committee bundle, unlike the analytical compact tuple.

The original checkpoint's nine-policy panel preserves actual global maximum exposure and post-event violating steps: PPO 2/0, CMAB 4/0, FMG 6/0, GBA 6/0, OCD 4/0, Risk-Prior 2/0, SA 7/2, Random 10/1412, Static 7/5. Security budget is six. Integrity PASS is not a statement that every policy satisfies its security guard. No guard failure is used to force TPS to zero.

Original parent checkpoint SHA-256: `eabcd94a6c1a86c67a78241f894c3e9580905f04f71f8518bb32a1a45d40467b`; parent evaluation SHA-256: `60cba893a83a4ab6726d0252fee651d1e91fa2999862ce5c68e73dace00b717b`. The twelve-checkpoint extension and retrospective selected-switch configuration exports are separate from this original panel. No A4/A5 result or physical SRAM measurement is implied.

## Primary-source citation qualification

[ETHDILITHIUM README, pinned commit f9f4fe1d84e68304b425e054c273c276fe44ba11](https://github.com/ZKNoxHQ/ETHDILITHIUM/blob/f9f4fe1d84e68304b425e054c273c276fe44ba11/README.md) identifies the non-packed benchmark and a newer experimental packed branch. The [pinned verifier interface](https://github.com/ZKNoxHQ/ETHDILITHIUM/blob/f9f4fe1d84e68304b425e054c273c276fe44ba11/src/ZKNOX_dilithium.sol) requires a 2,420-byte signature, identifying ML-DSA-44 rather than ML-DSA-65/87.

[EIP-8051](https://eips.ethereum.org/EIPS/eip-8051) remains a draft; its proposed 4,500-gas verification and 20,512-byte expanded key are not executed Q-Relay measurements. The historical 14 × 8.1M estimate is conditional on the non-packed implementation, not proof that every fallback must exceed a transaction cap.

The Zenodo API resolves concept DOI [10.5281/zenodo.20390757](https://doi.org/10.5281/zenodo.20390757) to version record 20390758, titled “RC-IoT Post-Quantum ASC-DRL Experimental Testbed”. This existing systems archive is not a deposit of the new ablations or release candidate. Public release/tag/deposit remains pending the author's licence choice.
