# Q-Relay virtualised emulation artifact - R1

Version v1.1.0. Source/configuration/formal models: MIT. Group-owned data,
checkpoints, plots and technical documentation: CC-BY-4.0. See LICENSE_SCOPE.md,
THIRD_PARTY_NOTICES.md and the exact LICENSE_ASSIGNMENTS.json per-file map.

This artifact is independent of the manuscript submission package. Manuscripts,
reviewer response, cover letter, portraits and withdrawn B4 node-sweep results
are excluded. Node-count scaling remains analytical; no measured knee is claimed.

## Evidence

- Original controller: `results/paper_recovery_security_speed_v5/heldout_disaster_seed_657111/evaluation.json`.
- A0-A3: `revision_R1_results/03_ablation/final_campaign_export_20261012/` contains
  twelve checkpoint manifests, 240 paired outcomes and the two-page artifact report.
- B3: `revision_R1_results/04_optional/inference_sensitivity_20261012/` contains
  360 new inference-only trajectories and forty reused defaults, including negatives.
- Claims and frozen-stream audit: `revision_R1_results/02_verification/completion_20261012/`.
- Local gas: `revision_R1_results/04_optional/ledger_export_20261012/`.
- Threshold/functional/model evidence: `results/paper_recovery_threshold_v3/`,
  `results/stage_bc/`, `results/e2e_settlement_v1/` and `formal/`.
- Analytical bounds: `analysis/overhead_scaling.py`.

Three training checkpoints per arm are the independent training replicates, not
the sixty shared-scenario trajectories. All three primary hierarchical intervals
include zero and none is Holm-significant. History removal descriptively raises
late TPS; B3 unfavorable settings and CMAB guard violations are retained. No
unseen-population robustness or safety guarantee follows from these finite tests.

## Reproduction boundary

Use Python 3.12, NumPy and PyTorch; recorded training used Ubuntu/WSL, RTX2050,
11 physical workers/33 logical environments and 9M transitions/checkpoint. A
single-scenario command is:

`python experiments/minor_revision_completion_20261012/reproduce_one_candidate.py --execute`

This creates a once-only smoke in a new folder, not formal campaign output. Its
expected seed657201 late TPS is 231.47056412218697, not the twenty-scenario mean.
Do not automatically run old training/evaluation launchers, replay sealed panels
or overwrite data. The already completed clone-overlay smoke is supplied in
INFERENCE_SMOKE_VALIDATION.json; it was not rerun for publication. Import/source/
checkpoint-only qualification may run without `--execute` and creates no episode.

PUBLIC_SOURCE_PROVENANCE.json distinguishes original scientific hashes, prepared
candidate hashes and distributed hashes. Privacy redactions affect metadata only,
never executable scientific code, configurations or checkpoint bytes. A redacted
receipt retains its original historical receipt_hash as an identifier; it is NOT
claimed to be a byte-identical or canonically identical original receipt.

Real threshold ML-DSA was executed only at n<=6. A4/A5, physical SRAM/stack,
enclave execution, native on-chain ML-DSA and public-chain finality are unmeasured.
The mixed archive has an exact file manifest and licence map. Its new archived
version DOI is linked from the GitHub release when actually minted and verified;
the historical concept DOI alone does not identify this extension.
