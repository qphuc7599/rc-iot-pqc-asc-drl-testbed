# Stage B/C evidence audit

This report separates executed evidence from calibrated simulation, analytical sensitivity, and unexecuted deployment dependencies.

| Item | Status | Evidence | Validity boundary |
|---|---|---|---|
| Fanout enforcement | Executed | 10,000 epochs; 30,000 compromise checks | Admission layer only |
| Asynchronous fault injection | Executed | 5,000 replicates x 7 faults | Approval flags represent already-verified cryptographic layers; this campaign isolates asynchronous ordering and persistence. |
| Archive failover | Executed | 62 objects; p95 RTO 4.076 ms; RPO=0 | Replicas have persistent processes, databases, user identities, and Linux network namespaces, but still share one physical host. |
| Quorum/settlement model | Executed finite model + bounded TLC | Q13 min intersection=5; Q14=7; TLC safe states=7 + unsafe counterexample | TLC exhaustively checks the tractable homologous boundary n=7,f=2 (Q=5 safe, Q=4 counterexample). The separate exact combinatorial checker evaluates the manuscript instance n=21,f=6,Q=13/14. |
| Backend rounds/bytes | Analytical Monte Carlo sensitivity | 36 cells x 100,000 draws | Each cell is a conservative shared-uplink network-only lower bound. It does not claim that TALUS, Quorus, Mithril, or another threshold implementation was executed. |
| Unsaturated SLA/batch sweep | Calibrated discrete-event simulation | 12 cells x 8 replicates x 50,000 batches | Anchored to the executed primary path; not a new physical deployment |
| Equal-budget baseline tuning | Executed | 4 families; 600,000 interactions/family | OCD-inspired node-selection heuristic only. Native OCD in Xie et al. maps cross-chain transactions to pre-existing committees via CRC-MN, RC-ST, and CET-CD, so it is not a drop-in node selector. |
| Equal-budget held-out policy-aware panel | Executed development gate | 20 paired held-out seeds; PPO rank 2/5 | One PPO training seed; development evidence only; no confirmatory superiority claim |
| Local-EVM settlement adapter | Executed | 10 clean deterministic repetitions; 200 checks | 10 deterministic local-EVM repetitions reproduce the same functional decisions and gas. They are reproducibility repeats, not independent public-chain samples; native ML-DSA, TEE attestation, and economic finality remain unexecuted. |

## Remaining physical/backend gates

| Gate | Status | Reason |
|---|---|---|
| Native threshold ML-DSA backend | Not executed | No compatible backend implementation/toolchain is present |
| Native chain ML-DSA verifier/precompile | Not executed | Current EVM path uses a host-validated digest registry |
| Bare-metal/HIL | Not executed | No target boards or ARM embedded toolchain are locally available |
| Physically independent archive sites | Not executed | Network namespaces isolate processes but share one host |

## Tool availability snapshot

- Docker daemon: reachable; containers would still share this physical host
- Java/TLC: executed; see the hashed TLC artifact
- ARM GCC: unavailable

No unexecuted item above is promoted to an empirical result.
