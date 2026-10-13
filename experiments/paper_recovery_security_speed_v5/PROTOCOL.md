# Security-v4 high-throughput runtime v5

The scientific controller and disaster environment are unchanged. The runtime
keeps 11 physical Python processes but places three independent environments
in each process. This raises the action batch from 11 to 33 and reduces policy
forward/IPC rounds per 3,000-transition update from 273 to 91.

Benchmark seed 657011 runs 20 updates (60,000 transitions), so every logical
environment crosses the original permanent 30% kill at step 1,000. Formal seed
657111 can be claimed only if both the post-warmup and last-ten-update rates are
at least 400 step/s and the conservative 9M projection is at most 6.5 hours.
