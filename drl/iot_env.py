"""
iot_env.py — IoT Network Environment v16 (ML-DSA Rejection Sampling)

Scientifically rigorous IoT blockchain simulation with:
- Round-based Threshold Signature BFT consensus
- ML-DSA-44 signing model with rejection sampling variance
  (Fiat-Shamir with Aborts: stochastic long-tail latency)
- Dual bottleneck: CPU threshold-quorum latency + bandwidth (10 Mbps)
- Dynamic node churn, stealth Byzantine, battery cliff

TPS Model:
  Round time = propagation + t-th fastest honest sign time + propagation + verify
  TPS = min(CPU_TPS, bandwidth_TPS)
  Threshold quorum latency determines round speed
  -> PPO must keep a fast honest quorum available proactively
  -> Rejection sampling creates stochastic stragglers

Physical constraints modeled:
1. BATTERY — Realistic capacity (gateway=large, BLE sensor=coin cell)
   Drain proportional to actual signing workload * power_norm
   Li-ion voltage cliff at 20% SoC
2. NETWORK — 10 Mbps effective bandwidth, 5ms propagation
3. CPU — ML-DSA sign time per MCU type (architecture-calibrated)
4. BFT — f < ⌊n/3⌋ threshold with cubic degradation
5. SERVICE FATIGUE — Thermal throttling for high-power MCUs
6. NODE CHURN — IoT mobility, signal loss, partition recovery
7. STEALTH BYZANTINE — APT with delayed activation
"""

import numpy as np
import subprocess
from collections import Counter  # used by external scripts
import gymnasium as gym
from gymnasium import spaces


class IoTNetworkEnv(gym.Env):
    """
    IoT Network Environment for DRL-based BFT committee selection.
    
    TIMESTEP: 1 step = 1 minute of real-world operation.
    - 1000 steps = 16.7 hours (training episode)
    - 2000 steps = 33.3 hours (evaluation episode)
    - Battery drain calibrated to this timescale.
    """

    # Device specs — ML-DSA-44 latency calibrated from pqm4 + architecture penalties
    #
    # METHODOLOGY (replacing naive linear clock scaling):
    #   1. Baseline: pqm4 ML-DSA-44 (m4f optimized) cycle counts on Cortex-M4
    #      keygen=1,426,025  sign=3,943,121  verify=1,421,623 cycles
    #   2. Architecture-specific IPC penalty factors (vs Cortex-M4):
    #      - Cortex-M7:  0.55× (dual-issue superscalar, 6-stage pipeline)
    #      - Cortex-M4:  1.00× (pqm4 reference platform, m4f assembly)
    #      - Cortex-M4F: 1.00× (same ISA, nRF52840 = M4F @ 64 MHz)
    #      - Cortex-M0+: 1.85× (no UMULL/SMULL, no DSP/SIMD, Thumb-1 only)
    #      - Xtensa LX6: 1.50× (different ISA, no pqm4 assembly, C reference)
    #      - Xtensa LX7: 1.30× (PIE vector extensions partially compensate)
    #   3. Latency = (pqm4_cycles × penalty) / freq_MHz
    #   Sources: pqm4 (github.com/mupq/pqm4), ARM Cortex-M TRMs,
    #            published M0+ vs M4 PQC benchmarks (1.8–1.9× penalty),
    #            ESP-IDF cryptographic benchmark reports
    #
    # cpu:         Docker --cpus limit (normalized: STM32H7-M7 = 1.0)
    # power_draw:  From official datasheets (mW active)
    # battery_cap: Normalized capacity based on real deployment
    # base_tps:    Theoretical max TPS contribution per node
    # sign_ms:     Mean ML-DSA-44 sign latency (ms), architecture-calibrated
    # verify_ms:   Mean ML-DSA-44 verify latency (ms), architecture-calibrated
    DEVICE_TYPES = {
        "ESP32":       {"cpu": 0.333, "battery_cap": 0.70, "power_draw": 160,
                        "reliability": 0.90, "base_tps": 8.0,
                        "sign_ms": 24.645, "verify_ms": 8.885},
        "ESP32-S3":    {"cpu": 0.400, "battery_cap": 0.65, "power_draw": 170,
                        "reliability": 0.88, "base_tps": 10.0,
                        "sign_ms": 21.359, "verify_ms": 7.700},
        "STM32L4-M4":  {"cpu": 0.200, "battery_cap": 0.40, "power_draw": 30,
                        "reliability": 0.95, "base_tps": 4.0,
                        "sign_ms": 49.289, "verify_ms": 17.770},
        "STM32F4-M4":  {"cpu": 0.533, "battery_cap": 0.85, "power_draw": 50,
                        "reliability": 0.87, "base_tps": 12.0,
                        "sign_ms": 23.471, "verify_ms": 8.462},
        "STM32H7-M7":  {"cpu": 1.000, "battery_cap": 1.00, "power_draw": 90,
                        "reliability": 0.85, "base_tps": 14.0,
                        "sign_ms": 4.518, "verify_ms": 1.629},
        "nRF52840":    {"cpu": 0.133, "battery_cap": 0.25, "power_draw": 23,
                        "reliability": 0.92, "base_tps": 2.5,
                        "sign_ms": 61.611, "verify_ms": 22.213},
        "RP2040":      {"cpu": 0.267, "battery_cap": 0.50, "power_draw": 45,
                        "reliability": 0.88, "base_tps": 5.5,
                        "sign_ms": 54.848, "verify_ms": 19.774},
    }

    # === Network / Consensus Constants ===
    # Calibrated from Step 3 Docker testbed (100 nodes, WiFi 802.11g via NS-3)
    # Step 3 measured: avg 4,361 TPS, bandwidth ceiling ~1,001 TPS for committee
    ML_DSA_SIG_BYTES = 2420       # ML-DSA-44 signature (FIPS 204, from Step 2)
    TX_PAYLOAD_BYTES = 256        # average transaction payload
    CONSENSUS_BW_BYTES = 1_250_000  # 10 Mbps effective (802.11g, Step 3 §3.2.1)
    BATCH_SIZE = 50               # transactions per consensus round (Step 3 batching)
    PROPAGATION_MS = 5.0          # network propagation delay (ms)
    VERIFY_PER_SIG_MS = 0.796     # Gateway ML-DSA-44 verify per sig (Step 2: 796µs avg)

    # === ML-DSA Rejection Sampling Parameters ===
    # ML-DSA-44 uses Fiat-Shamir with Aborts (FIPS 204 §5.2):
    #   Each signing attempt generates z, checks ||z||∞ < γ₁ − β.
    #   If check fails → abort and restart.
    #   Acceptance probability per attempt: p ≈ exp(−256/γ₁) ≈ 0.235
    #   Number of attempts ~ Geometric(p=0.235), mean ≈ 4.25
    #
    # Signing time is modeled as: t_sign = t_fixed + t_attempt × Geometric(p)
    #   where t_fixed accounts for context setup, hashing, NTT init (~25% of mean)
    #   and t_attempt is the variable rejection-sampling portion (~75% of mean).
    # This two-component model matches published pqm4 benchmark statistics:
    #   CV ≈ 0.65 (published range: 0.61-0.73 on Cortex-M4)
    #   P99 ≈ 3.2-3.4× mean
    # Source: FIPS 204 §5.2, pqm4 benchmark variance analysis, Becker et al. 2022
    MLDSA_ACCEPT_PROB = 0.235     # per-attempt acceptance probability
    MLDSA_MEAN_ATTEMPTS = 1.0 / 0.235  # ≈ 4.255 attempts on average
    MLDSA_FIXED_FRACTION = 0.25   # fraction of mean sign time that is fixed overhead
    MLDSA_VARIABLE_FRACTION = 0.75  # fraction subject to rejection sampling

    NODE_FEATURES = 9  # per-node observation features
    GLOBAL_FEATURES = 5

    def __init__(self, num_nodes=100, committee_size=21, max_steps=1000,
                 battery_drain_base=0.00008, battery_drain_committee=0.0015,
                 container_prefix="rc-iot-testbed_iot_node_",
                 use_real_containers=False, training_mode=True,
                 training_profile="robust"):
        super().__init__()

        self.num_nodes = num_nodes
        self.committee_size = committee_size
        self.max_steps = max_steps
        self.battery_drain_base = battery_drain_base
        self.battery_drain_committee = battery_drain_committee
        self.container_prefix = container_prefix
        self.use_real_containers = use_real_containers
        self.training_mode = training_mode
        self.training_profile = training_profile
        self.disaster_triggered = False
        self.last_disaster_kill_count = 0

        # Real deployment distribution — exact counts from experiment report
        # (bao_cao_thuc_nghiem.md Table 1.3, docker-compose.yml)
        DISTRIBUTION = {
            "ESP32": 0.25,       # 25 nodes — most common, WiFi-capable
            "ESP32-S3": 0.10,    # 10 nodes — newer, less deployed
            "STM32L4-M4": 0.15,  # 15 nodes — ultra-low-power sensor
            "STM32F4-M4": 0.15,  # 15 nodes — workhorse industrial IoT
            "STM32H7-M7": 0.10,  # 10 nodes — gateway-class
            "nRF52840": 0.10,    # 10 nodes — BLE sensor (coin cell)
            "RP2040": 0.15,      # 15 nodes — budget microcontroller
        }
        type_names = list(self.DEVICE_TYPES.keys())
        node_assignments = []
        for dev_type in type_names:
            count = max(1, int(num_nodes * DISTRIBUTION[dev_type]))
            node_assignments.extend([dev_type] * count)
        # Fill remaining with most common type
        while len(node_assignments) < num_nodes:
            node_assignments.append("ESP32")
        node_assignments = node_assignments[:num_nodes]
        # Shuffle so same types aren't contiguous
        rng = np.random.RandomState(0)  # fixed seed for reproducibility
        rng.shuffle(node_assignments)

        self.node_types = node_assignments
        self.type_ids = np.array([type_names.index(t) for t in self.node_types])

        self.node_cpu = np.array([self.DEVICE_TYPES[t]["cpu"]
                                  for t in self.node_types], dtype=np.float32)
        self.node_bat_cap_base = np.array([self.DEVICE_TYPES[t]["battery_cap"]
                                           for t in self.node_types], dtype=np.float32)
        self.node_power_draw = np.array([self.DEVICE_TYPES[t]["power_draw"]
                                         for t in self.node_types], dtype=np.float32)
        self.node_reliability = np.array([self.DEVICE_TYPES[t]["reliability"]
                                          for t in self.node_types], dtype=np.float32)
        self.node_base_tps = np.array([self.DEVICE_TYPES[t]["base_tps"]
                                       for t in self.node_types], dtype=np.float32)
        self.node_sign_ms = np.array([self.DEVICE_TYPES[t]["sign_ms"]
                                      for t in self.node_types], dtype=np.float32)
        self.node_power_norm = self.node_power_draw / 170.0  # normalize to ESP32-S3 (highest)
        self.node_tps_norm = self.node_base_tps / 14.0

        obs_dim = num_nodes * self.NODE_FEATURES + self.GLOBAL_FEATURES
        self.observation_space = spaces.Box(
            low=-1.0, high=2.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-5.0, high=5.0, shape=(num_nodes,), dtype=np.float32
        )
        self.history = []
        self._init_state()

    def _init_state(self):
        # Battery degradation: max capacity decreases over lifetime
        self.node_bat_cap = self.node_bat_cap_base.copy()
        self.batteries = self.node_bat_cap.copy()
        self.alive = np.ones(self.num_nodes, dtype=np.float32)
        self.trust_scores = np.ones(self.num_nodes, dtype=np.float32)
        self.in_committee = np.zeros(self.num_nodes, dtype=np.float32)
        self.step_count = 0
        self.disaster_triggered = False
        self.last_disaster_kill_count = 0
        self.total_committee_steps = np.zeros(self.num_nodes, dtype=np.float32)
        # Sliding window: track committee membership over last 100 steps
        self._service_window_size = 100
        self._service_window = np.zeros((self._service_window_size, self.num_nodes),
                                        dtype=np.float32)
        self._service_window_idx = 0

        # === Byzantine state ===
        # is_byzantine: 0=honest, 1=active Byzantine
        # stealth_timer: countdown before dormant node activates
        #   -1 = not compromised, >0 = dormant (appears honest), 0 = active
        self.is_byzantine = np.zeros(self.num_nodes, dtype=np.float32)
        self.stealth_timer = np.full(self.num_nodes, -1, dtype=np.float32)

        # Network jitter factor per node (changes each step)
        self.jitter = np.ones(self.num_nodes, dtype=np.float32)

        # === NODE CHURN state (realistic state machine) ===
        # Each node has independent churn state:
        #   0 = ACTIVE, 1 = SLEEPING, 2 = SIGNAL_LOST, 3 = FAILED (rebooting)
        self.churn_state = np.zeros(self.num_nodes, dtype=np.int32)
        self.churn_timer = np.zeros(self.num_nodes, dtype=np.int32)
        # Per-node sleep cycle: each node has its own sleep schedule
        # Sleep every 60-180 steps (1-3 hours real-time) for 10-30 steps
        self.sleep_interval = np.random.randint(60, 181, size=self.num_nodes)
        self.next_sleep_step = np.random.randint(30, 120, size=self.num_nodes)
        # Legacy offline tracking (for backward compat)
        self.is_offline = np.zeros(self.num_nodes, dtype=np.float32)
        self.offline_battery_save = np.zeros(self.num_nodes, dtype=np.float32)
        # Physical disaster kills are permanent. Ordinary churn states
        # (sleep/signal loss/reboot) may recover, but disaster-killed nodes must not.
        self.permanent_dead = np.zeros(self.num_nodes, dtype=np.float32)

        if self.training_mode and self.training_profile == "robust":
            # Robust training mirrors Experiment 3: active Byzantine faults are
            # present from the start, and a 30% physical node-kill occurs near
            # the evaluation disaster point. The concrete nodes are randomized
            # per reset/worker seed, so test seeds remain held out.
            alive_mask = np.where(self.alive > 0)[0]
            byz_count = int(len(alive_mask) * 0.20)
            if byz_count > 0:
                byz_ids = np.random.choice(alive_mask, byz_count, replace=False)
                self.make_byzantine(byz_ids, stealth_ratio=0.0)

            # Mild battery variation avoids a brittle full-battery policy while
            # staying close to the evaluation initial state.
            for i in range(self.num_nodes):
                if self.alive[i] > 0:
                    self.batteries[i] *= np.random.uniform(0.85, 1.0)

            if self.max_steps >= 1200:
                self._scheduled_disaster_step = np.random.randint(900, 1101)
            else:
                self._scheduled_disaster_step = max(1, int(self.max_steps * 0.5))

        elif self.training_mode:
            # Random initial failures (2-8) — realistic IoT: some nodes offline at boot
            fail_count = np.random.randint(2, 9)
            if fail_count > 0:
                fail_ids = np.random.choice(self.num_nodes, fail_count, replace=False)
                self.alive[fail_ids] = 0.0
                self.batteries[fail_ids] = 0.0

            # Random initial Byzantine (10-18%) — realistic APT threat model
            byz_count = np.random.randint(8, 18)
            alive_mask = np.where(self.alive > 0)[0]
            if len(alive_mask) > byz_count:
                byz_ids = np.random.choice(alive_mask, byz_count, replace=False)
                for bid in byz_ids:
                    if np.random.random() < 0.50:
                        # Stealth APT: dormant, maintains VERY HIGH trust
                        # Key change: stealth nodes nearly indistinguishable
                        self.stealth_timer[bid] = np.random.randint(100, 500)
                        self.trust_scores[bid] = np.random.uniform(0.80, 0.97)
                    else:
                        # Active immediately but some have high trust initially
                        self.is_byzantine[bid] = 1.0
                        self.trust_scores[bid] = np.random.uniform(0.35, 0.70)

            # Random initial battery variation — realistic: not all start at 100%
            for i in range(self.num_nodes):
                if self.alive[i] > 0:
                    self.batteries[i] *= np.random.uniform(0.55, 1.0)

            # 60% chance of mid-episode disaster
            if np.random.random() < 0.6:
                self._scheduled_disaster_step = np.random.randint(250, 750)
            else:
                self._scheduled_disaster_step = -1
        else:
            self._scheduled_disaster_step = -1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            # Most of this environment predates Gymnasium's Generator API and
            # still uses np.random directly; seed it here for reproducible
            # reset(seed=...) behavior.
            np.random.seed(seed)
        self.history = []  # clear history on reset to prevent memory leak
        self._init_state()
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1

        # Scheduled training disaster
        if self.training_mode and self.step_count == self._scheduled_disaster_step:
            self._trigger_training_disaster()

        # === STEALTH BYZANTINE ACTIVATION (vectorized) ===
        active_stealth = self.stealth_timer > 0
        self.stealth_timer[active_stealth] -= 1
        activating = self.stealth_timer == 0
        if np.any(activating):
            self.is_byzantine[activating] = 1.0
            self.stealth_timer[activating] = -2
            # Keep current trust (don't reset) — Byzantine behavior
            # will cause trust to decay naturally via the trust evolution code

        # === NETWORK JITTER (changes each step — more volatile) ===
        new_jitter = np.random.uniform(0.6, 1.0, size=self.num_nodes).astype(np.float32)
        # 10% chance of bad jitter per node (WiFi interference, congestion)
        bad_jitter = np.random.random(self.num_nodes) < 0.10
        new_jitter[bad_jitter] = np.random.uniform(0.15, 0.45, size=bad_jitter.sum())
        # Preserve locked jitter (external storm injection)
        locked = getattr(self, 'jitter_locked', None)
        if locked is not None:
            self.jitter = np.where(locked, self.jitter, new_jitter)
        else:
            self.jitter = new_jitter

        # === NODE CHURN — realistic per-node state machine ===
        self._do_realistic_churn()

        # === Committee selection from raw scores ===
        scores = np.array(action, dtype=np.float32)
        scores[self.alive == 0] = -1e9

        alive_indices = np.where(self.alive > 0)[0]
        k = min(self.committee_size, len(alive_indices))

        new_committee = np.zeros(self.num_nodes, dtype=np.float32)
        if k > 0:
            top_k_idx = np.argsort(scores)[-k:]
            new_committee[top_k_idx] = 1.0

        prev_committee = self.in_committee.copy()
        self.in_committee = new_committee

        # Track total time each node has served
        self.total_committee_steps += self.in_committee
        # Update sliding window (ring buffer)
        # Use (step_count - 1) so step 1 writes to slot 0, reads [:1] gets current data
        widx = (self.step_count - 1) % self._service_window_size
        self._service_window[widx] = self.in_committee

        # === BATTERY DRAIN + AGING (Phase 1: base drain, vectorized) ===
        alive_mask = self.alive > 0
        committee_mask = (self.in_committee > 0) & alive_mask
        idle_mask = alive_mask & ~committee_mask

        # Committee nodes: capacity aging
        self.node_bat_cap[committee_mask] *= 0.9999

        # Drain calculation
        drain = np.full(self.num_nodes, self.battery_drain_base, dtype=np.float32)
        drain[committee_mask] = self.battery_drain_committee * self.node_power_norm[committee_mask]

        # Voltage sag penalty for low-battery committee nodes
        bat_ratios = self.batteries / np.maximum(self.node_bat_cap, 0.01)
        low_bat_committee = committee_mask & (bat_ratios < 0.3)
        drain[low_bat_committee] *= 1.5

        # Apply drain (only alive nodes)
        self.batteries[alive_mask] = np.maximum(0.0, self.batteries[alive_mask] - drain[alive_mask])

        # Kill drained nodes
        dead_now = alive_mask & (self.batteries <= 0)
        self.alive[dead_now] = 0.0
        self.in_committee[dead_now] = 0.0

        # === TRUST EVOLUTION (vectorized) ===
        alive_mask = self.alive > 0  # refresh after deaths
        n = self.num_nodes

        # Active Byzantine: slow decay
        byz_mask = alive_mask & (self.is_byzantine > 0)
        if np.any(byz_mask):
            n_byz = int(byz_mask.sum())
            decay = np.random.uniform(0.003, 0.012, size=n_byz)
            self.trust_scores[byz_mask] = np.maximum(0.10, self.trust_scores[byz_mask] - decay)

        # Dormant stealth: trust increases
        stealth_mask = alive_mask & (self.stealth_timer > 0)
        if np.any(stealth_mask):
            n_st = int(stealth_mask.sum())
            self.trust_scores[stealth_mask] = np.minimum(1.0,
                self.trust_scores[stealth_mask] + np.random.uniform(0.001, 0.008, size=n_st))

        # Honest: random walk
        honest_mask = alive_mask & (self.is_byzantine == 0) & (self.stealth_timer < 0)
        if np.any(honest_mask):
            n_h = int(honest_mask.sum())
            self.trust_scores[honest_mask] = np.clip(
                self.trust_scores[honest_mask] + np.random.uniform(-0.008, 0.012, size=n_h),
                0.0, 1.0)

        # === BYZANTINE CONTAGION — higher risk in larger committees ===
        # With committee=21, more surface area for contagion
        if k > 0:
            for i in range(self.num_nodes):
                if (self.in_committee[i] > 0 and self.is_byzantine[i] > 0
                        and self.alive[i] > 0):
                    honest_committee = [j for j in range(self.num_nodes)
                                       if (self.in_committee[j] > 0
                                           and self.is_byzantine[j] == 0
                                           and self.stealth_timer[j] < 0
                                           and self.alive[j] > 0)]
                    # 1% contagion per byz node in committee (was 0.5%)
                    if honest_committee and np.random.random() < 0.01:
                        victim = np.random.choice(honest_committee)
                        self.stealth_timer[victim] = np.random.randint(80, 300)

        # === CORRELATED FAILURES ===
        # When a node dies, same-type nodes have 5% chance of also failing
        for i in range(self.num_nodes):
            if prev_committee[i] > 0 and self.alive[i] == 0 and self.churn_state[i] == 0:
                same_type = np.where(
                    (self.type_ids == self.type_ids[i]) &
                    (self.alive > 0) &
                    (np.arange(self.num_nodes) != i)
                )[0]
                for j in same_type:
                    if np.random.random() < 0.02:
                        self.alive[j] = 0.0
                        self.batteries[j] = 0.0

        # === Random training failures ===
        if self.training_mode and np.random.random() < 0.003:
            alive_idx = np.where(self.alive > 0)[0]
            if len(alive_idx) > self.committee_size + 5:
                fail_id = np.random.choice(alive_idx)
                self.alive[fail_id] = 0.0
                self.batteries[fail_id] = 0.0

        # === TPS CALCULATION WITH ML-DSA SIGNING MODEL (vectorized) ===
        committee_mask = (self.in_committee > 0) & (self.alive > 0)
        committee_indices = np.where(committee_mask)[0]
        committee_active = len(committee_indices)

        byz_in_comm = (self.is_byzantine[committee_indices] > 0) if committee_active > 0 else np.array([])
        byzantine_in_committee = int(byz_in_comm.sum()) if len(byz_in_comm) > 0 else 0

        honest_indices = committee_indices[~byz_in_comm] if committee_active > 0 else np.array([], dtype=int)
        threshold_signers = (self.committee_size - 1) // 3 + 1

        if len(honest_indices) > 0:
            bat_ratios = self.batteries[honest_indices] / np.maximum(self.node_bat_cap[honest_indices], 0.01)

            # Battery cliff — smooth sigmoid transition (no discontinuity)
            # Maps bat_ratio through a smooth curve: high bat=~1.0, low bat→0
            # Sigmoid centered at 0.25 with steepness 15
            bat_factors = 0.1 + 0.9 / (1.0 + np.exp(-15.0 * (bat_ratios - 0.25)))

            # Thermal fatigue (vectorized) -- coefficient 1.5: continuous service
            # causes significant thermal throttling on embedded devices
            # Uses SLIDING WINDOW (last 100 steps) so fatigue doesn't dilute over time
            window_fill = min(self.step_count, self._service_window_size)
            if window_fill > 0:
                recent_service = self._service_window[:window_fill, honest_indices].sum(axis=0) / window_fill
            else:
                recent_service = np.zeros(len(honest_indices))
            thermal_sens = self.node_power_norm[honest_indices]
            fatigues = np.maximum(0.1, 1.0 - 1.5 * recent_service * thermal_sens)

            # Degradation and base sign times
            degradations = np.maximum(0.08, bat_factors * fatigues)
            base_sign_ms = self.node_sign_ms[honest_indices] / degradations
            base_sign_ms /= np.maximum(self.jitter[honest_indices], 0.3)
            base_sign_ms /= np.maximum(self.node_reliability[honest_indices], 0.5)

            # ML-DSA rejection sampling: stochastic sign latency
            # Model: t_sign = t_fixed + t_attempt × Geometric(p)
            # t_fixed = 25% of mean (context setup, hashing, NTT init)
            # t_attempt = 75% of mean / mean_attempts (per rejection round)
            # This preserves correct mean while producing:
            #   CV ≈ 0.65, P99/mean ≈ 3.2-3.4× (matches pqm4 measurements)
            n_honest = len(honest_indices)
            fixed_ms = base_sign_ms * self.MLDSA_FIXED_FRACTION
            variable_mean = base_sign_ms * self.MLDSA_VARIABLE_FRACTION
            per_attempt_ms = variable_mean / self.MLDSA_MEAN_ATTEMPTS
            rejection_attempts = self.np_random.geometric(
                p=self.MLDSA_ACCEPT_PROB, size=n_honest
            )
            actual_sign_ms = fixed_ms + per_attempt_ms * rejection_attempts

            honest_sign_times = actual_sign_ms.tolist()
        else:
            honest_sign_times = []

        # BFT threshold: f < ⌊n/3⌋ for safety
        bft_threshold = max(int(np.ceil(committee_active / 3.0)), 1)
        honest_count = len(honest_sign_times)

        if byzantine_in_committee >= bft_threshold:
            # Consensus FAILURE: too many Byzantine → no agreement
            tps = 0.0
        elif honest_count < threshold_signers:
            tps = 0.0
        else:
            # === CPU bottleneck ===
            # TALUS/threshold ML-DSA only needs the first t valid shares. The
            # coordinator does not wait for all committee members.
            quorum_sign_ms = sorted(honest_sign_times)[threshold_signers - 1]
            # The gateway verifies one final standard ML-DSA signature.
            verify_ms = self.VERIFY_PER_SIG_MS
            round_ms = (self.PROPAGATION_MS + quorum_sign_ms
                        + self.PROPAGATION_MS + verify_ms)
            cpu_tps = (self.BATCH_SIZE / round_ms) * 1000.0

            # === Bandwidth bottleneck ===
            # Each round transmits: committee_active signatures + batch data
            bytes_per_round = (committee_active * self.ML_DSA_SIG_BYTES
                               + self.BATCH_SIZE * self.TX_PAYLOAD_BYTES)
            max_rounds_per_sec = self.CONSENSUS_BW_BYTES / max(bytes_per_round, 1)
            bw_tps = max_rounds_per_sec * self.BATCH_SIZE

            # TPS = minimum of CPU and bandwidth bottlenecks
            tps = min(cpu_tps, bw_tps)

            # Byzantine degradation (within tolerance but causes retransmits)
            if byzantine_in_committee > 0:
                safety_ratio = byzantine_in_committee / bft_threshold
                consensus_efficiency = 1.0 - safety_ratio ** 3
                tps *= max(0.0, consensus_efficiency)

            # Small random noise
            tps = max(0.0, tps + np.random.normal(0, 0.5))

        tps = max(0.0, tps)

        # === COMMITTEE TRANSITION OVERHEAD ===
        # BFT key share redistribution: each new member needs threshold
        # key share via DKG round (~50ms per member), blocking consensus
        # Only count VOLUNTARY swaps (agent chose to change), not churn-forced
        voluntary_out = int(np.sum((prev_committee > 0) & (new_committee == 0) & (self.alive > 0)))
        voluntary_in = int(np.sum((prev_committee == 0) & (new_committee > 0)))
        swap_count = voluntary_out + voluntary_in
        # Allow 5 natural rotations per step, penalize beyond
        excess_swaps = max(0, swap_count - 5)
        # Each excess swap costs ~4% throughput (DKG + key redistribution)
        transition_efficiency = max(0.5, 1.0 - 0.04 * excess_swaps)
        tps = tps * transition_efficiency

        # === BATTERY DRAIN (Phase 2: workload-proportional) ===
        # Higher TPS = more signatures = more energy consumed
        if committee_active > 0 and tps > 0:
            work_per_node = tps / max(committee_active, 1)
            signing_drain = (work_per_node / 1000.0) * self.battery_drain_committee
            for i in range(self.num_nodes):
                if self.in_committee[i] > 0 and self.alive[i] > 0:
                    extra = signing_drain * self.node_power_norm[i]
                    self.batteries[i] = max(0.0, self.batteries[i] - extra)
                    if self.batteries[i] <= 0:
                        self.alive[i] = 0.0
                        self.in_committee[i] = 0.0

        # === REWARD ===
        alive_count = int(self.alive.sum())

        # R1: TPS (DOMINANT) — normalized to bandwidth ceiling
        # Must be the strongest signal so PPO learns to pick fast nodes FIRST
        bw_cap = (self.CONSENSUS_BW_BYTES
                  / max(k * self.ML_DSA_SIG_BYTES
                        + self.BATCH_SIZE * self.TX_PAYLOAD_BYTES, 1)
                  ) * self.BATCH_SIZE if k > 0 else 1.0
        r_tps = min(15.0, (tps / max(bw_cap, 1.0)) * 15.0) if k > 0 else 0.0

        # R2: Byzantine penalty (reduced from -10 to -3 so it doesn't
        # drown out TPS and rotation signals — was 87% of total reward!)
        r_byzantine = -3.0 * byzantine_in_committee

        # R3: Alive ratio (mild)
        r_alive = (alive_count / self.num_nodes) * 2.0

        # R4: Node death penalty — only for ACTUAL deaths (battery drained),
        # NOT for churn-sleeping nodes (PPO shouldn't be punished for env churn)
        deaths = sum(1 for i in range(self.num_nodes)
                     if prev_committee[i] > 0 and self.alive[i] == 0
                     and self.churn_state[i] == 0)  # exclude sleeping/signal_lost/failed
        r_death = -5.0 * deaths

        # R5: Rotation incentive -- penalize stale committees
        # Uses sliding window service ratio (last 100 steps) for consistent signal
        committee_indices_reward = np.where(self.in_committee > 0)[0]
        if len(committee_indices_reward) > 0:
            window_fill = min(self.step_count, self._service_window_size)
            if window_fill > 0:
                recent_sr = (self._service_window[:window_fill, committee_indices_reward]
                             .sum(axis=0) / window_fill)
            else:
                recent_sr = np.zeros(len(committee_indices_reward))
            r_fatigue = -1.5 * np.mean(recent_sr)
        else:
            r_fatigue = 0.0

        # R6: Fresh node bonus — reward introducing new nodes to committee
        if self.step_count > 1:
            new_in_committee = np.sum(
                (new_committee > 0) & (prev_committee == 0) & (self.alive > 0))
            r_diversity = 0.15 * (new_in_committee / max(k, 1))
        else:
            r_diversity = 0.0

        r_recovery = 0.0
        if self.training_mode and self.training_profile == "robust":
            # Train for the evaluation objective, not only pre-disaster peak
            # throughput. After disaster, throughput and liveness matter more;
            # throughout training, discourage overusing the same committee.
            if self.disaster_triggered:
                r_tps *= 1.35
                r_alive *= 1.5
                r_death *= 1.5
                r_recovery = min(8.0, (tps / max(bw_cap, 1.0)) * 8.0)
            r_byzantine = -4.0 * byzantine_in_committee
            if committee_active > 0:
                r_fatigue *= 2.0
                r_diversity *= 2.0

        reward = (r_tps + r_byzantine + r_alive + r_death
                  + r_fatigue + r_diversity + r_recovery)

        step_info = {
            "step": self.step_count,
            "tps": float(tps),
            "alive": alive_count,
            "committee_active": int(self.in_committee.sum()),
            "threshold_signers": int(threshold_signers),
            "byzantine_in_committee": byzantine_in_committee,
            "avg_battery": float(np.mean(self.batteries[self.alive > 0])
                                  if alive_count > 0 else 0.0),
            "reward": float(reward),
            "disaster_triggered": bool(self.disaster_triggered),
            "scheduled_disaster_step": int(self._scheduled_disaster_step),
            "last_disaster_kill_count": int(self.last_disaster_kill_count),
            "swap_count": swap_count,
            "transition_eff": float(transition_efficiency),
        }
        self.history.append(step_info)

        terminated = alive_count < self.committee_size
        truncated = self.step_count >= self.max_steps

        return self._get_obs(), float(reward), terminated, truncated, step_info

    # NOTE: second definition below (line ~738) overrides this one.
    # This first version is kept for reference but Python uses the LAST definition.

    def _get_obs(self):
        """Per-node: [bat_ratio, alive, noisy_trust, cpu, tps_norm,
                      power_norm, in_committee, jitter, recent_service_ratio]"""
        obs = np.zeros(self.num_nodes * self.NODE_FEATURES + self.GLOBAL_FEATURES,
                       dtype=np.float32)
        # Use sliding window for service ratio (not diluted total)
        window_fill = min(self.step_count, self._service_window_size)
        if window_fill > 0:
            recent_service = self._service_window[:window_fill].sum(axis=0) / window_fill
        else:
            recent_service = np.zeros(self.num_nodes)

        for i in range(self.num_nodes):
            base = i * self.NODE_FEATURES
            obs[base + 0] = self.batteries[i] / max(self.node_bat_cap[i], 0.01)
            obs[base + 1] = self.alive[i]
            if self.alive[i] > 0:
                # Noisy trust (σ=0.15) — IoT sensor measurement uncertainty
                # Same noise level as all baselines (SA/GBA/OCD) for fairness
                noise = np.random.normal(0, 0.15)
                obs[base + 2] = np.clip(self.trust_scores[i] + noise, 0.0, 1.0)
            else:
                obs[base + 2] = 0.0
            obs[base + 3] = self.node_cpu[i]
            obs[base + 4] = self.node_tps_norm[i]
            obs[base + 5] = self.node_power_norm[i]
            obs[base + 6] = self.in_committee[i]
            obs[base + 7] = self.jitter[i]
            obs[base + 8] = recent_service[i]

        g = self.num_nodes * self.NODE_FEATURES
        alive_bat = self.batteries[self.alive > 0]
        obs[g + 0] = np.mean(alive_bat) if len(alive_bat) > 0 else 0.0
        obs[g + 1] = self.alive.sum() / self.num_nodes
        # Clip to [0,1] so evaluation (2000 steps) doesn't go out-of-distribution
        obs[g + 2] = min(1.0, self.step_count / self.max_steps)
        # Trust spread: proxy for Byzantine presence (no info leak)
        alive_trust = [self.trust_scores[i] for i in range(self.num_nodes)
                       if self.alive[i] > 0]
        obs[g + 3] = np.std(alive_trust) if len(alive_trust) > 1 else 0.0
        # Total stealth count is HIDDEN from observation (agent must infer)
        obs[g + 4] = np.mean(self.jitter[self.alive > 0]) if self.alive.sum() > 0 else 0.5

        return obs

    def kill_nodes(self, node_ids):
        for nid in node_ids:
            if 0 <= nid < self.num_nodes:
                self.permanent_dead[nid] = 1.0
                self.alive[nid] = 0.0
                self.batteries[nid] = 0.0
                self.in_committee[nid] = 0.0
                self.churn_state[nid] = 0
                self.churn_timer[nid] = 0
                self.is_offline[nid] = 0.0
                self.offline_battery_save[nid] = 0.0
                self.is_byzantine[nid] = 0.0
                self.stealth_timer[nid] = -1.0
                if self.use_real_containers:
                    try:
                        subprocess.run(["docker", "pause",
                                       f"{self.container_prefix}{nid+1}_1"],
                                     capture_output=True, timeout=5)
                    except Exception:
                        pass

    def make_byzantine(self, node_ids, stealth_ratio=0.3):
        """Make nodes Byzantine. stealth_ratio = fraction that are stealth."""
        for nid in node_ids:
            if (0 <= nid < self.num_nodes
                    and self.alive[nid] > 0
                    and self.permanent_dead[nid] == 0):
                if np.random.random() < stealth_ratio:
                    # Stealth: VERY high trust, long dormancy
                    self.stealth_timer[nid] = np.random.randint(80, 400)
                    self.trust_scores[nid] = np.random.uniform(0.80, 0.97)
                else:
                    self.is_byzantine[nid] = 1.0
                    self.trust_scores[nid] = np.random.uniform(0.35, 0.65)

    def get_history(self):
        return self.history


    def _do_realistic_churn(self):
        """Realistic per-node IoT churn state machine.

        Each node independently transitions between states:
          ACTIVE(0) → SLEEPING(1): Energy-saving sleep cycle
          ACTIVE(0) → SIGNAL_LOST(2): WiFi interference / mobility
          ACTIVE(0) → FAILED(3): Hardware failure (rare, long recovery)

        1 step = 1 minute. Timescales:
          Sleep: every 1-3 hours, lasts 10-30 minutes
          Signal loss: 2-5% chance per step, lasts 5-15 minutes
          Hardware failure: 0.1% chance per step, recovery 20-50 minutes

        CRITICAL for DRL advantage:
        - PPO sees reliability, jitter, service_ratio → can predict churn
        - SA/GBA only see bat × speed × trust → blind to upcoming sleep
        """
        alive_count = int(self.alive.sum())
        min_alive = self.committee_size + 5  # safety margin

        for i in range(self.num_nodes):
            if self.permanent_dead[i] > 0:
                self.alive[i] = 0.0
                self.churn_state[i] = 0
                self.churn_timer[i] = 0
                self.is_offline[i] = 0.0
                self.offline_battery_save[i] = 0.0
                continue

            # --- Handle nodes currently in churn states ---
            if self.churn_state[i] > 0:
                self.churn_timer[i] -= 1
                if self.churn_timer[i] <= 0:
                    # Recovery: node comes back online
                    self.churn_state[i] = 0
                    self.is_offline[i] = 0.0
                    self.alive[i] = 1.0
                    # Partial battery recovery (charged while offline)
                    self.batteries[i] = self.node_bat_cap[i] * np.random.uniform(0.6, 0.95)
                    # Trust reset — network hasn't seen this node recently
                    if self.is_byzantine[i] == 0 and self.stealth_timer[i] < 0:
                        self.trust_scores[i] = np.random.uniform(0.4, 0.65)
                    # Reset service ratio (fresh start)
                    self.total_committee_steps[i] = 0.0
                continue

            # --- Active nodes: check for churn events ---
            if self.alive[i] == 0:
                continue
            if alive_count <= min_alive:
                continue  # skip new churn events but still process recovery

            # 1) SLEEP CYCLE: periodic, predictable
            if self.step_count >= self.next_sleep_step[i]:
                self.churn_state[i] = 1  # SLEEPING
                sleep_duration = np.random.randint(10, 31)  # 10-30 minutes
                self.churn_timer[i] = sleep_duration
                self.next_sleep_step[i] = (self.step_count + sleep_duration
                                            + self.sleep_interval[i])
                self.is_offline[i] = 1.0
                self.offline_battery_save[i] = self.batteries[i]
                self.alive[i] = 0.0
                self.in_committee[i] = 0.0
                alive_count -= 1
                continue

            # 2) SIGNAL LOSS: random, temporary (WiFi interference)
            # Lower reliability → higher chance of signal loss
            signal_loss_prob = 0.03 * (1.0 - self.node_reliability[i])
            if np.random.random() < signal_loss_prob:
                self.churn_state[i] = 2  # SIGNAL_LOST
                self.churn_timer[i] = np.random.randint(5, 16)  # 5-15 minutes
                self.is_offline[i] = 1.0
                self.offline_battery_save[i] = self.batteries[i]
                self.alive[i] = 0.0
                self.in_committee[i] = 0.0
                alive_count -= 1
                continue

            # 3) HARDWARE FAILURE: rare, long recovery
            if np.random.random() < 0.001:  # 0.1% per step = ~6% per hour
                self.churn_state[i] = 3  # FAILED
                self.churn_timer[i] = np.random.randint(20, 51)  # 20-50 min reboot
                self.is_offline[i] = 1.0
                self.offline_battery_save[i] = self.batteries[i]
                self.alive[i] = 0.0
                self.in_committee[i] = 0.0
                alive_count -= 1
                continue

    def _do_node_churn(self):
        """Legacy wrapper for backward compatibility."""
        self._do_realistic_churn()

    def _trigger_training_disaster(self):
        if self.training_profile == "robust":
            kill_candidates = np.where(self.permanent_dead == 0)[0]
        else:
            kill_candidates = np.where(self.alive > 0)[0]
        if len(kill_candidates) < 20:
            return

        self.disaster_triggered = True

        # Robust profile mirrors Experiment 3 exactly in severity; the standard
        # profile keeps the older randomized stress range.
        if self.training_profile == "robust":
            kill_count = min(int(self.num_nodes * 0.30), len(kill_candidates))
        else:
            kill_ratio = np.random.uniform(0.20, 0.30)
            kill_count = int(len(kill_candidates) * kill_ratio)
        kill_ids = np.random.choice(kill_candidates, kill_count, replace=False)
        self.kill_nodes(kill_ids)
        self.last_disaster_kill_count = int(kill_count)

        if self.training_profile == "robust":
            # Keep the active Byzantine population at 20% of survivors after
            # the physical kill, matching Experiment 3. No stealth here: this
            # benchmark evaluates concurrent active Byzantine faults.
            surviving = np.where(self.alive > 0)[0]
            target_byz = int(len(surviving) * 0.20)
            active_byz = {
                int(i) for i in surviving
                if self.is_byzantine[i] > 0 and self.permanent_dead[i] == 0
            }
            need = max(0, target_byz - len(active_byz))
            if need > 0:
                honest_surviving = [
                    int(i) for i in surviving
                    if self.is_byzantine[i] == 0 and self.stealth_timer[i] < 0
                ]
                if honest_surviving:
                    byz_ids = np.random.choice(
                        honest_surviving,
                        min(need, len(honest_surviving)),
                        replace=False,
                    )
                    self.make_byzantine(byz_ids, stealth_ratio=0.0)
            return

        # Inject Byzantine among survivors (15-20%)
        surviving = np.where(self.alive > 0)[0]
        byz_ratio = np.random.uniform(0.18, 0.28)
        byz_count = int(len(surviving) * byz_ratio)
        # Only infect currently honest nodes
        honest_surviving = [i for i in surviving if self.is_byzantine[i] == 0]
        if len(honest_surviving) > byz_count:
            byz_ids = np.random.choice(honest_surviving, byz_count, replace=False)
            for bid in byz_ids:
                if np.random.random() < 0.45:
                    # Stealth — maintains VERY high trust (advanced APT)
                    self.stealth_timer[bid] = np.random.randint(80, 350)
                    self.trust_scores[bid] = np.random.uniform(0.80, 0.97)
                else:
                    # Active Byzantine
                    self.is_byzantine[bid] = 1.0
                    self.trust_scores[bid] = np.random.uniform(0.35, 0.65)
