#!/usr/bin/env python3
"""
ppo_agent.py — PPO v11 (Per-Node Scoring Network)

CRITICAL FIX: Instead of obs(9N+5)→action(N) flat network,
use a SHARED per-node scorer: node_features(9)+global_features(5)→score(1)

This is the correct inductive bias: "what makes a good committee node?"
is the SAME question for every node. The network learns this once.

Effective params: ~2K instead of ~100K → trains 50x faster.
"""

import os
import sys
import json
import argparse
import numpy as np
from collections import deque

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iot_env import IoTNetworkEnv
import multiprocessing as mp


def _worker(pipe, env_kwargs, worker_id=0, base_seed=None):
    """Worker process: owns one env, receives actions, sends back results."""
    if base_seed is not None:
        np.random.seed(int(base_seed) + worker_id * 1009)
    env = IoTNetworkEnv(**env_kwargs)
    if base_seed is not None:
        obs, _ = env.reset(seed=int(base_seed) + worker_id * 1009)
    else:
        obs, _ = env.reset()
    pipe.send(obs)  # send initial obs

    while True:
        cmd, data = pipe.recv()
        if cmd == 'step':
            obs, rew, done, trunc, info = env.step(data)
            if done or trunc:
                obs, _ = env.reset()
            pipe.send((obs, rew, done or trunc, info))
        elif cmd == 'reset':
            obs, _ = env.reset()
            pipe.send(obs)
        elif cmd == 'close':
            pipe.close()
            break


class SubprocVecEnv:
    """Run N envs in separate processes for TRUE parallelism."""

    def __init__(self, env_kwargs, n_envs=8, base_seed=None):
        self.n_envs = n_envs
        self.pipes = []
        self.procs = []

        for worker_id in range(n_envs):
            parent_pipe, child_pipe = mp.Pipe()
            proc = mp.Process(
                target=_worker,
                args=(child_pipe, env_kwargs, worker_id, base_seed),
                daemon=True,
            )
            proc.start()
            child_pipe.close()
            self.pipes.append(parent_pipe)
            self.procs.append(proc)

        # Receive initial observations
        self.obs = np.array([pipe.recv() for pipe in self.pipes])

    def step(self, actions):
        """Send actions to all workers, receive results in parallel."""
        for pipe, action in zip(self.pipes, actions):
            pipe.send(('step', action))
        results = [pipe.recv() for pipe in self.pipes]
        obs, rews, dones, infos = zip(*results)
        self.obs = np.array(obs)
        return self.obs, np.array(rews), np.array(dones), infos

    def reset_all(self):
        for pipe in self.pipes:
            pipe.send(('reset', None))
        self.obs = np.array([pipe.recv() for pipe in self.pipes])
        return self.obs

    def close(self):
        for pipe in self.pipes:
            try:
                pipe.send(('close', None))
                pipe.close()
            except Exception:
                pass
        for proc in self.procs:
            proc.join(timeout=3)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[WARN] PyTorch not installed.")


class PerNodeScoringNetwork(nn.Module):
    """
    Per-node scoring: each node gets scored by the SAME small network.
    Input: node features (9) + global features (5) = 14 dims
    Output: 1 score per node
    
    This is permutation-equivariant — correct inductive bias for
    committee selection.
    """

    def __init__(self, node_features=9, global_features=5, num_nodes=100,
                 hidden=128, committee_size=21, actor_mode="gaussian_masked",
                 gumbel_temperature=0.7, critic_mode="flat"):
        super().__init__()
        self.node_features = node_features
        self.global_features = global_features
        self.num_nodes = num_nodes
        self.committee_size = committee_size
        self.actor_mode = actor_mode
        self.gumbel_temperature = gumbel_temperature
        self.critic_mode = critic_mode
        input_dim = node_features + global_features

        # Shared scorer: processes each node's features + global context
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

        # Gaussian score exploration. Flat mode keeps the historical per-node
        # parameter for checkpoint compatibility; pooled mode uses a scalar so
        # the trainable parameter set is independent of N.
        if critic_mode == "pooled":
            self.actor_log_std = nn.Parameter(torch.tensor(-0.5))
        else:
            self.actor_log_std = nn.Parameter(torch.zeros(num_nodes) - 0.5)

        if critic_mode == "flat":
            # Historical critic: takes the full flattened observation. This is
            # kept for backward-compatible evaluation of existing checkpoints.
            obs_dim = num_nodes * node_features + global_features
            self.critic = nn.Sequential(
                nn.Linear(obs_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
        elif critic_mode == "pooled":
            # Permutation-invariant critic: encode each node independently,
            # then pool across the valid alive-node set. The parameter shapes
            # no longer depend on num_nodes, which enables clean N=50/200/500
            # retraining and checkpoint transfer experiments.
            self.critic_encoder = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            )
            self.critic_head = nn.Sequential(
                nn.Linear(hidden * 2 + global_features, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
        else:
            raise ValueError(f"Unsupported critic_mode: {critic_mode}")

    def _split_obs(self, obs):
        """Split flat obs into per-node features and global features."""
        batch_size = obs.shape[0]
        nf = self.node_features
        n = self.num_nodes

        node_obs = obs[:, :n * nf].reshape(batch_size, n, nf)
        global_obs = obs[:, n * nf:]  # (batch, global_features)

        return node_obs, global_obs

    def _valid_action_mask(self, obs):
        """Mask nodes that cannot be selected before policy sampling.

        Feature layout comes from IoTNetworkEnv._get_obs():
        per-node [battery_ratio, alive, trust, cpu, cap, power, inAC, jitter, service].
        """
        node_obs, _ = self._split_obs(obs)
        battery_ok = node_obs[:, :, 0] > 0.01
        alive_ok = node_obs[:, :, 1] > 0.5
        return battery_ok & alive_ok

    def _apply_action_mask(self, scores, obs, floor=-8.0):
        valid_mask = self._valid_action_mask(obs)
        masked_scores = scores.masked_fill(~valid_mask, floor)
        return masked_scores, valid_mask

    def _pooled_value(self, node_input, global_obs, valid_mask):
        emb = self.critic_encoder(node_input)
        mask_f = valid_mask.unsqueeze(-1).float()
        count = mask_f.sum(dim=1).clamp_min(1.0)
        mean_pool = (emb * mask_f).sum(dim=1) / count

        max_input = emb.masked_fill(~valid_mask.unsqueeze(-1), -1e9)
        max_pool = max_input.max(dim=1).values
        no_valid = valid_mask.sum(dim=1) == 0
        if no_valid.any():
            max_pool = max_pool.masked_fill(no_valid.unsqueeze(-1), 0.0)

        return self.critic_head(torch.cat([mean_pool, max_pool, global_obs], dim=-1))

    def forward(self, obs, return_mask=False):
        node_obs, global_obs = self._split_obs(obs)
        batch_size = obs.shape[0]

        # Expand global features to each node
        global_expanded = global_obs.unsqueeze(1).expand(
            batch_size, self.num_nodes, self.global_features
        )
        # Concatenate: (batch, num_nodes, node_features + global_features)
        node_input = torch.cat([node_obs, global_expanded], dim=-1)

        # Score each node with shared network
        scores = self.scorer(node_input).squeeze(-1)  # (batch, num_nodes)
        scores, valid_mask = self._apply_action_mask(scores, obs)

        if self.critic_mode == "pooled":
            value = self._pooled_value(node_input, global_obs, valid_mask)
        else:
            value = self.critic(obs)
        if return_mask:
            return scores, value, valid_mask
        return scores, value

    def _gaussian_dist(self, logits):
        log_std = torch.clamp(self.actor_log_std, -4.6, 0.0)
        if log_std.dim() == 0:
            std = torch.exp(log_std).expand_as(logits)
        else:
            std = torch.exp(log_std).expand_as(logits)
        return torch.distributions.Normal(logits, std)

    def _masked_gaussian_stats(self, dist, action, valid_mask):
        log_prob = dist.log_prob(action).masked_fill(~valid_mask, 0.0).sum(dim=-1)
        entropy = dist.entropy().masked_fill(~valid_mask, 0.0).sum(dim=-1)
        return log_prob, entropy

    def _plackett_luce_log_prob(self, logits, action, valid_mask):
        """Log-probability of the selected top-k set under sequential sampling.

        This provides a principled PPO-compatible objective for the optional
        Gumbel-TopK actor mode. The environment still receives a score vector;
        the selected committee is represented by the top-k entries of action.
        """
        k = min(self.committee_size, self.num_nodes)
        masked_action = action.masked_fill(~valid_mask, -1e9)
        selected = torch.topk(masked_action, k=k, dim=-1).indices
        work_logits = logits.masked_fill(~valid_mask, -1e9).clone()
        logp = torch.zeros(logits.shape[0], device=logits.device)

        for j in range(k):
            idx = selected[:, j]
            chosen = work_logits.gather(1, idx.unsqueeze(1)).squeeze(1)
            denom = torch.logsumexp(work_logits, dim=-1)
            valid_choice = torch.isfinite(chosen) & torch.isfinite(denom)
            logp = logp + torch.where(valid_choice, chosen - denom, torch.zeros_like(logp))
            remove = torch.zeros_like(work_logits, dtype=torch.bool)
            remove = remove.scatter(1, idx.unsqueeze(1), True)
            work_logits = work_logits.masked_fill(remove, -1e9)
        return logp

    def _masked_categorical_entropy(self, logits, valid_mask):
        masked_logits = logits.masked_fill(~valid_mask, -1e9)
        probs = torch.softmax(masked_logits, dim=-1)
        log_probs = torch.log_softmax(masked_logits, dim=-1)
        ent = -(probs * log_probs).masked_fill(~valid_mask, 0.0).sum(dim=-1)
        return ent

    def evaluate_actions(self, obs, action):
        logits, value, valid_mask = self.forward(obs, return_mask=True)
        if self.actor_mode == "gumbel_topk":
            log_prob = self._plackett_luce_log_prob(logits, action, valid_mask)
            entropy = self._masked_categorical_entropy(logits, valid_mask).mean()
            return log_prob, entropy, value.squeeze(-1)

        dist = self._gaussian_dist(logits)
        log_prob, entropy_vec = self._masked_gaussian_stats(dist, action, valid_mask)
        entropy = entropy_vec.mean()
        return log_prob, entropy, value.squeeze(-1)

    def get_action(self, obs, deterministic=False):
        mean, value, valid_mask = self.forward(obs, return_mask=True)

        if self.actor_mode == "gumbel_topk":
            if deterministic:
                action = mean.masked_fill(~valid_mask, -1e9)
            else:
                u = torch.rand_like(mean).clamp_(1e-6, 1.0 - 1e-6)
                gumbel = -torch.log(-torch.log(u))
                relaxed = torch.softmax(
                    (mean + gumbel) / max(self.gumbel_temperature, 1e-3),
                    dim=-1
                )
                action = (relaxed * float(self.num_nodes)).masked_fill(~valid_mask, -1e9)
            log_prob = self._plackett_luce_log_prob(mean, action, valid_mask)
            entropy = self._masked_categorical_entropy(mean, valid_mask) * float(self.committee_size)
            return action, log_prob, value.squeeze(-1), entropy

        dist = self._gaussian_dist(mean)

        if deterministic:
            action = mean
        else:
            action = dist.sample()

        action = action.masked_fill(~valid_mask, -1e9)
        log_prob, entropy = self._masked_gaussian_stats(dist, action, valid_mask)
        return action, log_prob, value.squeeze(-1), entropy


class PPOTrainer:

    def __init__(self, env, lr=3e-4, gamma=0.99, clip_eps=0.2,
                 epochs=10, save_dir="drl/models", n_envs=None, seed=None,
                 actor_mode="gaussian_masked", gumbel_temperature=0.7,
                 critic_mode="flat"):
        # Auto-detect: use all logical CPUs (leave 1 for main thread)
        if n_envs is None:
            n_envs = max(2, (os.cpu_count() or 4) - 1)
        self.env = env
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.save_dir = save_dir
        self.n_envs = n_envs
        self.seed = seed
        self.actor_mode = actor_mode
        self.gumbel_temperature = gumbel_temperature
        self.critic_mode = critic_mode

        # Create TRUE parallel environments only for training rollout
        # collection. Evaluation/sensitivity uses self.env directly and does
        # not need worker processes.
        self.vec_env = None
        if n_envs > 0:
            env_kwargs = dict(
                num_nodes=env.num_nodes,
                committee_size=env.committee_size,
                max_steps=env.max_steps,
                battery_drain_base=env.battery_drain_base,
                battery_drain_committee=env.battery_drain_committee,
                use_real_containers=False,
                training_mode=env.training_mode,
                training_profile=env.training_profile,
            )
            self.vec_env = SubprocVecEnv(env_kwargs, n_envs=n_envs, base_seed=seed)

        # Auto-detect GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs(save_dir, exist_ok=True)

        self.net = PerNodeScoringNetwork(
            node_features=env.NODE_FEATURES,
            global_features=env.GLOBAL_FEATURES,
            num_nodes=env.num_nodes,
            committee_size=env.committee_size,
            actor_mode=actor_mode,
            gumbel_temperature=gumbel_temperature,
            critic_mode=critic_mode,
        ).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.best_avg_reward = -float('inf')

        # Print param count
        total_params = sum(p.numel() for p in self.net.parameters())
        scorer_params = sum(p.numel() for p in self.net.scorer.parameters())
        print(f"[PPO] Device: {self.device}")
        if self.device.type == 'cuda':
            print(f"[PPO] GPU: {torch.cuda.get_device_name(0)}")
        if n_envs > 0:
            print(f"[PPO] Parallel envs: {n_envs} (multiprocessing)")
        else:
            print("[PPO] Parallel envs: disabled for evaluation")
        print(f"[PPO] Total params: {total_params:,} (scorer: {scorer_params:,})")
        print(f"[PPO] Actor mode: {actor_mode} (gumbel_temperature={gumbel_temperature})")
        print(f"[PPO] Critic mode: {critic_mode}")

    def collect_rollout(self, steps=1000):
        """Collect rollout from N_ENVS parallel environments.
        
        Worker processes run env.step() in parallel on CPU cores. The main
        process batches observations into one GPU forward pass for all envs.
        """
        if self.vec_env is None or self.n_envs <= 0:
            raise RuntimeError("collect_rollout requires n_envs > 0")
        N_ENVS = self.n_envs
        steps_per_env = max(1, int(np.ceil(steps / N_ENVS)))

        obs_list, act_list, rew_list, val_list, logp_list, done_list = \
            [], [], [], [], [], []
        tps_list, alive_list, bat_list = [], [], []
        last_dones = np.zeros(N_ENVS, dtype=bool)

        # Continue environments across PPO updates instead of resetting at the
        # start of every rollout. Resetting here made each worker see only
        # ceil(rollout_steps / n_envs) consecutive steps, so robust-profile
        # disasters scheduled around step 900--1100 were never reached when
        # using many workers.
        obs_batch = self.vec_env.obs
        obs_t = torch.FloatTensor(obs_batch).to(self.device)  # (N_ENVS, obs_dim)

        for _ in range(steps_per_env):
            with torch.no_grad():
                # 1 GPU forward pass for all envs
                action, log_prob, value, _ = self.net.get_action(obs_t)

            action_np = action.cpu().numpy()  # (N_ENVS, num_nodes)

            # Step all envs in parallel worker processes
            next_obs, rewards, dones, infos = self.vec_env.step(action_np)
            last_dones = np.asarray(dones, dtype=bool)

            # Collect data from all envs
            for e in range(N_ENVS):
                obs_list.append(obs_t[e].cpu())
                act_list.append(action[e].cpu())
                rew_list.append(rewards[e])
                val_list.append(value[e].item())
                logp_list.append(log_prob[e].item())
                done_list.append(bool(dones[e]))
                tps_list.append(infos[e].get('tps', 0))
                alive_list.append(infos[e].get('alive', 0))
                bat_list.append(infos[e].get('avg_battery', 0))

            obs_t = torch.FloatTensor(next_obs).to(self.device)

        # Bootstrap value for GAE
        with torch.no_grad():
            _, last_values = self.net(obs_t)
        last_val = last_values.squeeze(-1).cpu().numpy().astype(np.float32)
        last_val *= (1.0 - last_dones.astype(np.float32))

        return {
            'obs': torch.stack(obs_list),
            'act': torch.stack(act_list),
            'rew': np.array(rew_list, dtype=np.float32),
            'val': np.array(val_list, dtype=np.float32),
            'logp': np.array(logp_list, dtype=np.float32),
            'done': np.array(done_list),
            'last_val': last_val,
            'n_envs': N_ENVS,
            'time_steps': steps_per_env,
            'target_steps': steps,
            'actual_steps': len(rew_list),
            'tps': np.mean(tps_list),
            'alive': np.mean(alive_list),
            'bat': np.mean(bat_list),
        }

    def compute_gae(self, rewards, values, dones, last_val=0, gamma=0.99, lam=0.95):
        rewards = np.asarray(rewards, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)

        if rewards.ndim == 2:
            advantages = np.zeros_like(rewards, dtype=np.float32)
            last_gae = np.zeros(rewards.shape[1], dtype=np.float32)
            last_val = np.asarray(last_val, dtype=np.float32)
            for t in reversed(range(rewards.shape[0])):
                next_val = values[t + 1] if t < rewards.shape[0] - 1 else last_val
                next_nonterminal = 1.0 - dones[t]
                delta = rewards[t] + gamma * next_val * next_nonterminal - values[t]
                last_gae = delta + gamma * lam * next_nonterminal * last_gae
                advantages[t] = last_gae
            return advantages, advantages + values

        advantages = np.zeros_like(rewards)
        last_gae = 0
        for t in reversed(range(len(rewards))):
            next_val = values[t + 1] if t < len(rewards) - 1 else last_val
            delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
            last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
        return advantages, advantages + values

    def update(self, rollout):
        obs = rollout['obs']
        act = rollout['act']
        old_logp = torch.FloatTensor(rollout['logp'])

        n_envs = int(rollout.get('n_envs', self.n_envs))
        rewards = rollout['rew'].reshape(-1, n_envs)
        values = rollout['val'].reshape(-1, n_envs)
        dones = rollout['done'].reshape(-1, n_envs)
        advantages, returns = self.compute_gae(
            rewards, values, dones, last_val=rollout['last_val']
        )
        advantages = advantages.reshape(-1)
        returns = returns.reshape(-1)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size = min(1024, len(obs))  # large batch for GPU utilization
        indices = np.arange(len(obs))
        total_loss = 0
        num_updates = 0

        for _ in range(self.epochs):
            np.random.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                idx = indices[start:start + batch_size]
                mb_obs = obs[idx].to(self.device)
                mb_act = act[idx].to(self.device)
                mb_old_logp = old_logp[idx].to(self.device)
                mb_adv = advantages[idx].to(self.device)
                mb_ret = returns[idx].to(self.device)

                new_logp, entropy, values = self.net.evaluate_actions(mb_obs, mb_act)

                ratio = torch.exp(new_logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = 0.5 * (mb_ret - values.squeeze(-1)).pow(2).mean()
                loss = actor_loss + 0.5 * critic_loss - 0.03 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optimizer.step()
                total_loss += loss.item()
                num_updates += 1

        return total_loss / max(num_updates, 1)

    def train(self, num_episodes=3000, rollout_steps=1000):
        episode_rewards = []
        episode_tps = []
        episode_alive = []
        episode_battery = []
        losses = []
        avg50_rewards = []
        recent_rewards = deque(maxlen=50)

        print(f"[PPO] Training {num_episodes} updates, {rollout_steps} transitions/update")
        print(f"[PPO] Parallel envs: {self.n_envs} (multiprocessing)")
        print("[PPO] Worker environments continue across updates until done/truncated")
        print(f"[PPO] Nodes: {self.env.num_nodes}, Committee: {self.env.committee_size}")
        print(f"[PPO] Training profile: {self.env.training_profile}")
        print(f"[PPO] Battery drain: idle={self.env.battery_drain_base}, "
              f"committee={self.env.battery_drain_committee}")
        print(f"[PPO] Architecture: Per-Node Scoring Network")
        print(f"[PPO] Actor mode: {self.actor_mode}")
        print(f"[PPO] Critic mode: {self.critic_mode}")
        print(f"{'='*70}")

        import time
        t_start = time.time()

        for ep in range(num_episodes):
            t_ep = time.time()
            rollout = self.collect_rollout(rollout_steps)
            loss = self.update(rollout)

            ep_reward = float(rollout['rew'].sum()) / self.n_envs
            ep_tps = float(rollout['tps'])
            ep_alive = float(rollout['alive'])
            ep_bat = float(rollout['bat'])
            n_data = len(rollout['rew'])

            episode_rewards.append(ep_reward)
            episode_tps.append(ep_tps)
            episode_alive.append(ep_alive)
            episode_battery.append(ep_bat)
            losses.append(float(loss))
            recent_rewards.append(ep_reward)
            avg_r = float(np.mean(recent_rewards))
            avg50_rewards.append(avg_r)

            if avg_r > self.best_avg_reward:
                self.best_avg_reward = avg_r
                self.save(os.path.join(self.save_dir, "ppo_best.pt"))

            if (ep + 1) % 20 == 0:
                elapsed = time.time() - t_start
                ep_time = time.time() - t_ep
                steps_sec = n_data / ep_time
                eta = (num_episodes - ep - 1) * elapsed / (ep + 1)
                print(f"  Ep {ep+1:4d} | R: {ep_reward:7.1f} | Avg50: {avg_r:7.1f} | "
                      f"Loss: {loss:.4f} | TPS: {ep_tps:.0f} | "
                      f"Alive: {ep_alive:.0f} | Bat: {ep_bat:.2f} | "
                      f"{steps_sec:.0f} stp/s | ETA: {eta/60:.1f}min")

        self.save(os.path.join(self.save_dir, "ppo_final.pt"))

        log_path = os.path.join(self.save_dir, "training_log.json")
        with open(log_path, 'w') as f:
            json.dump({
                "episodes": num_episodes,
                "rollout_steps": rollout_steps,
                "n_envs": self.n_envs,
                "steps_per_env_per_update": int(np.ceil(rollout_steps / self.n_envs)),
                "env_continues_across_updates": True,
                "seed": self.seed,
                "config": {
                    "num_nodes": self.env.num_nodes,
                    "committee_size": self.env.committee_size,
                    "training_profile": self.env.training_profile,
                    "actor_mode": self.actor_mode,
                    "gumbel_temperature": self.gumbel_temperature,
                    "critic_mode": self.critic_mode,
                    "gamma": self.gamma,
                    "clip_eps": self.clip_eps,
                    "epochs": self.epochs,
                },
                "rewards": episode_rewards,
                "avg50_rewards": avg50_rewards,
                "tps": episode_tps,
                "alive": episode_alive,
                "avg_battery": episode_battery,
                "loss": losses,
                "actual_transitions_per_rollout": (
                    int(np.ceil(rollout_steps / self.n_envs)) * self.n_envs
                ),
            }, f, indent=2, cls=NumpyEncoder)

        # Cleanup workers
        self.vec_env.close()

        total_time = time.time() - t_start
        print(f"\n{'='*70}")
        print(f"[PPO] Done! Best avg reward: {self.best_avg_reward:.1f}")
        print(f"[PPO] Total time: {total_time/60:.1f} min")
        return episode_rewards

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path, required=False):
        if os.path.exists(path):
            self.net.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
            print(f"[PPO] Loaded model from {path} (device: {self.device})")
            return True
        else:
            msg = f"Model not found: {path}"
            if required:
                raise FileNotFoundError(msg)
            print(f"[WARN] {msg}")
            return False


# ============================================================
# Baselines
# ============================================================

def get_ocd_action(env, rng=None):
    """
    Optimal Cross-Chain Decision-Making (OCD) Algorithm
    ===================================================
    Faithful implementation of Algorithms 1-4 from:
      Xie et al., "Cross-Chain-Based Trustworthy Node Identity
      Governance in Internet of Things", IEEE IoT Journal, Vol.10,
      No.24, Dec 2023. DOI: 10.1109/JIOT.2023.3308130

    Adaptation for committee selection:
      - Cross-chain transactions (CT_i) → Committee slots (K positions)
      - Arbitration Committees (AC_α) → Candidate nodes (N alive nodes)
      - E_AC_α (committee decentralization) → Device type diversity
      - C(AC_α(CT_i)) (computing resource) → Energy cost per node
      - R(AC_α(CT_i)) (routing cost) → Communication unreliability
      - T(AC_α(CT_i)) (consensus time) → Inverse throughput contribution

    Pipeline (Algorithm 1):
      Step 1: Filter OAC-Matrix by E_c ≤ E_AC_α ≤ NIC → AC-Matrix
      Step 2: Map AC-Matrix → CRC-Matrix, RC-Matrix, CET-Matrix
      Step 3: Algorithm 2 (CRC-MN) → CRCS-Matrix
      Step 4: Algorithm 3 (RC-ST)  → ACS-Matrix'
      Step 5: Algorithm 4 (CET-CD) → OCTAS
    """
    if rng is None:
        rng = np.random.RandomState()
    action = np.zeros(env.num_nodes, dtype=np.float32)
    alive_idx = np.where(env.alive > 0)[0]
    K = min(env.committee_size, len(alive_idx))
    if K == 0:
        return action
    N = len(alive_idx)
    if N <= K:
        for idx in alive_idx:
            action[idx] = 1.0
        return action

    # ==============================================================
    # Build per-node cost vectors (OAC-Matrix tuples)
    # Each candidate j has: <E_j, C_j, R_j, T_j>
    # ==============================================================
    node_types_alive = [env.node_types[i] for i in alive_idx]
    type_set = sorted(set(node_types_alive))
    NIC = len(type_set)  # Number of IoT blockchain types (Chains)

    # E: device type ID for decentralization check
    E_node = np.array([type_set.index(env.node_types[alive_idx[j]])
                       for j in range(N)], dtype=int)

    # C: Computing resource cost — energy expenditure per node
    # Higher power draw + lower battery = higher cost
    C_node = np.zeros(N, dtype=np.float64)
    for j in range(N):
        nid = alive_idx[j]
        bat_ratio = env.batteries[nid] / max(env.node_bat_cap[nid], 0.01)
        power = env.node_power_norm[nid]
        # Energy cost: power consumption weighted by battery scarcity
        C_node[j] = power * (1.0 / max(bat_ratio, 0.05))

    # R: Routing cost — communication unreliability
    # Combines device reliability and current network jitter
    R_node = np.zeros(N, dtype=np.float64)
    for j in range(N):
        nid = alive_idx[j]
        rel = env.node_reliability[nid]
        jit = env.jitter[nid]
        R_node[j] = (1.0 - rel) + (1.0 - jit)

    # T: Consensus execution time — inverse of throughput contribution
    # Lower TPS / lower trust / lower battery → higher consensus time
    T_node = np.zeros(N, dtype=np.float64)
    for j in range(N):
        nid = alive_idx[j]
        bat_ratio = env.batteries[nid] / max(env.node_bat_cap[nid], 0.01)
        noisy_trust = np.clip(
            env.trust_scores[nid] + rng.normal(0, 0.15), 0.01, 1.0)
        tps = env.node_base_tps[nid]
        T_node[j] = 1.0 / max(tps * bat_ratio * noisy_trust, 0.01)

    # ==============================================================
    # ALGORITHM 1, Step 1: Generate AC-Matrix
    # Filter by decentralization: E_c ≤ E_AC_α ≤ NIC
    # Ensure committee has at least E_c different device types
    # ==============================================================
    # Paper Section II-E: E_c constrains decentralization to prevent
    # collusion attacks. Higher E_c = more device type diversity required.
    # With 7 device types, E_c=5 ensures committee draws from ≥5 types.
    E_c = min(5, NIC)  # minimum decentralization requirement

    # Pre-select one node per mandatory type to guarantee E_c diversity
    # Pick the node with lowest consensus time (highest TPS) from each type
    # This ensures diverse capabilities in committee (paper Section II-E:
    # "dynamically adjust the composition of committee nodes")
    mandatory = []
    used = set()
    for t_idx in range(min(E_c, len(type_set))):
        t_name = type_set[t_idx]
        candidates = [j for j in range(N)
                      if node_types_alive[j] == t_name and j not in used]
        if candidates:
            best = min(candidates, key=lambda x: T_node[x])
            mandatory.append(best)
            used.add(best)

    # ==============================================================
    # ALGORITHM 1, Step 2: Matrix Mapping
    # Build K × N cost matrices: CRC-Matrix, RC-Matrix, CET-Matrix
    #
    # Each cell [i][j] = cost of assigning committee slot i to node j
    # Rows represent committee slots (analogous to cross-chain transactions)
    # Columns represent candidate nodes (analogous to ACs)
    # ==============================================================
    CRC_Matrix = np.zeros((K, N), dtype=np.float64)
    RC_Matrix = np.zeros((K, N), dtype=np.float64)
    CET_Matrix = np.zeros((K, N), dtype=np.float64)

    for i in range(K):
        for j in range(N):
            # Per-slot variation: different committee positions interact
            # with different network segments (routing topology effect)
            slot_factor_r = 1.0 + 0.1 * np.sin(2.0 * np.pi * i / K
                                                + E_node[j] * 0.5)
            slot_factor_t = 1.0 + 0.05 * np.cos(2.0 * np.pi * i / K
                                                 + E_node[j] * 0.3)
            CRC_Matrix[i][j] = C_node[j]
            RC_Matrix[i][j] = R_node[j] * max(0.5, slot_factor_r)
            CET_Matrix[i][j] = T_node[j] * max(0.5, slot_factor_t)

    # Constraints — Paper Section II-E: upper bounds on total resource/routing
    # C_c set at 90th percentile to allow mix of high/low power nodes
    # (too tight = only low-power nodes selected, ignoring high-TPS ones)
    C_c = np.percentile(C_node, 90) * K   # computing resource budget
    R_c = np.percentile(R_node, 90) * K   # routing cost budget

    # ==============================================================
    # ALGORITHM 2: CRC-MN (Computational Resource Consumption-Based
    #              Matrix Normalization)
    #
    # Input:  CRC-Matrix, RC-Matrix
    # Output: CRCS-Matrix
    #
    # Step 1: Greedy on CRC-Matrix → ACS-Matrix meeting 0 ≤ C_total ≤ C_c
    # Step 2: Normalize RC-Matrix relative to greedy assignments
    #         CRCS[i][j] = RC[i][j] - RC[i][k] where k = ACS[i]
    # ==============================================================

    ACS = np.full(K, -1, dtype=int)  # ACS[i] = node index for slot i

    # Fill mandatory slots first (decentralization guarantee)
    for slot_idx, node_idx in enumerate(mandatory):
        if slot_idx < K:
            ACS[slot_idx] = node_idx

    # Greedy fill remaining: for each unfilled slot, pick cheapest
    # unused node that keeps C_total ≤ C_c
    for i in range(len(mandatory), K):
        best_j = -1
        best_cost = float('inf')
        for j in range(N):
            if j in used:
                continue
            if CRC_Matrix[i][j] < best_cost:
                # Check if adding this node keeps C_total within budget
                current_C = sum(CRC_Matrix[s][ACS[s]]
                                for s in range(i) if ACS[s] >= 0)
                if current_C + CRC_Matrix[i][j] <= C_c:
                    best_cost = CRC_Matrix[i][j]
                    best_j = j
        # If no node fits budget, pick cheapest anyway (relax constraint)
        if best_j == -1:
            for j in range(N):
                if j not in used and CRC_Matrix[i][j] < best_cost:
                    best_cost = CRC_Matrix[i][j]
                    best_j = j
        ACS[i] = best_j
        used.add(best_j)

    # Normalization: CRCS[i][j] = RC[i][j] - RC[i][ACS[i]]
    # After this, CRCS[i][ACS[i]] = 0 for all i
    # Negative values mean "better than current assignment"
    CRCS_Matrix = np.zeros((K, N), dtype=np.float64)
    for i in range(K):
        k = ACS[i]
        for j in range(N):
            CRCS_Matrix[i][j] = RC_Matrix[i][j] - RC_Matrix[i][k]

    # ==============================================================
    # ALGORITHM 3: RC-ST (Routing Cost-Based Smart Transfer)
    #
    # Input:  CRCS-Matrix, ACS-Matrix
    # Output: ACS-Matrix' (improved assignment)
    #
    # Core: For each slot i with a cheaper alternative (CRCS[i][j] < 0),
    #       find partner slot l to swap assignments such that total
    #       routing cost decreases: CRCS[i][j] + CRCS[l][k] < 0
    #       where k is i's current assignment.
    #
    # The "Smart Transfer" is a 2×2 swap operation between two
    # transaction-AC pairs that reduces total cost.
    # ==============================================================

    ACS_prime = _rc_st_smart_transfer(CRCS_Matrix, ACS.copy(), CRC_Matrix,
                                      C_c, K, N, used_set=None)

    # ==============================================================
    # ALGORITHM 4: CET-CD (Consensus Execution Time-Based
    #              Cross-Chain Decision-Making)
    #
    # Input:  ACS-Matrix', CET-Matrix
    # Output: OCTAS (Optimal Cross-chain Transaction Allocation Strategy)
    #
    # Step 1: Normalize CET-Matrix relative to ACS' assignments → RCS-Matrix
    #         RCS[i][j] = CET[i][j] - CET[i][ACS'[i]]
    # Step 2: Smart Transfer on RCS-Matrix to minimize consensus time
    # ==============================================================

    # Normalize CET-Matrix around current assignments (same as CRC-MN step 2)
    RCS_Matrix = np.zeros((K, N), dtype=np.float64)
    for i in range(K):
        assigned = ACS_prime[i]
        for j in range(N):
            RCS_Matrix[i][j] = CET_Matrix[i][j] - CET_Matrix[i][assigned]

    OCTAS = _cet_cd_smart_transfer(RCS_Matrix, ACS_prime.copy(), CRC_Matrix,
                                    C_c, R_c, RC_Matrix, K, N)

    # ==============================================================
    # Post-processing: Enforce decentralization constraint E_c
    # Smart Transfer may have swapped mandatory nodes out.
    # Re-inject one node per missing type (replace highest-T node).
    # Paper Section II-E: "preventing IoT blockchains from mastering
    # multiple committee nodes on the relay chain"
    # ==============================================================
    selected_set = set(OCTAS)
    selected_types = set(node_types_alive[j] for j in OCTAS if 0 <= j < N)
    missing_types = [t for t in type_set[:E_c] if t not in selected_types]

    for missing_t in missing_types:
        # Find best candidate of this type (lowest T = highest TPS)
        candidates = [j for j in range(N)
                      if node_types_alive[j] == missing_t and j not in selected_set]
        if not candidates:
            continue
        inject_node = min(candidates, key=lambda x: T_node[x])

        # Replace the slot with highest consensus time (least useful)
        worst_slot = max(range(K), key=lambda s: T_node[OCTAS[s]])
        old_node = OCTAS[worst_slot]
        OCTAS[worst_slot] = inject_node
        selected_set.discard(old_node)
        selected_set.add(inject_node)

    # ==============================================================
    # Convert OCTAS to action vector
    # ==============================================================
    for i in range(K):
        node_local = OCTAS[i]
        if 0 <= node_local < N:
            action[alive_idx[node_local]] = 1.0

    return action


def _rc_st_smart_transfer(CRCS, ACS, CRC, C_c, K, N, used_set=None):
    """
    Algorithm 3: RC-ST (Routing Cost-Based Smart Transfer)
    Faithful to pseudo-code in Xie et al. IEEE IoT Journal 2023, p.21588

    Paper pseudo-code (adapted for committee selection):
      for i in rows:
        for j in columns:
          if CRCS[i][j] < 0:
            find k where CRCS[i][k] = 0  (current assignment of i)
            min ← 0; changeIndex ← -1
            for l in rows:
              if CRCS[l][k] = 0 and min > CRCS[i][j] + CRCS[l][k]:
                min = CRCS[i][j] + CRCS[l][k]; changeIndex ← l
            if changeIndex != -1:
              Swap ACS[i] and ACS[changeIndex]
              row changeIndex -= CRCS[changeIndex][k]
              row i -= CRCS[i][j]   (paper says "add" but CRCS[i][j]<0)

    Adaptation note: In committee selection, each node fills exactly one
    slot (unique assignment). The paper allows shared ACs. We handle
    uniqueness by also considering direct reassignment when a node is free.

    The condition CRCS[l][k]=0 means AC k has equal routing cost as l's
    current assignment. By normalization, CRCS[l][ACS[l]]=0 always.
    So CRCS[l][k]=0 typically means k=ACS[l] (l is also assigned to k).
    In committee selection (unique assignments), this is impossible
    (only slot i uses k). We relax to |CRCS[l][k]| < ε (near-zero)
    to capture the paper's intent: swap when l barely loses from the
    exchange.
    """
    ACS = ACS.copy()
    CRCS = CRCS.copy()
    EPSILON = 0.05  # tolerance for "approximately zero" routing cost diff

    # --- Paper Algorithm 3: lines 3-29 ---
    for i in range(K):
        for j in range(N):
            # Paper line 6: if CRCS[i][j] < 0
            if CRCS[i][j] >= 0:
                continue

            # Paper lines 8-12: find k where CRCS[i][k] = 0
            # k is column of i's current assignment
            k = ACS[i]
            # Verify: CRCS[i][k] should be 0 (by normalization)
            # assert abs(CRCS[i][k]) < 1e-10

            # Paper lines 13-21: find best swap partner l
            min_val = 0.0
            change_index = -1

            for l in range(K):
                if l == i:
                    continue

                # Paper line 17: if CRCS[l][k] = 0 (relaxed to |·| < ε)
                # This checks: assigning l to AC k costs about the same
                # as l's current assignment → swap is ~free for l
                if abs(CRCS[l][k]) > EPSILON:
                    continue

                swap_delta = CRCS[i][j] + CRCS[l][k]
                if swap_delta < min_val:
                    # Verify C constraint after swap
                    new_C = 0
                    for s in range(K):
                        if s == i:
                            # i would get ACS[l] (paper: swap values)
                            new_C += CRC[s][ACS[l]]
                        elif s == l:
                            # l would get k (i's old assignment)
                            new_C += CRC[s][k]
                        else:
                            new_C += CRC[s][ACS[s]]
                    if new_C <= C_c:
                        min_val = swap_delta
                        change_index = l

            # Paper line 22: if changeIndex != -1
            if change_index >= 0:
                # Paper line 23: Swap ACS[i] and ACS[changeIndex]
                ACS[i], ACS[change_index] = ACS[change_index], ACS[i]

                # Paper line 24: Each element of changeIndex-th row
                #                minus CRCS[changeIndex][k]
                # Re-normalizes row changeIndex around its new assignment (k)
                shift_l = CRCS[change_index][k]
                CRCS[change_index, :] -= shift_l

                # Paper line 25: Each element of i-th row add CRCS[i][j]
                # CRCS[i][j] is negative, so adding it = subtracting |CRCS[i][j]|
                # This re-normalizes row i around its new assignment
                # Math: CRCS_new[i][c] = CRCS_old[i][c] - CRCS_old[i][new_assignment]
                # Since i now has ACS[changeIndex]_old, and we used CRCS[i][j]
                # as proxy, this works when ACS[change_index]_old ≈ j
                # (which is approximately true when CRCS[l][k] ≈ 0)
                shift_i = CRCS[i][ACS[i]]  # exact re-normalization
                CRCS[i, :] -= shift_i

    return ACS


def _cet_cd_smart_transfer(RCS, ACS, CRC, C_c, R_c, RC, K, N):
    """
    Algorithm 4: CET-CD (Consensus Execution Time-Based
                 Cross-Chain Decision-Making)
    Faithful to pseudo-code in Xie et al. IEEE IoT Journal 2023, p.21588

    Paper pseudo-code:
      1: Initialize RCS-Matrix and OCTAS
      2-5: RCS[i] ← CET[i] - CET[i][ACS'[i]]  (already done before call)
      6-18: Smart Transfer on RCS-Matrix (same pattern as RC-ST)
            if RCS[i][j] < 0:
              Find transaction k, after swapping with i, total time reduced
              Swap ACS'[i] and ACS'[k]; update RCS-Matrix
      19: OCTAS ← ACS'
      20: return OCTAS

    Key difference from RC-ST: must maintain BOTH C_c AND R_c constraints.
    """
    ACS = ACS.copy()
    RCS = RCS.copy()
    EPSILON = 0.05

    # --- Paper Algorithm 4: lines 7-18 ---
    for i in range(K):
        for j in range(N):
            # Paper line 10: if RCS[i][j] < 0
            if RCS[i][j] >= 0:
                continue

            k = ACS[i]  # current assignment for slot i

            # Paper line 11: Find transaction l, after swapping with i,
            # total time is reduced
            min_val = 0.0
            change_index = -1

            for l in range(K):
                if l == i:
                    continue

                # Same logic as RC-ST: check if swap is ~free for l
                if abs(RCS[l][k]) > EPSILON:
                    continue

                swap_delta = RCS[i][j] + RCS[l][k]
                if swap_delta < min_val:
                    # Check BOTH C and R constraints (paper Section III-C)
                    new_C = 0
                    new_R = 0
                    for s in range(K):
                        if s == i:
                            new_C += CRC[s][ACS[l]]
                            new_R += RC[s][ACS[l]]
                        elif s == l:
                            new_C += CRC[s][k]
                            new_R += RC[s][k]
                        else:
                            new_C += CRC[s][ACS[s]]
                            new_R += RC[s][ACS[s]]
                    if new_C <= C_c and new_R <= R_c:
                        min_val = swap_delta
                        change_index = l

            if change_index >= 0:
                # Paper line 12: Swap ACS'[i] and ACS'[changeIndex]
                ACS[i], ACS[change_index] = ACS[change_index], ACS[i]

                # Paper line 14: update RCS-Matrix
                # Re-normalize row changeIndex (now assigned to k)
                shift_l = RCS[change_index][k]
                RCS[change_index, :] -= shift_l

                # Re-normalize row i (now has old ACS[changeIndex])
                shift_i = RCS[i][ACS[i]]  # exact re-normalization
                RCS[i, :] -= shift_i

    return ACS


def _get_noisy_trust(env, rng):
    """Same noisy trust that PPO receives via observations (σ=0.15).
    Fair comparison: all baselines see the same imperfect information."""
    noisy = np.zeros(env.num_nodes, dtype=np.float32)
    for i in range(env.num_nodes):
        if env.alive[i] > 0:
            noise = rng.normal(0, 0.15)
            noisy[i] = np.clip(env.trust_scores[i] + noise, 0.0, 1.0)
    return noisy


def get_sa_action(env, rng=None, T_init=1.0, alpha=0.995, n_iter=150):
    """Simulated Annealing baseline — stronger than Hill Climbing (OCD).

    Uses temperature-based probabilistic acceptance of worse solutions
    to escape local optima.  The objective function balances:
      - State of charge   (battery headroom)
      - CPU capability    (normalized hardware speed)
      - Trust score       (noisy, same σ=0.15 as PPO observation)
      - Diversity bonus   (prefer heterogeneous device types)

    FAIR PLAY: uses noisy trust (σ=0.15), same as PPO's observation.
    """
    if rng is None:
        rng = np.random.RandomState()

    K = env.committee_size
    alive_idx = np.where(env.alive > 0)[0]
    if len(alive_idx) < K:
        action = np.zeros(env.num_nodes, dtype=np.float32)
        action[alive_idx] = 1.0
        return action

    # Fair: snapshot noisy trust ONCE per step (same as PPO sees one obs)
    noisy_trust = _get_noisy_trust(env, rng)

    # Device type array for diversity calculation
    dev_types_arr = getattr(
        env,
        'type_ids',
        getattr(env, 'node_device_type_idx', np.arange(env.num_nodes) % 7),
    )

    # --- Objective function ---
    def _score(committee):
        total = 0.0
        dev_types = set()
        for idx in committee:
            soc = np.clip(env.batteries[idx] / max(env.node_bat_cap[idx], 1e-6),
                          0.0, 1.0)
            trust = noisy_trust[idx]  # FAIR: noisy, not ground truth
            cpu = np.clip(env.node_cpu[idx], 0.0, 1.0)
            total += 0.5 * soc + 0.3 * trust + 0.2 * cpu
            dev_types.add(int(dev_types_arr[idx]))
        # Small diversity bonus: more device types = slightly better.
        diversity = len(dev_types) / min(K, 7)
        return total * (0.95 + 0.05 * diversity)

    # --- Initial solution: top-K by the same composite utility as GBA ---
    scores = np.zeros(len(alive_idx))
    for i, idx in enumerate(alive_idx):
        soc = np.clip(env.batteries[idx] / max(env.node_bat_cap[idx], 1e-6),
                      0.0, 1.0)
        trust = noisy_trust[idx]
        cpu = np.clip(env.node_cpu[idx], 0.0, 1.0)
        scores[i] = 0.5 * soc + 0.3 * trust + 0.2 * cpu
    top_k = np.argsort(scores)[-K:]
    current = [alive_idx[j] for j in top_k]
    current_score = _score(current)
    best = list(current)
    best_score = current_score

    # --- SA loop ---
    T = T_init
    non_committee = [idx for idx in alive_idx if idx not in current]
    for _ in range(n_iter):
        if not non_committee:
            break
        # Swap one committee member with one non-member
        i_swap = rng.randint(0, len(current))
        j_swap = rng.randint(0, len(non_committee))
        candidate = list(current)
        candidate[i_swap] = non_committee[j_swap]
        cand_score = _score(candidate)
        delta = cand_score - current_score
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-6)):
            current = candidate
            current_score = cand_score
            non_committee = [idx for idx in alive_idx if idx not in current]
            if current_score > best_score:
                best = list(current)
                best_score = current_score
        T *= alpha

    action = np.zeros(env.num_nodes, dtype=np.float32)
    for idx in best:
        action[idx] = 1.0
    return action


def get_gba_action(env, rng=None):
    """Greedy Battery-Aware baseline — rational heuristic.

    Selects committee by the paper's composite utility:
        score = 0.5 * SoC + 0.3 * noisy_trust + 0.2 * normalized_cpu.
    Simple but effective — if PPO cannot beat this, DRL complexity
    is not justified.

    FAIR PLAY: uses noisy trust (σ=0.15), same as PPO's observation.
    """
    if rng is None:
        rng = np.random.RandomState()

    K = env.committee_size
    alive_idx = np.where(env.alive > 0)[0]
    if len(alive_idx) < K:
        action = np.zeros(env.num_nodes, dtype=np.float32)
        action[alive_idx] = 1.0
        return action

    # Fair: snapshot noisy trust ONCE per step
    noisy_trust = _get_noisy_trust(env, rng)

    scores = np.zeros(env.num_nodes)
    for idx in alive_idx:
        soc = np.clip(env.batteries[idx] / max(env.node_bat_cap[idx], 1e-6),
                      0.0, 1.0)
        trust = noisy_trust[idx]  # FAIR: noisy, not ground truth
        cpu = np.clip(env.node_cpu[idx], 0.0, 1.0)
        scores[idx] = 0.5 * soc + 0.3 * trust + 0.2 * cpu

    # Top-K selection
    top_k = np.argsort(scores)[-K:]
    action = np.zeros(env.num_nodes, dtype=np.float32)
    for idx in top_k:
        action[idx] = 1.0
    return action


def get_static_action(env, initial_committee=None, rng=None):
    """Fixed-priority static baseline.

    The preferred committee is fixed at step 0. If preferred members die, the
    baseline refills the missing slots randomly from alive nodes so every
    strategy is evaluated with the same committee-size requirement.
    """
    if rng is None:
        rng = np.random.RandomState()

    action = np.zeros(env.num_nodes, dtype=np.float32)
    K = env.committee_size

    if initial_committee is not None:
        alive_fixed = [idx for idx in initial_committee if env.alive[idx] > 0]
        for idx in alive_fixed:
            action[idx] = 1.0
        # BFT: must fill to exactly K members
        if len(alive_fixed) < K:
            alive_idx = np.where(env.alive > 0)[0]
            non_fixed = [n for n in alive_idx if n not in alive_fixed]
            need = K - len(alive_fixed)
            if len(non_fixed) >= need:
                fill = rng.choice(non_fixed, size=need, replace=False)
            else:
                fill = non_fixed
            for idx in fill:
                action[idx] = 0.5  # lower priority than fixed members
    else:
        sorted_all = sorted(range(env.num_nodes),
                           key=lambda x: env.node_base_tps[x], reverse=True)
        for idx in sorted_all[:K]:
            action[idx] = 1.0
    return action


def _strategy_list(strategies=None):
    all_strategies = ["ppo", "sa", "gba", "ocd", "random", "static"]
    if strategies is None:
        return all_strategies
    if isinstance(strategies, str):
        strategies = [s.strip().lower() for s in strategies.split(",") if s.strip()]
    if not strategies:
        return all_strategies
    if "all" in strategies:
        return all_strategies
    unknown = [s for s in strategies if s not in all_strategies]
    if unknown:
        raise ValueError(f"Unknown strategies: {unknown}. Valid: {all_strategies}")
    return list(strategies)


def _parse_ratio_list(text):
    if isinstance(text, (list, tuple)):
        return [float(x) for x in text]
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def _summarize_runs(all_runs, strategy_names, disaster_step, total_steps):
    summary_stats = {}
    for strat in strategy_names:
        pre_list, post_list = [], []
        alive_list, imm_drop_list = [], []
        tps_100_list, tps_500_list, auc_list = [], [], []
        recovery90_list = []
        byz_committee_list = []
        for hist in all_runs[strat]:
            pre = [h['tps'] for h in hist[max(0, disaster_step-100):disaster_step]]
            post_all = [h['tps'] for h in hist[disaster_step:]]
            pre_tps = np.mean(pre) if pre else 0
            post_tps = np.mean(post_all) if post_all else 0
            alive = hist[-1]['alive'] if hist else 0

            imm_post = [h['tps'] for h in hist[disaster_step:disaster_step+50]]
            imm_drop = pre_tps - min(imm_post) if imm_post else 0

            post_100 = [h['tps'] for h in hist[disaster_step:disaster_step+100]]
            tps_100 = np.mean(post_100) if post_100 else 0

            post_500 = [h['tps'] for h in hist[disaster_step:disaster_step+500]]
            tps_500 = np.mean(post_500) if post_500 else 0

            auc = np.sum(post_all) if post_all else 0
            auc_normalized = auc / max(pre_tps * len(post_all), 1)

            post_hist = hist[disaster_step:]
            recovery90 = total_steps
            for j in range(len(post_hist) - 10):
                window = np.mean([post_hist[k]['tps'] for k in range(j, j+10)])
                if window >= pre_tps * 0.9:
                    recovery90 = j
                    break

            byz_post = [
                h.get('byzantine_in_committee', 0)
                for h in hist[disaster_step:]
            ]

            pre_list.append(pre_tps)
            post_list.append(post_tps)
            alive_list.append(alive)
            imm_drop_list.append(imm_drop)
            tps_100_list.append(tps_100)
            tps_500_list.append(tps_500)
            auc_list.append(auc_normalized)
            recovery90_list.append(recovery90)
            byz_committee_list.append(np.mean(byz_post) if byz_post else 0)

        summary_stats[strat] = {
            "pre_tps": pre_list, "post_tps": post_list,
            "tps_100": tps_100_list, "tps_500": tps_500_list,
            "auc": auc_list, "recovery90": recovery90_list,
            "alive": alive_list, "imm_drop": imm_drop_list,
            "byzantine_committee_post_mean": byz_committee_list,
            "retention": [
                post / pre if pre > 0 else 0.0
                for pre, post in zip(pre_list, post_list)
            ],
        }
    return summary_stats


def _run_single_disaster(trainer, kill_ratio, total_steps, disaster_step, seed,
                         byz_ratio=0.20, strategies=None):
    results = {}
    strategy_names = _strategy_list(strategies)
    byz_ratio = float(np.clip(byz_ratio, 0.0, 0.95))
    for strategy in strategy_names:
        np.random.seed(seed)
        trainer.env.training_mode = False
        trainer.env._scheduled_disaster_step = -1
        obs, _ = trainer.env.reset(seed=seed)
        trainer.env.history = []
        history = []

        strategy_rng = np.random.RandomState(seed + 200)

        rng = np.random.RandomState(seed + 100)
        kill_list = rng.choice(
            trainer.env.num_nodes,
            size=int(trainer.env.num_nodes * kill_ratio),
            replace=False,
        ).tolist()
        byz_order = rng.permutation(trainer.env.num_nodes).tolist()

        def _alive_nodes():
            return [
                n for n in byz_order
                if trainer.env.alive[n] > 0 and trainer.env.permanent_dead[n] == 0
            ]

        def _compromised_alive_set():
            compromised = (
                (trainer.env.is_byzantine > 0)
                | (trainer.env.stealth_timer > 0)
            )
            return {
                n for n in range(trainer.env.num_nodes)
                if compromised[n]
                and trainer.env.alive[n] > 0
                and trainer.env.permanent_dead[n] == 0
            }

        # Keep an active Byzantine population from the start of evaluation.
        # Experiment 3 measures recovery under concurrent Byzantine faults, not
        # a stealth-dormancy stress test.
        initial_alive = _alive_nodes()
        initial_byz_count = int(len(initial_alive) * byz_ratio)
        initial_byz_list = initial_alive[:initial_byz_count]
        trainer.env.make_byzantine(initial_byz_list, stealth_ratio=0.0)
        initial_byzantine = len(initial_byz_list)
        # The scenario starts with compromised nodes already present. Refresh
        # the observation so PPO sees the same post-injection trust state that
        # heuristic baselines read directly from the environment.
        obs = trainer.env._get_obs()

        static_committee = None
        if strategy == "static":
            sorted_all = sorted(range(trainer.env.num_nodes),
                               key=lambda x: trainer.env.node_base_tps[x], reverse=True)
            static_committee = sorted_all[:trainer.env.committee_size]

        for step in range(total_steps):
            event_killed = 0
            event_byzantine = 0
            if strategy == "ppo":
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(trainer.device)
                with torch.no_grad():
                    action, _, _, _ = trainer.net.get_action(obs_t, deterministic=True)
                action_np = action.squeeze(0).cpu().numpy()
            elif strategy == "sa":
                action_np = get_sa_action(trainer.env, rng=strategy_rng)
            elif strategy == "gba":
                action_np = get_gba_action(trainer.env, rng=strategy_rng)
            elif strategy == "ocd":
                action_np = get_ocd_action(trainer.env, rng=strategy_rng)
            elif strategy == "random":
                action_np = strategy_rng.rand(trainer.env.num_nodes).astype(np.float32)
            elif strategy == "static":
                action_np = get_static_action(trainer.env, initial_committee=static_committee, rng=strategy_rng)

            # Disaster BEFORE env.step — all strategies see it simultaneously
            if step == disaster_step:
                trainer.env.kill_nodes(kill_list)
                alive_survivors = _alive_nodes()
                target_byz_count = int(len(alive_survivors) * byz_ratio)
                compromised_alive = _compromised_alive_set()
                need = max(0, target_byz_count - len(compromised_alive))
                byz_list = [
                    n for n in alive_survivors
                    if n not in compromised_alive
                ][:need]
                if byz_list:
                    trainer.env.make_byzantine(byz_list, stealth_ratio=0.0)
                event_killed = len(kill_list)
                event_byzantine = len(byz_list)

            obs, reward, terminated, truncated, info = trainer.env.step(action_np)
            compromised_alive = _compromised_alive_set()
            active_byzantine_alive = int(np.sum(
                (trainer.env.is_byzantine > 0) & (trainer.env.alive > 0)
            ))

            history.append({
                "step": step,
                "tps": float(info.get("tps", 0)),
                "alive": int(info.get("alive", 0)),
                "committee_active": int(info.get("committee_active", 0)),
                "avg_battery": float(info.get("avg_battery", 0)),
                "byzantine_in_committee": int(info.get("byzantine_in_committee", 0)),
                "event_killed": event_killed,
                "event_byzantine": event_byzantine,
                "initial_byzantine": initial_byzantine,
                "compromised_alive": int(len(compromised_alive)),
                "active_byzantine_alive": active_byzantine_alive,
                "reward": float(reward),
            })
            if terminated:
                break

        results[strategy] = history
    return results


def run_disaster_test(trainer, model_path, kill_ratio=0.30, byz_ratio=0.20,
                      total_steps=2000, disaster_step=1000,
                      results_dir="results", seeds=None, strategies=None):
    if seeds is None:
        seeds = [42, 123, 456, 789, 1024, 2048, 3333, 4096, 5555, 7777]

    strategy_names = _strategy_list(strategies)
    trainer.load(model_path, required=True)
    os.makedirs(results_dir, exist_ok=True)

    all_runs = {s: [] for s in strategy_names}

    for si, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"[DISASTER] Seed {si+1}/{len(seeds)}: {seed}")
        run_results = _run_single_disaster(
            trainer, kill_ratio, total_steps, disaster_step, seed,
            byz_ratio=byz_ratio, strategies=strategy_names,
        )
        for strat, hist in run_results.items():
            all_runs[strat].append(hist)
            pre = [h['tps'] for h in hist[max(0,disaster_step-100):disaster_step]]
            post = [h['tps'] for h in hist[disaster_step:min(disaster_step+200,len(hist))]]
            pre_tps = np.mean(pre) if pre else 0
            post_tps = np.mean(post) if post else 0
            alive = hist[-1]['alive'] if hist else 0
            print(f"  {strat.upper():8s}: Pre={pre_tps:.1f} Post={post_tps:.1f} Alive={alive}")

    print(f"\n{'='*80}")
    print(f"  STATISTICAL SUMMARY ({len(seeds)} seeds)")
    print(f"{'='*80}")
    print(f"  {'Strategy':10s} {'Pre-TPS':>12s} {'Post-TPS':>12s} "
          f"{'TPS@t+100':>12s} {'TPS@t+500':>12s} {'AUC':>8s} {'Alive':>10s}")
    print(f"  {'-'*80}")

    summary_stats = _summarize_runs(
        all_runs, strategy_names, disaster_step, total_steps
    )
    for strat in strategy_names:
        pre_list = summary_stats[strat]["pre_tps"]
        post_list = summary_stats[strat]["post_tps"]
        tps_100_list = summary_stats[strat]["tps_100"]
        tps_500_list = summary_stats[strat]["tps_500"]
        auc_list = summary_stats[strat]["auc"]
        alive_list = summary_stats[strat]["alive"]
        print(f"  {strat.upper():10s} "
              f"{np.mean(pre_list):6.1f}±{np.std(pre_list):4.1f} "
              f"{np.mean(post_list):6.1f}±{np.std(post_list):4.1f} "
              f"{np.mean(tps_100_list):6.1f}±{np.std(tps_100_list):4.1f} "
              f"{np.mean(tps_500_list):6.1f}±{np.std(tps_500_list):4.1f} "
              f"{np.mean(auc_list):5.2f}±{np.std(auc_list):4.2f} "
              f"{np.mean(alive_list):6.1f}±{np.std(alive_list):4.1f}")

    stats_tests = []
    try:
        from scipy import stats as sp_stats
        print(f"\n  Statistical tests (two-sided Mann-Whitney U, Holm-corrected):")
        ppo_post = summary_stats.get("ppo", {}).get("post_tps", [])
        raw_tests = []
        for strat in [s for s in strategy_names if s != "ppo"]:
            other_post = summary_stats[strat]["post_tps"]
            if len(ppo_post) >= 3 and len(other_post) >= 3:
                u_stat, p_raw = sp_stats.mannwhitneyu(
                    ppo_post, other_post, alternative='two-sided'
                )
                n1, n2 = len(ppo_post), len(other_post)
                rank_biserial = (2.0 * float(u_stat) / (n1 * n2)) - 1.0
                raw_tests.append({
                    "contrast": f"ppo_vs_{strat}",
                    "u": float(u_stat),
                    "p_raw": float(p_raw),
                    "rank_biserial_r": float(rank_biserial),
                    "n_ppo": n1,
                    "n_baseline": n2,
                })

        # Holm step-down correction controls family-wise error for the five
        # paper contrasts while preserving more power than plain Bonferroni.
        m = len(raw_tests)
        prev_adj = 0.0
        for rank, item in enumerate(sorted(raw_tests, key=lambda x: x["p_raw"]), start=1):
            adj = min(1.0, max(prev_adj, (m - rank + 1) * item["p_raw"]))
            item["p_holm"] = float(adj)
            item["significant_0_01"] = bool(adj < 0.01)
            prev_adj = adj
        stats_tests = sorted(raw_tests, key=lambda x: x["contrast"])

        for item in stats_tests:
            sig = "*" if item["significant_0_01"] else "ns"
            print(
                f"    {item['contrast']:18s}: "
                f"U={item['u']:.1f}, p_raw={item['p_raw']:.4g}, "
                f"p_holm={item['p_holm']:.4g}, r={item['rank_biserial_r']:.3f} {sig}"
            )
    except ImportError:
        print("\n  [WARN] scipy not installed")

    output = {
        "config": {
            "kill_ratio": kill_ratio, "byz_ratio": byz_ratio,
            "disaster_step": disaster_step,
            "total_steps": total_steps, "seeds": seeds,
            "strategies": strategy_names,
        },
        "all_runs": all_runs,
        "summary": summary_stats,
        "statistics": {
            "post_tps_tests": stats_tests,
            "test": "two-sided Mann-Whitney U",
            "correction": "Holm",
        },
    }
    output_path = os.path.join(results_dir, "disaster_results.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\n[DISASTER] Saved to {output_path}")
    return output


def run_sensitivity_test(trainer, model_path, kill_ratios=None, byz_ratios=None,
                         total_steps=2000, disaster_step=1000,
                         results_dir="results", seeds=None,
                         strategies=None, include_histories=False):
    """Evaluate robustness outside the main 30% kill + 20% Byzantine scenario."""
    if kill_ratios is None:
        kill_ratios = [0.10, 0.20, 0.30, 0.40]
    if byz_ratios is None:
        byz_ratios = [0.00, 0.10, 0.20, 0.30]
    if seeds is None:
        seeds = [42, 123, 456, 789, 1024, 2048, 3333, 4096, 5555, 7777]

    strategy_names = _strategy_list(strategies or ["ppo", "gba"])
    trainer.load(model_path, required=True)
    os.makedirs(results_dir, exist_ok=True)

    scenario_outputs = []
    flat_rows = []

    total = len(kill_ratios) * len(byz_ratios)
    scenario_idx = 0
    for kill_ratio in kill_ratios:
        for byz_ratio in byz_ratios:
            scenario_idx += 1
            print(f"\n{'='*78}")
            print(
                f"[SENSITIVITY {scenario_idx}/{total}] "
                f"kill={kill_ratio:.0%}, byz={byz_ratio:.0%}, "
                f"strategies={','.join(strategy_names)}"
            )
            print(f"{'='*78}")

            all_runs = {s: [] for s in strategy_names}
            for si, seed in enumerate(seeds):
                run_results = _run_single_disaster(
                    trainer, kill_ratio, total_steps, disaster_step, seed,
                    byz_ratio=byz_ratio, strategies=strategy_names,
                )
                for strat, hist in run_results.items():
                    all_runs[strat].append(hist)
                if (si + 1) % max(1, len(seeds) // 5) == 0 or si == len(seeds) - 1:
                    ppo_hist = run_results.get("ppo", [])
                    if ppo_hist:
                        pre = [h['tps'] for h in ppo_hist[max(0, disaster_step-100):disaster_step]]
                        post = [h['tps'] for h in ppo_hist[disaster_step:]]
                        print(
                            f"  seed {si+1:2d}/{len(seeds)} "
                            f"PPO pre={np.mean(pre) if pre else 0:.1f} "
                            f"post={np.mean(post) if post else 0:.1f}"
                        )

            summary = _summarize_runs(
                all_runs, strategy_names, disaster_step, total_steps
            )

            print("  Summary:")
            for strat in strategy_names:
                stats = summary[strat]
                post = stats["post_tps"]
                retention = stats["retention"]
                auc = stats["auc"]
                alive = stats["alive"]
                print(
                    f"    {strat.upper():8s} "
                    f"Post={np.mean(post):7.1f}±{np.std(post):5.1f} "
                    f"Ret={np.mean(retention)*100:6.1f}% "
                    f"AUC={np.mean(auc):5.3f} "
                    f"Alive={np.mean(alive):5.1f}"
                )
                flat_rows.append({
                    "kill_ratio": kill_ratio,
                    "byz_ratio": byz_ratio,
                    "strategy": strat,
                    "post_tps_mean": float(np.mean(post)),
                    "post_tps_std": float(np.std(post)),
                    "pre_tps_mean": float(np.mean(stats["pre_tps"])),
                    "pre_tps_std": float(np.std(stats["pre_tps"])),
                    "retention_mean": float(np.mean(retention)),
                    "retention_std": float(np.std(retention)),
                    "auc_mean": float(np.mean(auc)),
                    "auc_std": float(np.std(auc)),
                    "alive_mean": float(np.mean(alive)),
                    "alive_std": float(np.std(alive)),
                    "tps_100_mean": float(np.mean(stats["tps_100"])),
                    "tps_500_mean": float(np.mean(stats["tps_500"])),
                    "byzantine_committee_post_mean": float(
                        np.mean(stats["byzantine_committee_post_mean"])
                    ),
                })

            scenario_output = {
                "kill_ratio": kill_ratio,
                "byz_ratio": byz_ratio,
                "summary": summary,
            }
            if include_histories:
                scenario_output["all_runs"] = all_runs
            scenario_outputs.append(scenario_output)

    output = {
        "config": {
            "kill_ratios": kill_ratios,
            "byz_ratios": byz_ratios,
            "disaster_step": disaster_step,
            "total_steps": total_steps,
            "seeds": seeds,
            "strategies": strategy_names,
            "model_path": model_path,
        },
        "scenarios": scenario_outputs,
        "rows": flat_rows,
    }
    output_path = os.path.join(results_dir, "drl_sensitivity.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)

    print(f"\n[SENSITIVITY] Saved to {output_path}")
    return output


def main():
    parser = argparse.ArgumentParser(description='PPO v11 - Per-Node Scoring')
    parser.add_argument('--mode', choices=['train', 'eval', 'disaster', 'sensitivity'], default='train')
    parser.add_argument('--episodes', type=int, default=3000)
    parser.add_argument('--steps', type=int, default=3000,
                        help='Steps per rollout (more = better GPU batch utilization)')
    parser.add_argument('--nodes', type=int, default=100)
    parser.add_argument('--committee', type=int, default=21)
    parser.add_argument('--workers', type=int, default=0,
                        help='Parallel env workers (0=auto-detect CPU count)')
    parser.add_argument('--model', default='drl/models/ppo_best.pt')
    parser.add_argument('--sim', action='store_true')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-dir', default='drl/models',
                        help='Directory for PPO checkpoints and training_log.json')
    parser.add_argument('--train-profile', choices=['robust', 'standard'],
                        default='robust',
                        help='Training fault profile: robust matches Experiment 3')
    parser.add_argument('--actor-mode',
                        choices=['gaussian_masked', 'gumbel_topk'],
                        default='gaussian_masked',
                        help='Policy head: masked Gaussian scores or experimental Gumbel-TopK relaxation')
    parser.add_argument('--gumbel-temperature', type=float, default=0.7,
                        help='Temperature for --actor-mode gumbel_topk')
    parser.add_argument('--critic-mode',
                        choices=['flat', 'pooled'],
                        default='flat',
                        help='Value head: flat keeps legacy checkpoints; pooled is permutation-invariant and N-transfer friendly')
    parser.add_argument('--kill-ratios', default='0.10,0.20,0.30,0.40',
                        help='Comma-separated kill ratios for sensitivity mode')
    parser.add_argument('--byz-ratios', default='0.00,0.10,0.20,0.30',
                        help='Comma-separated active Byzantine ratios for sensitivity mode')
    parser.add_argument('--strategies', default='ppo,gba',
                        help='Comma-separated strategies for disaster/sensitivity, or all')
    parser.add_argument('--total-steps', type=int, default=2000,
                        help='Evaluation horizon for disaster/sensitivity modes')
    parser.add_argument('--disaster-step', type=int, default=1000,
                        help='Fault injection step for disaster/sensitivity modes')
    parser.add_argument('--results-dir', default='results',
                        help='Directory for evaluation JSON outputs')
    parser.add_argument('--eval-seeds', default='42,123,456,789,1024,2048,3333,4096,5555,7777',
                        help='Comma-separated evaluation seeds')
    parser.add_argument('--include-histories', action='store_true',
                        help='Store full histories in drl_sensitivity.json')
    args = parser.parse_args()

    if not HAS_TORCH:
        print("[ERROR] PyTorch required.")
        sys.exit(1)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    is_training = (args.mode == 'train')
    max_steps = args.steps if is_training else args.total_steps
    env = IoTNetworkEnv(
        num_nodes=args.nodes,
        committee_size=args.committee,
        max_steps=max_steps,
        use_real_containers=not args.sim,
        training_mode=is_training,
        training_profile=args.train_profile,
    )

    if is_training:
        n_workers = args.workers if args.workers > 0 else None  # None = auto
    else:
        # Evaluation/sensitivity uses the single foreground environment; avoid
        # spawning unused rollout workers just to load a checkpoint.
        n_workers = 0
    trainer = PPOTrainer(
        env, lr=args.lr, n_envs=n_workers, seed=args.seed,
        save_dir=args.save_dir,
        actor_mode=args.actor_mode,
        gumbel_temperature=args.gumbel_temperature,
        critic_mode=args.critic_mode,
    )

    if args.mode == 'train':
        print(f"[PPO] Mode: TRAIN ({'sim' if args.sim else 'real'})")
        trainer.train(num_episodes=args.episodes, rollout_steps=args.steps)

    elif args.mode == 'eval':
        trainer.load(args.model)
        obs, _ = env.reset()
        total_reward = 0
        for step in range(1000):
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                action, _, _, _ = trainer.net.get_action(obs_t, deterministic=True)
            obs, reward, done, trunc, info = env.step(action.squeeze(0).numpy())
            total_reward += reward
            if (step + 1) % 100 == 0:
                print(f"  Step {step+1}: TPS={info['tps']:.1f}, "
                      f"Alive={info['alive']}, Battery={info['avg_battery']:.2f}")
            if done or trunc:
                break
        print(f"\nTotal reward: {total_reward:.1f}")

    elif args.mode == 'disaster':
        run_disaster_test(
            trainer, args.model,
            kill_ratio=0.30,
            byz_ratio=0.20,
            disaster_step=args.disaster_step,
            total_steps=args.total_steps,
            seeds=[int(s) for s in _parse_ratio_list(args.eval_seeds)],
            results_dir=args.results_dir,
            strategies=args.strategies,
        )

    elif args.mode == 'sensitivity':
        run_sensitivity_test(
            trainer, args.model,
            kill_ratios=_parse_ratio_list(args.kill_ratios),
            byz_ratios=_parse_ratio_list(args.byz_ratios),
            disaster_step=args.disaster_step,
            total_steps=args.total_steps,
            seeds=[int(s) for s in _parse_ratio_list(args.eval_seeds)],
            results_dir=args.results_dir,
            strategies=args.strategies,
            include_histories=args.include_histories,
        )


if __name__ == '__main__':
    main()
