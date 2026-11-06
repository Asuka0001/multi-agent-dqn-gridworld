"""
Cooperative multi-agent DQN on a grid world.

Task
----
n x n grid, one goal cell G, two agents of type A and two of type B, all spawned on
distinct cells. An agent obtains the full secret when it shares a cell (other than G)
with an agent of the opposite type. The episode ends when an agent holding the secret
steps onto G. In every round each of the four agents moves once, in a random order.

Observation (the "closest opposite-type agent" contract, 7 numbers):
    own position, goal offset, closest-opposite-agent offset, has_secret
Each agent type has its own DQN shared by the two agents of that type.

Design notes
------------
* Correct multi-agent transitions: an agent's next state is observed at its *next* turn
  (after the other agents moved), and rewards caused by others (being met) are credited
  to the pending transition.
* Persistent per-type replay buffers, Double DQN, Huber loss, target network,
  timeouts treated as truncation (bootstrapped) rather than terminal.
* Vectorised environments for training (n_envs games in parallel).
* Evaluation against baselines (random, hand-written heuristic) and against a provable
  lower bound on the number of rounds (best-case move order).
* Exhaustive evaluation over every start configuration (1,593,900 on a 5x5 grid)
  by compiling the greedy policy into a lookup table and simulating all games at once.
* Multiple seeds, results saved as JSON.

Usage
-----
    python marl_dqn.py                          # 3 seeds, full run
    python marl_dqn.py --seeds 0 --rounds 4000  # quick run
    python marl_dqn.py --no-double --abs-obs    # ablations
"""
import argparse
import itertools
import json
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TYPE = np.array([0, 0, 1, 1])          # agents 0,1 are type A; agents 2,3 are type B
OPP_IDS = np.array([[2, 3], [0, 1]])   # opposite agents, indexed by type
MOVES = np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]])  # up, down, left, right, wait
TYPE_NAMES = ("A", "B")


@dataclass
class Config:
    size: int = 5
    wait_action: bool = True
    max_rounds: int = 50          # episode cap for training and evaluation
    success_rounds: int = 15      # "success" = solved within this many rounds
    step_reward: float = -1.0
    meet_reward: float = 10.0
    goal_reward: float = 50.0
    gamma: float = 0.95
    lr: float = 5e-4
    hidden: int = 128
    n_envs: int = 32
    train_rounds: int = 15000
    batch_size: int = 256
    buffer_size: int = 100_000
    learn_start: int = 2000       # transitions per type before learning starts
    target_update: int = 500      # gradient steps between target-network syncs
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_frac: float = 0.5         # fraction of training over which epsilon decays
    double_dqn: bool = True
    relative_obs: bool = True
    eval_every: int = 500
    n_val: int = 5000

    @property
    def n_actions(self):
        return 5 if self.wait_action else 4


# ----------------------------------------------------------------------------
# Environment (vectorised over many games)
# ----------------------------------------------------------------------------

def sample_configs(n_games, size, rng):
    """Random start states: goal and 4 agents on distinct cells. Returns goal (N,2), pos (N,4,2)."""
    cells = np.argsort(rng.random((n_games, size * size)), axis=1)[:, :5]
    xy = np.stack([cells // size, cells % size], axis=-1)
    return xy[:, 0].copy(), xy[:, 1:].copy()


def all_configs(size):
    """Every start state up to swapping the two agents of the same type."""
    C = size * size
    rows = []
    for g in range(C):
        rest = [c for c in range(C) if c != g]
        for a1, a2 in itertools.combinations(rest, 2):
            rest2 = [c for c in rest if c != a1 and c != a2]
            bp = np.array(list(itertools.combinations(rest2, 2)))
            block = np.empty((len(bp), 5), dtype=np.int64)
            block[:, 0], block[:, 1], block[:, 2], block[:, 3:] = g, a1, a2, bp
            rows.append(block)
    cells = np.concatenate(rows)
    xy = np.stack([cells // size, cells % size], axis=-1)
    return xy[:, 0].copy(), xy[:, 1:].copy()


def closest_opp(pos, ids):
    """Position of the closest opposite-type agent (Manhattan, ties -> lower index)."""
    r = np.arange(len(ids))
    me = pos[r, ids]
    cand = pos[r[:, None], OPP_IDS[TYPE[ids]]]          # (M,2,2)
    d = np.abs(cand - me[:, None]).sum(-1)
    pick = (d[:, 1] < d[:, 0]).astype(np.int64)
    return cand[r, pick]


def obs_index(pos, goal, secret, rows, ids, size):
    """Encode an agent's observation as a single integer (used for tables and the buffer)."""
    C = size * size
    me = pos[rows, ids]
    opp = closest_opp(pos[rows], ids)
    g = goal[rows]
    cell = lambda p: p[:, 0] * size + p[:, 1]
    return ((cell(me) * C + cell(g)) * C + cell(opp)) * 2 + secret[rows, ids].astype(np.int64)


def decode_obs(size):
    """All observation indices decoded into (me, goal, opp) xy arrays and secret flag."""
    C = size * size
    idx = np.arange(C * C * C * 2)
    s = idx % 2
    rest = idx // 2
    o, rest = rest % C, rest // C
    g, m = rest % C, rest // C
    xy = lambda c: np.stack([c // size, c % size], axis=-1)
    return xy(m), xy(g), xy(o), s


def build_features(size, relative):
    me, g, o, s = decode_obs(size)
    k = float(size - 1)
    if relative:
        f = np.concatenate([me / k, (g - me) / k, (o - me) / k, s[:, None]], axis=1)
    else:
        f = np.concatenate([me / k, g / k, o / k, s[:, None]], axis=1)
    return torch.tensor(f, dtype=torch.float32)


def take_turn(pos, goal, secret, rows, ids, actions, size):
    """Move one agent in each of the given games and resolve meetings / delivery."""
    newp = np.clip(pos[rows, ids] + MOVES[actions], 0, size - 1)
    pos[rows, ids] = newp
    at_goal = (newp == goal[rows]).all(-1)
    same = (pos[rows] == newp[:, None]).all(-1)                        # (M,4)
    meet = same & (TYPE[None, :] != TYPE[ids][:, None]) & ~at_goal[:, None]
    any_meet = meet.any(1)
    gained_self = any_meet & ~secret[rows, ids]
    gained_other = meet & ~secret[rows]
    secret[rows, ids] |= any_meet
    secret[rows] = secret[rows] | meet
    delivered = at_goal & secret[rows, ids]
    return gained_self, gained_other, delivered


def simulate(goal, pos, policy, cfg, rng):
    """Run games to completion. policy(obs_idx, types, rng) -> actions.
    Returns rounds needed per game (cfg.max_rounds + 1 if unsolved)."""
    goal, pos = goal.copy(), pos.copy()
    N = len(goal)
    secret = np.zeros((N, 4), dtype=bool)
    steps = np.full(N, cfg.max_rounds + 1)
    active = np.arange(N)
    for r in range(1, cfg.max_rounds + 1):
        order = np.argsort(rng.random((len(active), 4)), axis=1)
        alive = np.ones(len(active), dtype=bool)
        for k in range(4):
            loc = np.nonzero(alive)[0]
            rows, ids = active[loc], order[loc, k]
            oi = obs_index(pos, goal, secret, rows, ids, cfg.size)
            a = policy(oi, TYPE[ids], rng)
            _, _, delivered = take_turn(pos, goal, secret, rows, ids, a, cfg.size)
            steps[rows[delivered]] = r
            alive[loc[delivered]] = False
        active = active[alive]
        if len(active) == 0:
            break
    return steps


def lower_bound(goal, pos, size, chunk=200_000):
    """Fewest rounds any policy could need, assuming the most favourable move order.
    For a meeting cell m != G between A-agent a and B-agent b:
        max(d_a, d_b) + d(m, G) - [d_a != d_b]
    (the earlier arrival can wait at m and move on in the same round the other arrives)."""
    cells = np.stack(np.divmod(np.arange(size * size), size), axis=-1)
    out = np.empty(len(goal), dtype=np.int64)
    for s in range(0, len(goal), chunk):
        g, p = goal[s:s + chunk], pos[s:s + chunk]
        dmg = np.abs(cells[None] - g[:, None]).sum(-1)
        dmg = np.where(dmg == 0, 10_000, dmg)               # no trading on the goal
        best = np.full(len(g), 10_000)
        for a in (0, 1):
            da = np.abs(cells[None] - p[:, a][:, None]).sum(-1)
            for b in (2, 3):
                db = np.abs(cells[None] - p[:, b][:, None]).sum(-1)
                lb = np.maximum(da, db) + dmg - (da != db)
                best = np.minimum(best, lb.min(1))
        out[s:s + chunk] = best
    return out


# ----------------------------------------------------------------------------
# Policies
# ----------------------------------------------------------------------------

def random_policy(cfg):
    return lambda oi, types, rng: rng.integers(cfg.n_actions, size=len(oi))


def heuristic_table(cfg):
    """Walk to the closest opposite agent; once holding the secret walk to the goal."""
    me, g, o, s = decode_obs(cfg.size)
    target = np.where(s[:, None] == 1, g, o)
    d = target - me
    act = np.where(d[:, 0] < 0, 0, np.where(d[:, 0] > 0, 1,
                   np.where(d[:, 1] < 0, 2, np.where(d[:, 1] > 0, 3, 4))))
    if not cfg.wait_action:
        act = np.where(act == 4, 0, act)
    return np.stack([act, act])                      # same rule for both types


def table_policy(table):
    return lambda oi, types, rng: table[types, oi]


# ----------------------------------------------------------------------------
# DQN
# ----------------------------------------------------------------------------

class QNet(nn.Module):
    def __init__(self, n_in, hidden, n_out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_out))

    def forward(self, x):
        return self.net(x)


class Learner:
    """One DQN + replay buffer per agent type. Observations are stored as table indices."""

    def __init__(self, cfg, feats, rng):
        self.cfg, self.feats, self.rng = cfg, feats, rng
        self.q = QNet(feats.shape[1], cfg.hidden, cfg.n_actions)
        self.q_targ = QNet(feats.shape[1], cfg.hidden, cfg.n_actions)
        self.q_targ.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        n = cfg.buffer_size
        self.s = np.zeros(n, np.int64)
        self.a = np.zeros(n, np.int64)
        self.r = np.zeros(n, np.float32)
        self.s2 = np.zeros(n, np.int64)
        self.d = np.zeros(n, np.float32)
        self.ptr = self.size = self.updates = 0

    def add(self, s, a, r, s2, d):
        for i in range(len(s)):
            j = self.ptr
            self.s[j], self.a[j], self.r[j], self.s2[j], self.d[j] = s[i], a[i], r[i], s2[i], d[i]
            self.ptr = (j + 1) % self.cfg.buffer_size
            self.size = min(self.size + 1, self.cfg.buffer_size)

    @torch.no_grad()
    def act(self, oi):
        return self.q(self.feats[oi]).argmax(1).numpy()

    @torch.no_grad()
    def table(self):
        return self.q(self.feats).argmax(1).numpy()

    def update(self):
        cfg = self.cfg
        if self.size < cfg.learn_start:
            return None
        j = self.rng.integers(self.size, size=cfg.batch_size)
        s, s2 = self.feats[self.s[j]], self.feats[self.s2[j]]
        a = torch.from_numpy(self.a[j])
        r, d = torch.from_numpy(self.r[j]), torch.from_numpy(self.d[j])
        with torch.no_grad():
            if cfg.double_dqn:
                a2 = self.q(s2).argmax(1, keepdim=True)
                qn = self.q_targ(s2).gather(1, a2).squeeze(1)
            else:
                qn = self.q_targ(s2).max(1).values
            y = r + cfg.gamma * (1 - d) * qn
        q = self.q(s).gather(1, a[:, None]).squeeze(1)
        loss = F.smooth_l1_loss(q, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        self.updates += 1
        if self.updates % cfg.target_update == 0:
            self.q_targ.load_state_dict(self.q.state_dict())
        return loss.item()


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------

def summarize(steps, lb, cfg):
    solved = steps <= cfg.max_rounds
    return {
        "n_games": int(len(steps)),
        "success_rate_15": float((steps <= cfg.success_rounds).mean()),
        "solve_rate": float(solved.mean()),
        "mean_rounds": float(steps[solved].mean()) if solved.any() else None,
        "mean_excess_over_lb": float((steps - lb)[solved].mean()) if solved.any() else None,
        "optimal_rate": float((steps == lb).mean()),
        "worst_rounds": int(steps.max()),
    }


def train(cfg, seed, feats, log_every=1000):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    learners = [Learner(cfg, feats, rng), Learner(cfg, feats, rng)]
    E, size = cfg.n_envs, cfg.size

    val_goal, val_pos = sample_configs(cfg.n_val, size, np.random.default_rng(10_000 + seed))
    val_lb = lower_bound(val_goal, val_pos, size)

    goal, pos = sample_configs(E, size, rng)
    secret = np.zeros((E, 4), bool)
    rounds = np.zeros(E, np.int64)
    pend = np.zeros((E, 4), bool)
    ps = np.zeros((E, 4), np.int64)
    pa = np.zeros((E, 4), np.int64)
    pr = np.zeros((E, 4), np.float32)

    def store(rows, ids, s2, done):
        for typ in (0, 1):
            m = TYPE[ids] == typ
            if m.any():
                r_, i_ = rows[m], ids[m]
                learners[typ].add(ps[r_, i_], pa[r_, i_], pr[r_, i_], s2[m],
                                  np.full(m.sum(), float(done)))
        pend[rows, ids] = False

    def flush_game(env_rows, done):
        """Close every pending transition of the given games."""
        if len(env_rows) == 0:
            return
        rr = np.repeat(env_rows, 4)
        ii = np.tile(np.arange(4), len(env_rows))
        m = pend[rr, ii]
        rr, ii = rr[m], ii[m]
        s2 = ps[rr, ii] if done else obs_index(pos, goal, secret, rr, ii, size)
        store(rr, ii, s2, done)

    history, episodes, ep_rounds = [], 0, []
    best, best_key, first_98 = None, None, None
    t0 = time.time()
    for t in range(cfg.train_rounds):
        eps = max(cfg.eps_end, cfg.eps_start - (cfg.eps_start - cfg.eps_end) * t / (cfg.eps_frac * cfg.train_rounds))
        order = np.argsort(rng.random((E, 4)), axis=1)
        alive = np.ones(E, bool)
        for k in range(4):
            rows = np.nonzero(alive)[0]
            ids = order[rows, k]
            oi = obs_index(pos, goal, secret, rows, ids, size)
            has = pend[rows, ids]
            if has.any():
                store(rows[has], ids[has], oi[has], False)
            a = rng.integers(cfg.n_actions, size=len(rows))
            greedy = rng.random(len(rows)) >= eps
            for typ in (0, 1):
                m = greedy & (TYPE[ids] == typ)
                if m.any():
                    a[m] = learners[typ].act(oi[m])
            ps[rows, ids], pa[rows, ids], pr[rows, ids] = oi, a, cfg.step_reward
            pend[rows, ids] = True
            gs, go, delivered = take_turn(pos, goal, secret, rows, ids, a, size)
            pr[rows, ids] += cfg.meet_reward * gs
            pr[rows] += cfg.meet_reward * go * pend[rows]
            pr[rows[delivered], ids[delivered]] += cfg.goal_reward
            fin = rows[delivered]
            flush_game(fin, done=True)
            alive[fin] = False
            ep_rounds.extend((rounds[fin] + 1).tolist())
        rounds += 1
        trunc = np.nonzero(alive & (rounds >= cfg.max_rounds))[0]
        flush_game(trunc, done=False)
        ep_rounds.extend([cfg.max_rounds + 1] * len(trunc))
        reset = np.concatenate([np.nonzero(~alive)[0], trunc])
        if len(reset):
            g2, p2 = sample_configs(len(reset), size, rng)
            goal[reset], pos[reset] = g2, p2
            secret[reset], rounds[reset] = False, 0
            episodes += len(reset)

        for L in learners:
            L.update()

        if (t + 1) % cfg.eval_every == 0 or t + 1 == cfg.train_rounds:
            table = np.stack([L.table() for L in learners])
            steps = simulate(val_goal, val_pos, table_policy(table), cfg, np.random.default_rng(1))
            s = summarize(steps, val_lb, cfg)
            s.update(round=t + 1, episodes=episodes, epsilon=round(eps, 3),
                     train_mean_rounds=float(np.mean(ep_rounds[-500:])) if ep_rounds else None)
            history.append(s)
            if first_98 is None and s["success_rate_15"] >= 0.98:
                first_98 = episodes
            key = (s["success_rate_15"], -(s["mean_excess_over_lb"] or 1e9))
            if best_key is None or key > best_key:
                best_key = key
                best = [{k: v.clone() for k, v in L.q.state_dict().items()} for L in learners]
            if (t + 1) % log_every == 0 or t + 1 == cfg.train_rounds:
                print(f"  seed {seed} round {t+1:>6} ep {episodes:>6} eps {eps:.2f} | "
                      f"val success15 {s['success_rate_15']:.3%} excess {s['mean_excess_over_lb']:.3f} "
                      f"| {time.time()-t0:.0f}s", flush=True)

    for L, sd in zip(learners, best):
        L.q.load_state_dict(sd)
    return learners, history, {"episodes": episodes, "episodes_to_98pct": first_98,
                               "train_seconds": time.time() - t0}


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--rounds", type=int, default=None, help="training rounds (each = 1 move of every agent in n_envs games)")
    p.add_argument("--size", type=int, default=None)
    p.add_argument("--no-double", action="store_true")
    p.add_argument("--abs-obs", action="store_true", help="absolute instead of relative coordinates")
    p.add_argument("--no-wait", action="store_true")
    p.add_argument("--meet-reward", type=float, default=None)
    p.add_argument("--sample-eval", type=int, default=0,
                   help="evaluate on N random starts instead of all start configurations")
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    args = p.parse_args()

    cfg = Config()
    if args.rounds: cfg.train_rounds = args.rounds
    if args.size: cfg.size = args.size
    if args.no_double: cfg.double_dqn = False
    if args.abs_obs: cfg.relative_obs = False
    if args.no_wait: cfg.wait_action = False
    if args.meet_reward is not None: cfg.meet_reward = args.meet_reward
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(1)

    feats = build_features(cfg.size, cfg.relative_obs)
    if args.sample_eval or cfg.size > 5:
        n = args.sample_eval or 200_000
        test_goal, test_pos = sample_configs(n, cfg.size, np.random.default_rng(999))
        test_name = f"{n} random start states"
    else:
        test_goal, test_pos = all_configs(cfg.size)
        test_name = f"all {len(test_goal):,} start states"
    test_lb = lower_bound(test_goal, test_pos, cfg.size)
    print(f"Test set: {test_name}; mean lower bound {test_lb.mean():.2f} rounds")

    results = {"config": asdict(cfg), "test_set": test_name, "baselines": {}, "dqn": []}
    for name, pol in [("random", random_policy(cfg)),
                      ("heuristic", table_policy(heuristic_table(cfg)))]:
        t0 = time.time()
        steps = simulate(test_goal, test_pos, pol, cfg, np.random.default_rng(0))
        results["baselines"][name] = summarize(steps, test_lb, cfg)
        print(f"{name:>9}: {results['baselines'][name]}  ({time.time()-t0:.0f}s)")

    for seed in args.seeds:
        print(f"Training seed {seed} ...")
        learners, history, info = train(cfg, seed, feats)
        table = np.stack([L.table() for L in learners])
        t0 = time.time()
        steps = simulate(test_goal, test_pos, table_policy(table), cfg, np.random.default_rng(0))
        res = summarize(steps, test_lb, cfg)
        res.update(seed=seed, eval_seconds=time.time() - t0, **info)
        print(f"      dqn: {res}")
        results["dqn"].append({"test": res, "history": history})
        for typ, L in enumerate(learners):
            torch.save(L.q.state_dict(), os.path.join(args.out, f"dqn_seed{seed}_type{TYPE_NAMES[typ]}.pt"))

    def ms(key):
        v = np.array([r["test"][key] for r in results["dqn"]], dtype=float)
        return f"{v.mean():.4g} ± {v.std():.2g}"

    print(f"\n=== Summary on {test_name} ===")
    print(f"{'policy':<12}{'success<=15':>16}{'mean rounds':>16}{'excess vs LB':>16}{'= LB':>16}")
    for name, r in results["baselines"].items():
        print(f"{name:<12}{r['success_rate_15']:>16.4f}{r['mean_rounds']:>16.3f}"
              f"{r['mean_excess_over_lb']:>16.3f}{r['optimal_rate']:>16.4f}")
    print(f"{'dqn':<12}{ms('success_rate_15'):>16}{ms('mean_rounds'):>16}"
          f"{ms('mean_excess_over_lb'):>16}{ms('optimal_rate'):>16}   ({len(args.seeds)} seeds)")
    path = os.path.join(args.out, "results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
