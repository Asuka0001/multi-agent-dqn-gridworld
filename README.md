# Cooperative Multi-Agent Deep Q-Learning

Four agents of two types learn, with Deep Q-Networks, to cooperate in a grid world. Two agents of
opposite types must meet to exchange a key, then one of them carries it to a goal.
The repository covers training, comparison against baselines, and **exhaustive evaluation over
every possible start state** against a provable lower bound on episode length.

## Task

- 5×5 grid, one goal cell, two type-A and two type-B agents, all spawned on distinct cells.
- An agent obtains the key when it shares a cell (not the goal) with an agent of the opposite type.
- The episode ends when an agent holding the key reaches the goal.
- In each round every agent moves once (up / down / left / right / wait), **in a random order**.
- Each agent observes 7 values: its position, the offset to the goal, the offset to the closest
  opposite-type agent, and whether it holds the key.
- The two agents of the same type share one Q-network.

## Method

| Component | Choice |
|---|---|
| Learner | DQN per agent type, Double DQN targets, Huber loss, target network, gradient clipping |
| Experience | Persistent replay buffer per type (100k), 32 games run in parallel |
| Multi-agent transitions | An agent's next state is observed at its *next* turn, after the others have moved. Rewards caused by other agents (being met) are credited to the agent's pending transition |
| Rewards | −1 per move, +10 on obtaining the key, +50 on delivery; time-outs are treated as truncation (bootstrapped), not as terminal |
| Exploration | ε linearly 1.0 → 0.05 over the first half of training |

### Evaluation

- **Exhaustive test set.** Every start state on the 5×5 grid, 1,593,900 in total
  (25·24·23·22·21, divided by 4 because agents of the same type are interchangeable).
  The greedy policy is compiled into a lookup table over all 31,250 observations, so all games
  are simulated at once in about 5 s.
- **Lower bound.** For any meeting cell *m* ≠ goal between an A-agent and a B-agent, no policy can
  finish in fewer than `max(d_A, d_B) + d(m, goal) − [d_A ≠ d_B]` rounds. The earlier agent can
  wait at *m* and leave in the same round the other one arrives. The minimum over *m* and
  agent pairs gives a bound for each start state, and *excess* is reported as rounds above it.
- **Baselines.** A uniformly random policy, and a hand-written rule: walk to the closest
  opposite-type agent, then walk to the goal.
- **3 training seeds**, reported as mean ± std.

## Results

All 1,593,900 start states, episodes capped at 50 rounds:

| Policy | Success (≤ 15 rounds) | Mean rounds | Excess over lower bound | Optimal (= bound) |
|---|---|---|---|---|
| Random | 29.5% | 22.43 | 19.93 | 0.6% |
| Rule-based | 98.55% | 3.55 | 0.98 | 31.4% |
| **DQN (3 seeds)** | **99.96% ± 0.02%** | **3.26 ± 0.04** | **0.70 ± 0.04** | **41.3% ± 1.8%** |

Each seed trains in about 30 s on one CPU thread (15,000 rounds × 32 games, about 108k episodes).
Validation success first exceeded 98% after 5k–15k episodes, depending on the seed.

### Ablations (3 seeds each, same exhaustive test)

| Variant | Success (≤ 15) | Excess over bound |
|---|---|---|
| Full model | 99.96% | 0.70 ± 0.04 |
| No key-exchange reward (`--meet-reward 0`) | 99.997% | 0.80 ± 0.03 |
| Vanilla DQN targets (`--no-double`) | 99.98% | 0.72 ± 0.03 |
| Absolute coordinates (`--abs-obs`) | 99.99% | 0.69 ± 0.01 |

The key-exchange reward shortens paths by about 12%. On a 5×5 grid, Double DQN and relative
coordinates make no measurable difference.

### Limitations

- About 0.04% of start states still hit the 50-round cap, where the agents get stuck in a loop.
- The rule-based baseline is deliberately simple. The task is small enough that near-optimal
  hand-written strategies exist, so larger grids or obstacles would separate learned and
  scripted policies more clearly.

## Usage

```bash
pip install -r requirements.txt

python marl_dqn.py                               # 3 seeds, exhaustive evaluation (~2 min)
python marl_dqn.py --seeds 0 --rounds 4000       # quick run
python marl_dqn.py --no-double --out results/ablations/no_double   # ablation
python marl_dqn.py --size 8 --sample-eval 200000 # larger grid, sampled evaluation
```

Outputs go to `results/`: `results.json` (config, baseline and per-seed test metrics, validation
history) and the trained Q-network weights `dqn_seed{N}_type{A,B}.pt`.

## Repository layout

```
marl_dqn.py              environment, DQN, training, baselines, lower bound, evaluation
results/results.json     main run (3 seeds)
results/train.log        console output of the main run
results/*.pt             trained Q-networks
results/ablations/       logs and results.json of each ablation
```

## Background

This started as a reinforcement-learning assignment at Monash University. The code here is an
independent rewrite that corrects the learning setup and adds the evaluation framework above.
