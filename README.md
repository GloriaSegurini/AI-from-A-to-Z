# AI-from-A-to-Z

Content of the course *AI from A to Z*. You can find the course here:
https://www.udemy.com/share/101Wpy3@diqw5ALSFt7VcxYG1phbXHlON3lKPyc0kpW_wFJuLzFznUSQupUV8W_CoLmRfzc6oQ==/

## Q-Learning for Process Optimization

This repository documents my study path on **Reinforcement Learning**, with a focus on the part of the course that introduces **Markov Decision Processes (MDPs)** and **Q-Learning**, up to the first practical implementation applied to **warehouse flow optimization**.

The goal of this repository is not only to provide code, but also to collect the **theoretical foundations** behind the model: what problem is being solved, how the environment is formalized, why the Q-values are updated in a certain way, and how the final policy emerges from experience.

---

### Table of Contents

- [Project Overview](#project-overview)
- [Learning Goals](#learning-goals)
- [From Decision Making to Reinforcement Learning](#from-decision-making-to-reinforcement-learning)
- [Markov Decision Processes](#markov-decision-processes)
  - [States](#states)
  - [Actions](#actions)
  - [Transition Rule](#transition-rule)
  - [Reward Function](#reward-function)
  - [Markov Assumption](#markov-assumption)
  - [Policy and Optimal Policy](#policy-and-optimal-policy)
- [Future Cumulative Reward](#future-cumulative-reward)
  - [Return](#return)
  - [Discount Factor](#discount-factor)
- [Q-Learning](#q-learning)
  - [What the Q-Value Represents](#what-the-q-value-represents)
  - [Temporal Difference](#temporal-difference)
  - [Bellman-Style Update Rule](#bellman-style-update-rule)
  - [Action Selection](#action-selection)
- [Case Study: Warehouse Route Optimization](#case-study-warehouse-route-optimization)
  - [Business Problem](#business-problem)
  - [Environment Encoding](#environment-encoding)
  - [Reward Matrix](#reward-matrix)
  - [Why the Method Works](#why-the-method-works)
- [Implementation Logic](#implementation-logic)
  - [Training Phase](#training-phase)
  - [Route Extraction](#route-extraction)
  - [Intermediary Location](#intermediary-location)
- [Key Concepts Learned](#key-concepts-learned)
- [Notes on Theory](#notes-on-theory)
- [Repository Status](#repository-status)
- [References](#references)

---

### Project Overview

In this phase of the course, the objective is to understand how an agent can learn to make decisions in an environment by interacting with it and receiving rewards.

The main algorithm studied so far is **Q-Learning**, a model-free Reinforcement Learning algorithm that learns how good it is to perform an action in a given state.

The practical application used in the course is a warehouse optimization problem: an autonomous robot must move through different locations and learn the best path to a target location, optionally passing through an intermediate priority point.

---

### Learning Goals

The purpose of this repository is to consolidate the following concepts:

- how to formalize a sequential decision problem as an **MDP**
- how to define **states**, **actions**, **transitions**, and **rewards**
- how to model the notion of **future cumulative reward**
- how **Q-values** encode expected long-term utility
- how the **Temporal Difference** drives learning
- how the **Q-Learning update rule** progressively improves decisions
- how these ideas are translated into code for a concrete optimization problem

---

### From Decision Making to Reinforcement Learning

In Reinforcement Learning, an agent is not directly told what the best action is. Instead, it interacts with an environment, receives rewards, and gradually learns which behaviors lead to better long-term outcomes.

This is different from supervised learning:

- in supervised learning, the correct output is given
- in reinforcement learning, the agent must discover a good strategy through trial and error

The central challenge is therefore:

> Given the current state of the environment, which action should the agent choose in order to maximize long-term reward?

This question is formalized through **Markov Decision Processes**.

---

### Markov Decision Processes

A **Markov Decision Process (MDP)** is defined as a tuple:

$$
(S, A, T, R)
$$

where:

- $S$ is the set of states
- $A$ is the set of actions
- $T$ is the transition rule
- $R$ is the reward function

#### States

A **state** is a representation of the current situation of the environment.

In general, a state can be:

- a vector of encoded values
- an image
- any representation that captures the information relevant for decision making

In the warehouse case study, a state is simply the robot's **current location**. The locations are encoded as integers:

$$
S = \{0,1,2,\dots,11\}
$$

with the mapping:

- A -> 0
- B -> 1
- C -> 2
- ...
- L -> 11

#### Actions

An **action** is a choice available to the agent in a given state.

In the warehouse example, actions are encoded using the same integer labels as the locations. Conceptually, choosing an action means deciding the **next location** to move to, provided that move is allowed by the warehouse connectivity graph.

#### Transition Rule

The transition rule describes the probability of moving to a future state given the current state and chosen action:

$$
T : (s_t, a_t, s_{t+1}) \mapsto P(s_{t+1} \mid s_t, a_t)
$$

This tells us how likely it is to end up in state $s_{t+1}$ after taking action $a_t$ in state $s_t$.

In general RL settings, transitions may be stochastic. In the warehouse case, the code samples a valid next state among the playable actions during training, which acts as the interaction mechanism used to learn the Q-values.

#### Reward Function

The reward function assigns a numerical value to taking an action in a state:

$$
R : (s_t, a_t) \mapsto r_t
$$

where $r_t$ is the immediate reward obtained after choosing action $a_t$ in state $s_t$.

In the warehouse implementation, rewards are encoded as a **matrix**:

- rows = current states
- columns = actions / next locations
- cell values = rewards

At first, the matrix mainly contains:

- `1` for allowed moves
- `0` for invalid moves

Then a large reward is assigned to the target location, so that learning converges toward routes that eventually reach it.

#### Markov Assumption

The defining assumption of an MDP is that the future depends only on the current state and current action, not on the full history:

$$
P(s_{t+1} \mid s_0, a_0, s_1, a_1, \dots, s_t, a_t)
=
P(s_{t+1} \mid s_t, a_t)
$$

This is the **Markov property**.

So the environment is treated as "memoryless" in the sense that everything relevant for the next decision is assumed to be contained in the current state.

#### Policy and Optimal Policy

A **policy** is a rule that tells the agent what action to take in each state:

$$
\pi : S \mapsto A
$$

The objective is to find an **optimal policy** $\pi^*$, that is, a policy that maximizes cumulative reward over time.

---

### Future Cumulative Reward

#### Return

A central idea in Reinforcement Learning is that we do not care only about the immediate reward $r_t$, but about the **return**, meaning the cumulative future reward from time $t$ onward.

Without discounting:

$$
R_t = r_t + r_{t+1} + r_{t+2} + \dots + r_n
$$

This expresses the idea that a good action is not just one that gives a reward now, but one that leads to a good trajectory overall.

#### Discount Factor

Future rewards are usually discounted because rewards far in the future are more uncertain and should contribute less than immediate ones.

The discounted return is:

$$
R_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots + \gamma^{n-t} r_n
$$

where:

$$
\gamma \in [0,1]
$$

is the **discount factor**.

Interpretation:

- if $\gamma \approx 0$, the agent is short-sighted and prioritizes immediate reward
- if $\gamma \approx 1$, the agent values long-term reward more strongly

This notion is introduced before Q-Learning because the Q-value is fundamentally about expected discounted future return.

---

### Q-Learning

Q-Learning is the core method studied in this phase.

Its purpose is to learn, for every state-action pair, how good that decision is in terms of long-term reward.

#### What the Q-Value Represents

To each pair $(s,a)$, we associate a value:

$$
Q(s,a)
$$

Intuitively, $Q(s,a)$ measures the expected usefulness of taking action $a$ in state $s$, and then continuing in a way that favors high future return.

A precise way to think about it is:

$$
Q^*(s,a)
$$

which is the optimal expected return obtained by:

1. starting from state $s$
2. taking action $a$
3. then behaving optimally from the next state onward

That is the key reason why Q-values are so important: once they are known, the agent can act by choosing the action with the highest Q-value.

#### Temporal Difference

The **Temporal Difference (TD)** is the signal used to update Q-values.

At time $t$, after taking action $a_t$ in state $s_t$, receiving reward $r_t$, and reaching $s_{t+1}$, the TD is:

$$
TD_t(s_t, a_t) =
r_t + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t)
$$

This compares:

- the **current estimate** $Q(s_t,a_t)$
- with a **better-informed target** based on:
  - the reward just received
  - the best future Q-value available from the next state

Interpretation:

- if TD is positive, the outcome was better than expected
- if TD is negative, the outcome was worse than expected
- if TD is zero, the estimate was already consistent with the observed transition

#### Bellman-Style Update Rule

The Q-value is updated using the Temporal Difference:

$$
Q_t(s_t, a_t) = Q_{t-1}(s_t, a_t) + \alpha \, TD_t(s_t, a_t)
$$

equivalently:

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) +
\alpha \left[r_t + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t)\right]
$$

where:

- $\alpha$ is the **learning rate**
- $\gamma$ is the **discount factor**

Interpretation of $\alpha$:

- small $\alpha$: slower but smoother learning
- large $\alpha$: faster but more aggressive updates

In the implementation used in the course:

$$
\gamma = 0.75
\qquad
\alpha = 0.9
$$

#### Action Selection

Once Q-values start to encode useful information, the agent needs a rule to choose actions.

A basic choice is **argmax**:

$$
a_t = \arg\max_a Q(s_t, a)
$$

meaning: choose the action with the highest Q-value in the current state.

The course also introduces **softmax** action selection, where actions are sampled from a probability distribution derived from the Q-values, rather than deterministically choosing the maximum. This is presented as a more exploration-friendly alternative in general, although the warehouse case study ends up using the argmax logic for extracting the final route.

---

### Case Study: Warehouse Route Optimization

#### Business Problem

The practical problem is to optimize the movement of an autonomous robot inside an e-commerce warehouse with 12 locations, from **A** to **L**.

A system ranks product pickup priorities in real time. The robot must:

- reach the highest-priority location
- ideally do so through the shortest or most favorable route
- optionally pass through an intermediary high-priority location

#### Environment Encoding

The environment is encoded as a graph:

- nodes = warehouse locations
- allowed moves = edges between locations
- each location is mapped to an integer state

The code uses:

- `location_to_state` to map letters to integers
- `state_to_location` to map integers back to letters
- a reward matrix `R` to encode which moves are possible

This makes the environment easy to handle numerically with NumPy matrices.

#### Reward Matrix

The reward matrix has shape $12 \times 12$.

For each pair $(s,a)$:

- `1` means the move is allowed
- `0` means the move is not allowed

Then, when a destination is chosen, the implementation creates a copy of the reward matrix and sets a large reward on the destination state:

$$
R_{\text{new}}[s_{\text{goal}}, s_{\text{goal}}] = 1000
$$

This high reward acts as a strong attractor during learning, causing the Q-values to propagate useful information backward through the graph.

#### Why the Method Works

Even though the robot is not explicitly programmed with shortest paths, repeated Q-updates make actions that lead efficiently toward the goal accumulate higher values.

So the agent is not told:

> "from E go to I, then J, then K..."

Instead, it discovers through repeated updates that certain transitions are more valuable because they lead, directly or indirectly, to the high-reward goal.

That is the essence of reinforcement learning:

- do not hard-code the route
- define the environment and the incentives
- let value propagation reveal the good decisions

---

### Implementation Logic

#### Training Phase

The training procedure used in the implementation follows these steps:

1. initialize the Q-matrix with zeros
2. repeatedly sample a random current state
3. collect the playable actions from that state
4. randomly choose one of those valid actions
5. compute the Temporal Difference
6. update the corresponding Q-value

In code, this is implemented with a loop of 1000 iterations for each route computation.

#### Route Extraction

Once the Q-values are learned, the route is extracted greedily:

- start from the initial location
- convert it to its state index
- pick the action with the maximum Q-value on that row
- move to the corresponding next location
- repeat until the destination is reached

So the learned Q-table becomes a compact representation of routing knowledge.

#### Intermediary Location

The implementation also introduces a `best_route(starting_location, intermediary_location, ending_location)` function.

Its logic is simple:

- compute the route from start to intermediary
- compute the route from intermediary to destination
- concatenate the two paths, avoiding duplication of the intermediary node

This allows the robot to pass through a second priority location before reaching the final top-priority target.

---

### Key Concepts Learned

Through this project, the following ideas become concrete:

#### 1. A reward function defines behavior

The agent does not need to be manually told what path to follow. Its behavior emerges from the reward structure.

#### 2. Q-values encode long-term utility

A high Q-value does not simply mean "good immediate reward". It means "good decision when future consequences are also considered".

#### 3. The next state matters because of bootstrapping

The update rule uses:

$$
\max_a Q(s_{t+1}, a)
$$

which means the estimate of the present action depends on the estimated value of future actions. This is what makes Q-Learning a **bootstrapping** method.

#### 4. Learning happens from iterative correction

The Temporal Difference is essentially an error signal: the model adjusts its old estimate toward a target built from new experience.

#### 5. Optimal behavior emerges from local updates

The final route is not stored as an explicit plan from the start. It emerges because the local state-action values become progressively more accurate.

---

### Notes on Theory

A useful precise interpretation is the following:

- $Q^*(s,a)$ = optimal return obtained by taking action $a$ in state $s$, and then following the optimal policy afterward
- $V^*(s)$ = best possible value achievable from state $s$

These are related by:

$$
V^*(s) = \max_a Q^*(s,a)
$$

and the optimal policy can be written as:

$$
\pi^*(s) = \arg\max_a Q^*(s,a)
$$

This makes the logic very compact:

1. learn how good every action is in every state
2. pick the action with the largest learned value
3. that induced greedy policy approximates the optimal one

---

### Repository Status

At this stage, this repository covers:

- Markov Decision Processes
- future cumulative reward and discounting
- Q-values
- Temporal Difference
- Q-Learning update rule
- action selection via argmax / softmax at a conceptual level
- practical Q-Learning implementation for warehouse flow optimization

The next natural extensions, once the study path progresses further, will be:

- Deep Q-Learning
- experience replay
- convolutional approaches for image-based states
- more advanced reinforcement learning architectures

---

### References

- Course: *AI from A to Z* on Udemy
- Local implementation in this repository: [q_learning.py](./q_learning.py)
