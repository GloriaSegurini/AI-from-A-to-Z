# AI-from-A-to-Z

Content of the course *AI from A to Z*. You can find the course here:
https://www.udemy.com/share/101Wpy3@diqw5ALSFt7VcxYG1phbXHlON3lKPyc0kpW_wFJuLzFznUSQupUV8W_CoLmRfzc6oQ==/

## Deep Q-Learning for LunarLander

This repository documents my study path on **Reinforcement Learning**, moving from the theoretical foundations of **Markov Decision Processes** and **Q-Learning** toward a first implementation of **Deep Q-Learning (DQN)** applied to the `LunarLander-v2` environment from Gymnasium.

At this stage, the focus is not only on running the code, but on understanding the role of each component in the learning pipeline: the neural network that approximates the Q-function, the replay buffer that stores experience, the agent that interacts with the environment, and the logic that determines when learning actually happens.

---

## Project Overview

The objective of this phase of the course is to train an agent to land a spacecraft safely on the moon.

The environment used is `LunarLander-v2`, where the agent must learn how to choose among four discrete actions:

- do nothing
- fire the left orientation engine
- fire the main engine
- fire the right orientation engine

At each time step, the agent receives an 8-dimensional state vector describing the current condition of the lander. The Deep Q-Network takes this state as input and outputs one Q-value for each action, representing the estimated long-term utility of choosing that action in that state.

---

## Learning Goals

The purpose of this repository is to consolidate the following concepts:

- how a reinforcement learning problem is formalized as interaction between an agent and an environment
- what a **Q-function** represents in terms of expected future discounted reward
- why Deep Q-Learning is needed when the state space is too large or continuous for tabular Q-Learning
- how a neural network can approximate `Q(s, a)` by mapping one state to multiple action-values
- how **experience replay** stabilizes training
- why DQN uses both a **local network** and a **target network**
- how the code translates these ideas into a working training loop

---

## Environment: LunarLander-v2

The environment is created with Gymnasium:

```python
env = gym.make('LunarLander-v2')
```

From it, the code extracts three key parameters:

- `state_shape`
- `state_size`
- `number_actions`

For this environment:

- `state_shape = (8,)`
- `state_size = 8`
- `number_actions = 4`

The 8 entries of the state vector represent:

1. horizontal position `x`
2. vertical position `y`
3. horizontal velocity `vx`
4. vertical velocity `vy`
5. angle of the lander
6. angular velocity
7. left leg ground contact
8. right leg ground contact

These values provide the information the agent needs to decide how to correct descent, orientation, and contact with the landing area.

---

## Neural Network Architecture

The class `Network` defines the function approximator used in the DQN.

It inherits from `nn.Module` and contains:

- one input layer of size `state_size`
- two hidden fully connected layers of 64 neurons each
- one output layer of size `action_size`

In code, this corresponds to:

```python
self.fc1 = nn.Linear(state_size, 64)
self.fc2 = nn.Linear(64, 64)
self.fc3 = nn.Linear(64, action_size)
```

The network receives one state as input and returns four Q-values:

```text
[Q(s,a0), Q(s,a1), Q(s,a2), Q(s,a3)]
```

This means that the network does not output a single value for the state alone, but one value for each possible action in that state.

The `forward` method propagates the input through the two hidden layers with `ReLU` activations and finally returns the raw Q-values:

```python
x = self.fc1(state)
x = F.relu(x)
x = self.fc2(x)
x = F.relu(x)
return self.fc3(x)
```

No softmax is applied, because in DQN we are not predicting probabilities. We are predicting action-values.

---

## Hyperparameters

The code initializes the main hyperparameters used in training:

```python
learning_rate = 5e-4
minibatch_size = 100
discount_factor = 0.99
replay_buffer_size = int(1e5)
interpolation_parameter = 1e-3
```

Their roles are the following:

- `learning_rate`: controls how large each update step is during optimization
- `minibatch_size`: number of sampled experiences used in one learning update
- `discount_factor`: determines how much future rewards matter relative to immediate rewards
- `replay_buffer_size`: maximum number of experiences stored in memory
- `interpolation_parameter`: soft update coefficient used to gradually update the target network

These values are not derived from a strict universal rule. In practice, they are usually chosen through experimentation and prior empirical knowledge.

---

## Replay Memory

The class `ReplayMemory` implements **experience replay**.

Instead of training on the most recent transition only, the agent stores past experiences in a memory buffer and later samples random minibatches from it. This helps reduce temporal correlation between consecutive observations and makes training more stable.

Each stored experience has the form:

```text
(state, action, reward, next_state, done)
```

The class contains three main elements:

- `capacity`: maximum number of experiences the buffer can hold
- `memory`: the internal list that stores transitions
- `device`: CPU or GPU where sampled tensors will be moved

### `push(event)`

This method appends a new experience to memory. If the buffer exceeds its maximum capacity, the oldest experience is removed.

### `sample(batch_size)`

This method randomly samples a minibatch of experiences and separates them into:

- `states`
- `actions`
- `rewards`
- `next_states`
- `dones`

These are then converted into PyTorch tensors so they can be used directly during learning.

---

## Agent Structure

The class `Agent` is the main DQN agent.

Its constructor initializes:

- the device (`CPU` or `GPU`)
- `state_size`
- `action_size`
- `local_qnetwork`
- `target_qnetwork`
- `optimizer`
- `memory`
- `t_step`

Two separate Q-networks are used:

- `local_qnetwork`: the network that is directly optimized through gradient descent
- `target_qnetwork`: the network used to compute more stable target values during learning

Both networks have the same architecture, but they play different roles during training.

The optimizer used is Adam:

```python
self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr = learning_rate)
```

This optimizer updates the weights of the local Q-network based on the loss computed during training.

---

## Step Logic

The `step(...)` method of the agent performs two tasks:

1. store the newly observed experience in replay memory
2. decide when to trigger a learning update

The experience is first pushed into memory:

```python
self.memory.push((state, action, reward, next_state, done))
```

Then the agent updates a time step counter:

```python
self.t_step = (self.t_step + 1) % 4
```

This means the agent checks whether it should learn every 4 environment steps.

It is important to distinguish:

- `t_step` controls **when** learning is attempted
- `minibatch_size` controls **how many** experiences are used when learning occurs

So the logic is not "learn every 100 samples". Instead, it is:

- keep collecting experiences at every step
- every 4 steps, check whether memory is large enough
- if memory contains more than `minibatch_size` experiences, sample a minibatch and call `learn(...)`

This gives a clearer separation between data collection frequency and update frequency.

---

## Conceptual Understanding Reached So Far

Up to the currently commented part of the code, the most important conceptual clarifications are:

- the network outputs Q-values for actions, not probabilities
- a Q-value refers to a state-action pair `(s, a)`, not to the state alone
- the replay buffer stores real transitions from interaction, not outputs of the neural network
- minibatches are made of many different experiences, not repeated evaluations of the same state
- the learning target is built from reward plus discounted future value
- the local network and target network share the same architecture, but not the same training role

In other words, the implementation already shows how DQN combines:

- function approximation with neural networks
- data reuse through replay memory
- more stable target construction through a second network

Even before the full `learn()` method is analyzed, these pieces already define the core logic of the algorithm.

---

## Repository Status

Current progress in the code walkthrough:

- environment setup completed
- neural network architecture commented and explained
- forward propagation logic commented and explained
- hyperparameters commented and explained
- replay memory class commented and explained
- agent constructor commented and explained
- `step(...)` method commented and explained

Next step in the study path:

- action selection with `act(...)`
- then the actual learning procedure inside `learn(...)`

---

## References

- Gymnasium documentation
- PyTorch documentation
- Course: *AI from A to Z*
