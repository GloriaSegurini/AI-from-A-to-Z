"""
Q-Learning for Process Optimization
Case Study: Optimizing the Flows in an E-Commerce Warehouse

Problem statement
-----------------
An autonomous warehouse robot moves inside an e-commerce warehouse composed
of 12 storage locations, labeled from A to L.

At any given time, the warehouse management system ranks these locations
according to product-picking priority. The robot's main objective is to
reach the current top-priority location by following the best possible route
from its current position.

In the extended version of the use case, the robot may also be asked to pass
through an intermediary high-priority location before reaching the final one.
However, in this first part we only focus on defining the environment needed
to train a Q-Learning agent.

Environment definition
----------------------
To model the problem, we define:
1. States   -> the robot's possible current locations
2. Actions  -> the possible next moves between locations
3. Rewards  -> a reward matrix encoding which moves are allowed

Important note about the rewards:
- A reward of 1 means that the move from one location to another is allowed.
- A reward of 0 means that the move is not allowed.
- A very high reward (e.g. 1000) is assigned to the top-priority location
  so that the agent learns to move toward it.

In this initial example, location G is considered the top-priority destination,
therefore its self-transition (G -> G) receives a reward of 1000.
"""

# ======================================================================
# IMPORTS
# ======================================================================

import numpy as np

# ======================================================================
# Q-LEARNING HYPERPARAMETERS
# ======================================================================

# Gamma (discount factor):
# controls how much future rewards matter compared to immediate rewards.
# A value close to 1 gives more importance to long-term reward.
gamma = 0.75

# Alpha (learning rate):
# controls how strongly Q-values are updated after each experience.
# A high value means faster learning, but also larger updates.
alpha = 0.9

# ======================================================================
# PART 1 - DEFINING THE ENVIRONMENT
# ======================================================================

# ----------------------------------------------------------------------
# 1. STATES
# ----------------------------------------------------------------------
# Each warehouse location is encoded as an integer state.
# This is useful because Q-Learning works conveniently with matrices:
# rows/columns can directly correspond to state indices.
#
# Mapping:
# A -> 0, B -> 1, ..., L -> 11
location_to_state = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6,
    'H': 7,
    'I': 8,
    'J': 9,
    'K': 10,
    'L': 11
}

# Optional reverse mapping.
# Not strictly necessary in Part 1, but very useful later when we want
# to convert a predicted route from numeric states back to location names.
state_to_location = {state: location for location, state in location_to_state.items()}

# ----------------------------------------------------------------------
# 2. ACTIONS
# ----------------------------------------------------------------------
# The actions are encoded with the same indices as the states.
#
# Intuition:
# if the robot is currently in one location, an action represents the next
# location it chooses to move to.
#
# Example:
# if the robot is in J (state 9), valid actions correspond only to the
# locations directly connected to J in the warehouse graph.
actions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# ----------------------------------------------------------------------
# 3. REWARDS
# ----------------------------------------------------------------------
# The reward matrix R has shape (12, 12):
# - rows    -> current state
# - columns -> chosen action / next location
#
# R[s, a] tells us the reward obtained when the robot is in state s
# and chooses action a.
#
# Reward design in this use case:
# - 1    -> the move is allowed
# - 0    -> the move is not allowed
# - 1000 -> very high reward assigned to the top-priority destination
#
# Why use a matrix?
# Because both the state space and the action space are discrete and finite.
# This makes the environment easy to encode and easy to use inside the
# Q-Learning update rule.
#
# Reading example:
# - Row 0 corresponds to location A.
# - Column 1 corresponds to action "go to B".
# - Therefore, R[0, 1] = 1 means that from A the robot can move to B.
#
# Another example:
# - Row 1 corresponds to B.
# - B is connected to A, C and F.
# - Therefore row 1 contains 1s in columns 0, 2 and 5.
R = np.array([
    # A  B  C  D  E  F  G  H  I  J  K  L
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # A
    [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # B
    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # C
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # D
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # E
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # F
    [0, 0, 1, 0, 0, 0, 1000, 1, 0, 0, 0, 0],  # G (top-priority location)
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],  # H
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],  # I
    [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],  # J
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],  # K
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]   # L
])

# ======================================================================
# INTERPRETATION OF THE ENVIRONMENT
# ======================================================================
#
# At this stage, the warehouse is represented as a graph:
# - each location is a node
# - each allowed movement is an edge
#
# The reward matrix encodes the graph structure numerically.
#
# The high reward on G means that, during training, the Q-Learning agent
# will progressively learn that moving toward G is highly beneficial.
#
# In later parts of the implementation:
# - a Q-table will be initialized
# - the agent will explore the environment
# - Q-values will be updated iteratively
# - the best route from any starting location to the destination will emerge
#   from the learned Q-values
#
# In other words:
# we are not hardcoding the shortest path explicitly;
# we are defining the environment and the incentives so that the agent can
# learn the best path by experience.