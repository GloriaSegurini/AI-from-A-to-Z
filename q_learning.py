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

# ======================================================================
# PART 2 - BUILDING THE AI SOLUTION WITH Q-LEARNING
# ======================================================================

# ----------------------------------------------------------------------
# INITIALIZING THE Q-VALUES
# ----------------------------------------------------------------------
# We create the Q-table, i.e. the matrix that will store the learned
# quality of each (state, action) pair.
#
# Shape: 12 x 12
# - rows    -> current state s_t
# - columns -> action a_t
#
# At the beginning of Q-Learning, all Q-values are initialized to 0.
# This follows the standard initialization described in the algorithm:
# the agent starts with no prior knowledge about which actions are good.
Q = np.array(np.zeros([12, 12]))

# ----------------------------------------------------------------------
# IMPLEMENTING THE Q-LEARNING PROCESS
# ----------------------------------------------------------------------
# We now run the learning loop for 1000 iterations.
# Each iteration simulates one learning step:
# 1. select a random current state
# 2. find all playable actions from that state
# 3. sample one valid next state/action
# 4. compute the Temporal Difference (TD)
# 5. update the corresponding Q-value
for i in range(1000):

    # Randomly pick one state among the 12 possible states.
    # This means that, during training, the agent explores the environment
    # from different starting points instead of always beginning from
    # the same location.
    current_state = np.random.randint(0, 12)

    # This list will contain only the actions that are valid from the
    # current state, i.e. the next locations that can actually be reached.
    playable_actions = []

    # We scan all 12 possible actions.
    for j in range(12):

        # If the reward is > 0, then this action is allowed.
        # In this implementation:
        # - reward 1    -> valid move
        # - reward 1000 -> destination / top-priority goal
        # - reward 0    -> invalid move
        if R[current_state, j] > 0:
            playable_actions.append(j)

    # From the valid actions, we randomly choose one.
    # This is the action the agent "tries" in this iteration.
    next_state = np.random.choice(playable_actions)

    # Temporal Difference (TD):
    # TD = immediate reward
    #      + discounted best future Q-value
    #      - current Q-value estimate
    #
    # Formula:
    # TD_t(s_t, a_t) = R(s_t, a_t) + gamma * max_a Q(s_{t+1}, a) - Q(s_t, a_t)
    #
    # Intuition:
    # we compare:
    # - what we currently think about taking this action
    # with
    # - a better target based on the reward just observed + the best
    #   achievable value from the next state
    TD = (
        R[current_state, next_state]
        + gamma * Q[next_state, np.argmax(Q[next_state,])]
        - Q[current_state, next_state]
    )

    # Bellman-style Q-value update:
    #
    # Q(s_t, a_t) <- Q(s_t, a_t) + alpha * TD
    #
    # If TD is positive, the action was better than expected and its
    # Q-value increases.
    # If TD is negative, the action was worse than expected and its
    # Q-value decreases.
    Q[current_state, next_state] = Q[current_state, next_state] + alpha * TD

# Optional inspection of the learned Q-table.
# We cast to int only for readability when printing.
print("Q-Values:")
print(Q.astype(int))

# ----------------------------------------------------------------------
# MAPPING STATES BACK TO HUMAN-READABLE LOCATIONS
# ----------------------------------------------------------------------
# The training works with numeric states (0..11), but when we want to show
# a final route, we want warehouse labels such as A, B, C, ...
#
# This dictionary is simply the inverse of location_to_state.
state_to_location = {
    state: location for location, state in location_to_state.items()
}

# ----------------------------------------------------------------------
# FIRST VERSION OF THE ROUTE FUNCTION
# ----------------------------------------------------------------------
# This function extracts the optimal route from a starting location to an
# ending location, using the Q-table already learned above.
#
# Important:
# this version assumes that the Q-table has already been trained for the
# desired destination (e.g. G was manually assigned reward 1000 in R).
def route(starting_location, ending_location):

    # Initialize the route with the starting location itself.
    route = [starting_location]

    # next_location keeps track of where the agent currently is while
    # reconstructing the path.
    next_location = starting_location

    # We keep moving until we reach the destination.
    while next_location != ending_location:

        # Convert the current location label (e.g. 'E') into its state index.
        starting_state = location_to_state[starting_location]

        # Among all actions available from this state, choose the one with
        # the highest Q-value.
        # This is the greedy exploitation step:
        # choose the action that looks best according to what has been learned.
        next_state = np.argmax(Q[starting_state,])

        # Convert the chosen next state index back into a location label.
        next_location = state_to_location[next_state]

        # Append that next location to the path.
        route.append(next_location)

        # Update the current location so that the loop can continue from there.
        starting_location = next_location

    # Return the full route as a list of warehouse labels.
    return route

# Example: shortest route from E to G, based on the learned Q-values.
print("Route: ", route('E', 'G'))


# ----------------------------------------------------------------------
# IMPROVED VERSION: DYNAMIC ROUTE FUNCTION
# ----------------------------------------------------------------------
# This second version is more general and more useful.
#
# Instead of assuming that the reward matrix R already contains a manually
# assigned reward of 1000 for one fixed destination, this function:
# 1. creates a copy of the reward matrix
# 2. automatically assigns reward 1000 to the chosen ending location
# 3. retrains a fresh Q-table on this updated reward matrix
# 4. extracts the route using the newly learned Q-values
#
# This is exactly the improvement described in the course:
# keep the original R as a neutral connectivity matrix (0s and 1s),
# and update the target dynamically inside the route() function.
def route(starting_location, ending_location):

    # Create a copy of the reward matrix so that the original environment
    # remains unchanged.
    R_new = np.copy(R) #NB: R_new should have all 0s and 1s, without the 1000 that was originally assigned to G

    # Convert the ending location into its state index.
    ending_state = location_to_state[ending_location]

    # Assign a very high reward to the destination itself.
    # This makes the destination the attractive target toward which
    # Q-values will propagate during training.
    R_new[ending_state, ending_state] = 1000

    # Initialize a fresh Q-table for this specific destination.
    Q = np.array(np.zeros([12, 12]))

    # Train Q-values again, now using the updated reward matrix R_new.
    for i in range(1000):

        # Randomly choose a current state.
        current_state = np.random.randint(0, 12)

        # Build the list of valid actions from that state.
        playable_actions = []

        for j in range(12):
            if R_new[current_state, j] > 0:
                playable_actions.append(j)

        # Randomly sample one valid action / next state.
        next_state = np.random.choice(playable_actions)

        # Compute the Temporal Difference using R_new instead of R,
        # because now the destination reward is dynamic.
        TD = (
            R_new[current_state, next_state]
            + gamma * Q[next_state, np.argmax(Q[next_state,])]
            - Q[current_state, next_state]
        )

        # Update the corresponding Q-value.
        Q[current_state, next_state] = Q[current_state, next_state] + alpha * TD

    # Once training is complete, reconstruct the route greedily.
    route = [starting_location]
    next_location = starting_location

    while next_location != ending_location:

        # Convert current location to numeric state.
        starting_state = location_to_state[starting_location]

        # Choose the action with maximum learned Q-value.
        next_state = np.argmax(Q[starting_state,])

        # Convert that state back into a warehouse label.
        next_location = state_to_location[next_state]

        # Append it to the route.
        route.append(next_location)

        # Move forward in the path reconstruction.
        starting_location = next_location

    # Return the final shortest / optimal learned route.
    return route

    
# ======================================================================
# PART 3 - GOING INTO PRODUCTION
# ======================================================================

# In this final production-oriented step, we extend the previous solution.
#
# Up to now, the function route(starting_location, ending_location)
# was able to compute the best path from any starting location to any
# final destination, by dynamically assigning reward 1000 to the ending
# location and then training a fresh Q-table on that goal.
#
# However, the warehouse case study introduces an additional business need:
# the robot should have the option to go through an intermediary location
# before reaching the final top-priority location.
#
# Why?
# Because the ranking system does not only provide a top-priority location,
# but also other highly relevant ones. In the example from the PDF:
# - G is the top-priority location
# - K is the second top-priority location
# - L is the third top-priority location
#
# Therefore, instead of only asking:
#   "What is the best route from start to end?"
#
# we now want to support:
#   "What is the best route from start to end, while going through
#    an intermediary top-priority location on the way?"
#
# ----------------------------------------------------------------------
# THE THREE POSSIBLE IDEAS DISCUSSED IN THE PDF
# ----------------------------------------------------------------------
#
# The PDF presents three possible ways to force the robot to go through
# the intermediary location K before reaching G:
#
# 1. Reward shaping on the preferred edge:
#    give a high positive reward to the action leading from J to K
#    (for example reward 500 on cell [9, 10]).
#    This would make the action J -> K more attractive than alternatives
#    such as J -> F, while still keeping the final destination reward
#    (1000) as the highest one.
#
# 2. Penalize the undesired edge:
#    give a negative reward to the action leading from J to F
#    (for example reward -500 on cell [9, 5]).
#    This would discourage the robot from taking that route, indirectly
#    pushing it toward K.
#
# 3. Split the problem into two route computations:
#    compute the best route from:
#       starting_location -> intermediary_location
#    and then compute the best route from:
#       intermediary_location -> ending_location
#    Finally, concatenate the two routes.
#
# According to the PDF, the first two ideas are easy to implement manually
# in a very specific case, but hard to automate in a general way.
# The reason is that it is easy to identify the intermediary location itself,
# but not easy to know in advance which previous location or edge should be
# boosted or penalized, because that depends on the actual optimal path,
# which changes with the starting and ending locations.
#
# For that reason, the implementation chooses the third approach:
# it is simpler, more general, and only requires calling the already-built
# route() function twice.
#
# Source: the PDF explicitly explains these three alternatives and states
# that the third one is selected because it is much easier to automate.
#
# ----------------------------------------------------------------------
# FINAL FUNCTION: BEST ROUTE WITH AN INTERMEDIARY LOCATION
# ----------------------------------------------------------------------
def best_route(starting_location, intermediary_location, ending_location):

    # First segment:
    # compute the optimal route from the starting location
    # to the intermediary location.
    #
    # Example:
    # route('E', 'K') might return:
    # ['E', 'I', 'J', 'K']
    first_leg = route(starting_location, intermediary_location)

    # Second segment:
    # compute the optimal route from the intermediary location
    # to the final destination.
    #
    # Example:
    # route('K', 'G') might return:
    # ['K', 'L', 'H', 'G']
    second_leg = route(intermediary_location, ending_location)

    # We concatenate the two routes, but we skip the first element of the
    # second route using [1:].
    #
    # Why?
    # Because second_leg starts with the intermediary location itself,
    # which is already the last element of first_leg.
    #
    # Without [1:], we would duplicate the intermediary node:
    # ['E', 'I', 'J', 'K'] + ['K', 'L', 'H', 'G']
    # ->
    # ['E', 'I', 'J', 'K', 'K', 'L', 'H', 'G']
    #
    # With [1:], we correctly obtain:
    # ['E', 'I', 'J', 'K', 'L', 'H', 'G']
    return first_leg + second_leg[1:]


# ----------------------------------------------------------------------
# TESTING THE FINAL PRODUCTION FUNCTION
# ----------------------------------------------------------------------
# Example from the PDF:
# - start from E
# - go through K (intermediary high-priority location)
# - end at G (top-priority location)
#
# Expected output:
# ['E', 'I', 'J', 'K', 'L', 'H', 'G']
print('Route:')
print(best_route('E', 'K', 'G'))
