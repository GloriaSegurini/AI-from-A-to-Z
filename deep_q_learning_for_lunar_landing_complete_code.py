# Deep Q-Learning for Lunar Landing

# Part 0 - Installing the required packages and importing the libraries

# Installing Gymnasium

# !pip install gymnasium
# !pip install "gymnasium[atari, accept-rom-license]"
# !apt-get install -y swig
# !pip install gymnasium[box2d]

# Importing the libraries

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from collections import deque, namedtuple

# Part 1 - Building the AI

# Creating the architecture of the Neural Network

class Network(nn.Module):
  """
  Neural network that approximates the Q-function.
  Given a state of the environment, it outputs one Q-value for each possible action.
  For LunarLander:
  - state_size = 8 because each state is described by 8 numbers:
    1) x position of the lander
    2) y position of the lander
    3) x linear velocity
    4) y linear velocity
    5) angle of the lander
    6) angular velocity
    7) left leg contact with the ground (0 or 1)
    8) right leg contact with the ground (0 or 1)
  These 8 values are the information the agent receives at each time step to understand
  where it is, how it is moving, how tilted it is, and whether it has touched the ground.
  - action_size = 4 because the agent can choose among 4 discrete actions:
    0) do nothing
    1) fire the left orientation engine
    2) fire the main engine
    3) fire the right orientation engine
  So the network maps an 8-dimensional description of the current situation
  to 4 Q-values, one for each possible action.
  """

  def __init__(self, state_size, action_size, seed = 42):
    # We inherit from nn.Module because PyTorch neural networks are built as subclasses of it.
    # Calling super(...) activates that inheritance and initializes the base class properly.
    super(Network, self).__init__()

    # Fixing the random seed helps make initialization reproducible.
    # The lesson uses 42 as a default seed value.
    self.seed = torch.manual_seed(seed)

    # fc1 = first full connection between:
    # - the input layer, which contains state_size neurons
    # - the first hidden fully connected layer, which contains 64 neurons
    # Here state_size is 8 for LunarLander, because the state is an 8-dimensional vector.
    # The value 64 was not imposed by a theoretical rule:
    # it was chosen through experimentation and trial and error as a good architecture.
    self.fc1 = nn.Linear(state_size, 64)

    # fc2 = second full connection between:
    # - the first hidden layer of 64 neurons
    # - the second hidden layer, also of 64 neurons
    # At this point we are choosing to use a second intermediate layer
    # instead of connecting directly to the output layer.
    # Again, using 64 neurons here is an empirical design choice that worked well.
    self.fc2 = nn.Linear(64, 64)

    # fc3 = third full connection between:
    # - the second hidden layer of 64 neurons
    # - the output layer, which contains action_size neurons
    # The output layer has one neuron per possible action.
    # For LunarLander, action_size = 4, so the network outputs 4 Q-values:
    # one for each action the agent can choose.
    # We stop here because this architecture uses exactly two hidden fully connected layers.
    self.fc3 = nn.Linear(64, action_size)

  def forward(self, state):
    # The forward method propagates the signal from the input layer to the output layer.
    # Its main input is the current state of the environment.
    # In LunarLander, this state is an 8-dimensional vector.

    # First step of the forward propagation:
    # we send the input state through fc1, the first full connection,
    # to move from the input layer to the first hidden fully connected layer.
    x = self.fc1(state)

    # Then we activate that signal with ReLU (rectifier activation function).
    # This is the activation function mentioned in the lesson after the first full connection.
    x = F.relu(x)

    # Second step of the forward propagation:
    # now x is the signal coming from the first hidden layer,
    # so we pass x through fc2 to reach the second hidden fully connected layer.
    x = self.fc2(x)

    # Once again, we apply the ReLU activation function
    # to activate the neurons of the second hidden layer.
    x = F.relu(x)

    # Final step:
    # fc3 takes the activated signal x and forwards it to the output layer.
    # The output layer contains one value per action, so for LunarLander it returns 4 Q-values.
    # These are raw action values, not probabilities.
    return self.fc3(x)

# Part 2 - Training the AI

# Setting up the environment

import gymnasium as gym

# We import Gymnasium, the library that provides reinforcement learning environments.
# Here it will give us access to the LunarLander environment.
env = gym.make('LunarLander-v2')

# We create the environment by calling gym.make(...) with the exact environment name.
# 'LunarLander-v2' is the environment in which our AI will be trained.

# observation_space.shape tells us the shape of one state returned by the environment to the agent.
# For LunarLander, the state is a vector of 8 values, so the shape is (8,).
state_shape = env.observation_space.shape

# The state size is the number of elements inside one state vector.
# Since observation_space.shape is (8,), taking index 0 gives 8.
# This is the number of input values that will enter the neural network.
state_size = env.observation_space.shape[0]

# action_space.n gives the number of discrete actions available to the agent.
# For LunarLander, this value is 4.
# This is the number of outputs the neural network must produce.
number_actions = env.action_space.n

# These prints let us verify that the environment gives the expected parameters:
# - state shape: (8,)
# - state size: 8
# - number of actions: 4
print('State shape: ', state_shape)
print('State size: ', state_size)
print('Number of actions: ', number_actions)

# Initializing the hyperparameters

# Learning rate:
# controls how large each update step is when we optimize the neural network.
# A value that is too large can make training unstable,
# while a value that is too small can make learning very slow.
# Here 5e-4 (0.0005) is an empirical choice that worked well after experimentation.
learning_rate = 5e-4

# Minibatch size:
# number of experiences sampled from memory and used together in one training update.
# In practice, 100 is a common and effective choice for Deep Q-Learning.
minibatch_size = 100

# Discount factor (gamma):
# tells the agent how much future rewards matter compared to immediate rewards.
# If it were close to 0, the agent would be shortsighted and focus mostly on immediate reward.
# Since it is close to 1, the agent gives strong importance to long-term reward.
discount_factor = 0.99

# Replay buffer size:
# maximum number of past experiences stored in memory.
# Each experience typically contains (state, action, reward, next_state, done).
# Experience replay helps break correlations between consecutive observations,
# which makes training more stable and efficient.
# Here we allow the agent to store up to 100,000 experiences.
replay_buffer_size = int(1e5)

# Interpolation parameter (tau):
# used for the soft update of the target network.
# It controls how much of the local network is copied into the target network at each update.
# A small value like 1e-3 means the target network changes slowly,
# which helps stabilize DQN training.
interpolation_parameter = 1e-3

# Implementing Experience Replay

class ReplayMemory(object):

  def __init__(self, capacity):
    # We choose the device on which tensors will be stored:
    # use the GPU ("cuda:0") if it is available, otherwise use the CPU.
    # This is useful because training can be much faster on a GPU.
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # capacity = maximum number of experiences that the replay buffer can store.
    # This value will be given later when we create the ReplayMemory object.
    self.capacity = capacity

    # memory = list that will contain the stored experiences.
    # Each experience is a tuple like:
    # (state, action, reward, next_state, done)
    # We start with an empty list and fill it gradually during interaction with the environment.
    self.memory = []

  def push(self, event):
    # Add the new experience (event) to the replay memory.
    self.memory.append(event)

    # If we exceed the maximum capacity, we remove the oldest experience.
    # The oldest one is at index 0 because it was inserted first.
    # This keeps the replay buffer size bounded.
    if len(self.memory) > self.capacity:
      del self.memory[0]

  def sample(self, batch_size):
    # Randomly select batch_size experiences from the replay memory.
    # Random sampling helps break correlations between consecutive experiences,
    # which stabilizes Deep Q-Learning.
    experiences = random.sample(self.memory, k = batch_size)

    # We now separate the sampled experiences into 5 groups:
    # states, actions, rewards, next_states, and dones.
    # np.vstack stacks them vertically to form proper batches,
    # then we convert them to PyTorch tensors and send them to the chosen device.
    states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(self.device)
    actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long().to(self.device)
    rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(self.device)
    next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(self.device)

    # done is a Boolean flag telling us whether the episode ended after that transition.
    # We convert it to 0/1 first (uint8), then to float so it can be used in tensor computations.
    dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

    # Return the full minibatch, already split into tensors ready for learning.
    return states, next_states, actions, rewards, dones

# Implementing the DQN class

class Agent():

  def __init__(self, state_size, action_size):
    # Choose whether computations will run on GPU or CPU.
    # This is the same device logic used earlier in ReplayMemory.
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Store the size of the state space as an object variable.
    # For LunarLander, this is 8.
    self.state_size = state_size

    # Store the size of the action space as an object variable.
    # For LunarLander, this is 4.
    self.action_size = action_size

    # local_qnetwork = the network that is directly trained and updated by gradient descent.
    # It has the same architecture as the Network class we defined before.
    self.local_qnetwork = Network(state_size, action_size).to(self.device)

    # target_qnetwork = a second Q-network with the same architecture.
    # It is used to compute more stable target values during learning.
    self.target_qnetwork = Network(state_size, action_size).to(self.device)

    # Optimizer used to update the weights of the local Q-network.
    # Adam takes the parameters (weights) of local_qnetwork and updates them
    # using the learning_rate defined earlier.
    self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr = learning_rate)

    # Replay memory of the agent.
    # It stores past experiences so that we can sample them later for training.
    self.memory = ReplayMemory(replay_buffer_size)

    # Time step counter.
    # It will be used later to decide when the agent should trigger a learning update.
    self.t_step = 0

  def step(self, state, action, reward, next_state, done):
    # First role of step():
    # store the current experience in replay memory.
    # The experience is the tuple (state, action, reward, next_state, done).
    self.memory.push((state, action, reward, next_state, done))

    # Second role of step():
    # update the time step counter and reset it every 4 steps using modulo.
    # This lets the agent trigger learning only every 4 environment steps.
    # This is different from minibatch_size:
    # - t_step controls WHEN learning happens
    # - minibatch_size controls HOW MANY experiences are used when learning happens
    self.t_step = (self.t_step + 1) % 4

    # If the counter comes back to 0, it means we have reached another group of 4 steps.
    if self.t_step == 0:

      # Before learning, we check that replay memory already contains
      # more experiences than the minibatch size.
      # This ensures that we can sample a full minibatch for training.
      # So:
      # - every 4 steps, we ASK whether it is time to learn
      # - only if memory is large enough, we actually sample 100 experiences and update the network
      if len(self.memory.memory) > minibatch_size:

        # Randomly sample a minibatch of experiences from replay memory.
        experiences = self.memory.sample(100)

        # Learn from that minibatch using the discount factor.
        # The learn() method will be implemented later in the Agent class.
        self.learn(experiences, discount_factor)

#   def act(self, state, epsilon = 0.):
#     state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
#     self.local_qnetwork.eval()
#     with torch.no_grad():
#       action_values = self.local_qnetwork(state)
#     self.local_qnetwork.train()
#     if random.random() > epsilon:
#       return np.argmax(action_values.cpu().data.numpy())
#     else:
#       return random.choice(np.arange(self.action_size))

#   def learn(self, experiences, discount_factor):
#     states, next_states, actions, rewards, dones = experiences
#     next_q_targets = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
#     q_targets = rewards + discount_factor * next_q_targets * (1 - dones)
#     q_expected = self.local_qnetwork(states).gather(1, actions)
#     loss = F.mse_loss(q_expected, q_targets)
#     self.optimizer.zero_grad()
#     loss.backward()
#     self.optimizer.step()
#     self.soft_update(self.local_qnetwork, self.target_qnetwork, interpolation_parameter)

#   def soft_update(self, local_model, target_model, interpolation_parameter):
#     for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
#       target_param.data.copy_(interpolation_parameter * local_param.data + (1.0 - interpolation_parameter) * target_param.data)

# # Initializing the DQN agent

# agent = Agent(state_size, number_actions)

# # Training the DQN agent

# number_episodes = 2000
# maximum_number_timesteps_per_episode = 1000
# epsilon_starting_value  = 1.0
# epsilon_ending_value  = 0.01
# epsilon_decay_value  = 0.995
# epsilon = epsilon_starting_value
# scores_on_100_episodes = deque(maxlen = 100)

# for episode in range(1, number_episodes + 1):
#   state, _ = env.reset()
#   score = 0
#   for t in range(maximum_number_timesteps_per_episode):
#     action = agent.act(state, epsilon)
#     next_state, reward, done, _, _ = env.step(action)
#     agent.step(state, action, reward, next_state, done)
#     state = next_state
#     score += reward
#     if done:
#       break
#   scores_on_100_episodes.append(score)
#   epsilon = max(epsilon_ending_value, epsilon_decay_value * epsilon)
#   print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)), end = "")
#   if episode % 100 == 0:
#     print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)))
#   if np.mean(scores_on_100_episodes) >= 200.0:
#     print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode - 100, np.mean(scores_on_100_episodes)))
#     torch.save(agent.local_qnetwork.state_dict(), 'checkpoint.pth')
#     break

# # Part 3 - Visualizing the results

# import glob
# import io
# import base64
# import imageio
# from IPython.display import HTML, display
# from gym.wrappers.monitoring.video_recorder import VideoRecorder

# def show_video_of_model(agent, env_name):
#     env = gym.make(env_name, render_mode='rgb_array')
#     state, _ = env.reset()
#     done = False
#     frames = []
#     while not done:
#         frame = env.render()
#         frames.append(frame)
#         action = agent.act(state)
#         state, reward, done, _, _ = env.step(action.item())
#     env.close()
#     imageio.mimsave('video.mp4', frames, fps=30)

# show_video_of_model(agent, 'LunarLander-v2')

# def show_video():
#     mp4list = glob.glob('*.mp4')
#     if len(mp4list) > 0:
#         mp4 = mp4list[0]
#         video = io.open(mp4, 'r+b').read()
#         encoded = base64.b64encode(video)
#         display(HTML(data='''<video alt="test" autoplay
#                 loop controls style="height: 400px;">
#                 <source src="data:video/mp4;base64,{0}" type="video/mp4" />
#              </video>'''.format(encoded.decode('ascii'))))
#     else:
#         print("Could not find video")

# show_video()
