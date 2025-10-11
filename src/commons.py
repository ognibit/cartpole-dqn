"""
Common parts to use in the different training setups
"""
import torch
import torch.nn as nn
from collections import namedtuple, deque
from abc import ABC, abstractmethod
import random

# Inspired by
# https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# Each component is a tensor
Transition = namedtuple('Transition',
                        ['state', 'action', 'next_state', 'reward'])

# This is the 'standard' neural network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer():
    """
    Exeperience Replay Buffer of transitions.
    The transitions are stored as tuple of
    (state, action, next_state, reward).
    A sample method is provided to uniformily choose a batch of transitions.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque([], maxlen=self.capacity)

    def push(self, transition: Transition) -> None:
        """
        Add the transition in the buffer.
        If the buffer is full, the oldest transition will be discarded
        """
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        """
        Uniform sample of batch_size transitions
        """
        # Return a k length list of unique elements chosen
        # from the population sequence. (without replacement)
        assert batch_size <= len(self.buffer)

        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class PoleLengthCurriculum(ABC):
    """
    Abstract class for scheduling the pole lenght in the cartpole environment
    """

    @abstractmethod
    def set_pole_length(self, env, steps_tot: int) -> float:
        """
        Set the pole lenght in the environment at beginning of the episode.

        steps_tot: current number of executed steps

        return the pole lenght
        """
        pass

class DefaultPoleLength(PoleLengthCurriculum):

    def set_pole_length(self, env, steps_tot: int) -> float:
        pole_len: float = env.unwrapped.length
        return pole_len

