"""
A basic implementation of DQN for cartpole.

Deep Q Learning with NN with
- target network
- constant espilon-greedy exploration
- replay buffer

"""

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

from commons import QNetwork, ReplayBuffer, Transition
from pprint import pprint

#FIXME use the literature or references for the hyperparameters values
CONFIG = {
    "checkpoint": "weights/baseline-dqn.pth",
    "replay_capacity": int(1e4),
    "batch_size": 32,
    "learning_rate": 3e-4,
    "epsilon": 0.0,
    "episodes": 2,
    "max_steps": 500,
    "max_steps": 500,
    "target_upd_steps": 4
}

def optimize_qnet(optimizer,
                  q_net:QNetwork,
                  t_net:QNetwork,
                  buffer:ReplayBuffer) -> None:
    pass
# optimize_qnet

def behavior_policy(q_net: QNetwork, state: torch.tensor, action_dim: int) -> int:
    """
    Select an action based on epsilon-greedy exploration method.
    return action
    """
    global CONFIG
    assert state.dim() == 1

    action: int = 0
    p: float = random.random()

    if p < CONFIG["epsilon"]:
        # a ~ A
        action = random.choice(range(action_dim))
    else:
        # a = argmax Q(s,a)
        action = q_net(state.unsqueeze(0)).max(1).indices.item()

    assert type(action) is int
    return action
# behavior_policy

def main(device: str):
    global CONFIG
    print(f"Running on device {device}")
    # setup the environment
    env = gym.make('CartPole-v1')
    #env = gym.make('CartPole-v1', render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    assert state_dim == 4
    assert action_dim == 2

    # Init the experience replay buffer
    buffer = ReplayBuffer(capacity=CONFIG["replay_capacity"])
    # Init policy and target network with the same weights
    q_net = QNetwork(state_dim, action_dim)
    t_net = QNetwork(state_dim, action_dim)
    t_net.load_state_dict(q_net.state_dict())

    # only the state-action value function is upgraded by SDG
    optimizer = optim.AdamW(q_net.parameters(),
                            lr=CONFIG["learning_rate"],
                            amsgrad=True)

    for episode in range(CONFIG["episodes"]):
        print(f"Start episode {episode}...")

        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device)

        finished = False
        steps = 0      # episode steps
        steps_tot = 0  # global number of steps, for t_net update
        # run an episode until the agent fails or it reaches max_steps
        while not finished:
            action: int = behavior_policy(q_net, state, action_dim)
            obs, reward, terminated, *_ = env.step(action)
            # truncated is not taken into account in this setup
            finished = terminated

            # assign next state
            # None is a flag for avoiding calculating the max Q in terminal states
            next_state = None
            if not terminated:
                next_state = torch.tensor(obs, dtype=torch.float32, device=device)

            trans: Transition = Transition(state=state,
                                           action=torch.tensor([action],
                                                               dtype=torch.long,
                                                               device=device),
                                           next_state=next_state,
                                           reward=torch.tensor([reward],
                                                               dtype=torch.long,
                                                               device=device))
            # save in the experience replay buffer
            buffer.push(trans)

            optimize_qnet(optimizer, q_net, t_net, buffer)

            # clone Q into the target network at fixed steps interval,
            # regardless the relative position in the episode.
            steps_tot += 1
            if steps_tot % CONFIG["target_upd_steps"] == 0:
                t_net.load_state_dict(q_net.state_dict())
                print("Target Net updated")


            # while loop conditions
            state = next_state
            steps += 1
            if steps >= CONFIG["max_steps"]:
                finished = True
        # while finished
        print(f"Terminated episode {episode} with {steps=}")
    # for episodes

    #FIXME training loop

#FIXME RESTORE
#    torch.save(q_net.state_dict(), CHECKPOINT)
#    print(f"Model saved as {CHECKPOINT}")

if __name__ == '__main__':
    print("Hyperparameters")
    pprint(CONFIG)

    # set random seeds for reproducibility
    # It affects only the agent code, the environment is still unpredictable
    seed: int = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # if GPU is to be used
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    main(device)
