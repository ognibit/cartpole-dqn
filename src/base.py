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
from torch import tensor
import random
import numpy as np

from commons import QNetwork, ReplayBuffer, Transition
from pprint import pprint

#FIXME use the literature or references for the hyperparameters values
CONFIG = {
    "checkpoint": "weights/baseline-dqn.pth",
    "replay_capacity": int(1e4),
    "batch_size": 64,
    "learning_rate": 5e-4,
    "epsilon": 0.05,
    "discount": 0.99,
    "episodes": 200,
    "max_steps": 1000,
    "target_upd_steps": 5
}


def sample_batch(buffer: ReplayBuffer) -> tuple[tensor, tensor, tensor, tensor, list[int]]:
    """
    Sample batch_size elements from the buffer and create the minibatches.
    Since the next_states can be shorter due to the presence of the terminal
    states, the next_states_mask gives the reference indexes of non-terminal
    states.

    For example:
    actions[next_states_mask] are the actions that led to non-terminal states.

    buffer: replay buffer. len(buffer) must be >= batch_size.

    return states, actions, next_states, rewards, next_states_mask
    """
    global CONFIG
    batch_size: int = CONFIG["batch_size"]
    assert len(buffer) >= batch_size

    trans: list[Transition] = buffer.sample(batch_size)

    b_states = torch.vstack([t.state for t in trans])
    assert b_states.dim() == 2
    assert b_states.shape[0] == batch_size
    assert b_states.shape[1] == 4

    b_actions = torch.vstack([t.action for t in trans])
    assert b_actions.dim() == 2
    assert b_actions.shape[0] == batch_size
    assert b_actions.shape[1] == 1

    # in the cartpole, all rewards are 1
    b_rewards = torch.vstack([t.reward for t in trans])
    assert b_rewards.dim() == 2
    assert b_rewards.shape[0] == batch_size
    assert b_rewards.shape[1] == 1

    # next_state can be None is case it is terminal and then it must be
    # properly managed. The nstates_mask marks the non-terminal states
    nstates_mask = [i for i in range(batch_size)
                    if trans[i].next_state is not None]

    # batch of next states can be shorter due to the missing terminal states
    b_next_states = torch.vstack([t.next_state for t in trans
                                  if t.next_state is not None])
    assert b_next_states.dim() == 2
    assert b_next_states.shape[0] <= batch_size
    assert b_next_states.shape[1] == 4

    return b_states, b_actions, b_next_states, b_rewards, nstates_mask
# sample_batch

def optimize_qnet(optimizer,
                  q_net:QNetwork,
                  t_net:QNetwork,
                  buffer:ReplayBuffer) -> float:
    """
    Sample from the replay buffer and optimize the q_net.

    return loss
    """
    global CONFIG
    batch_size: int = CONFIG["batch_size"]
    if len(buffer) < batch_size:
        # wait for enough transitions to sample
        return float('inf')

    device: str = CONFIG["device"]
    discount: float = CONFIG["discount"]
    states, actions, nextstates, rewards, mask_lst = sample_batch(buffer)
    mask = tensor(mask_lst, device=device)

    # t_net must not be updated via SDG
    with torch.no_grad():
        # targets [B, 1]
        targets = rewards
        # t_next(next) [B,A] -> max(1) [B] -> unsqueeze(1) [B,1]
        # q_max dim can be less then B because of terminal states
        q_max = t_net(nextstates).max(dim=1).values.unsqueeze(1)
        # y = r + gamma * maxQ  if nextstate is not terminal
        # y = r                 if nextstate is terminal
        # the mask disables the sum for the terminal states
        targets[mask] += discount * q_max

    # get Q(s,a), the policy predictions
    # q_net(s) [B,A] -> gather [B, 1] (select the values corresponding to actions)
    q = q_net(states).gather(1, actions)

    criterion = nn.MSELoss()
#    criterion = nn.SmoothL1Loss()
    loss = criterion(q, targets)

    # backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
# optimize_qnet

def behavior_policy(q_net: QNetwork, state: tensor, action_dim: int) -> int:
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
        with torch.no_grad():
            action = q_net(state.unsqueeze(0)).max(1).indices.item()

    assert type(action) is int
    return action
# behavior_policy

def main():
    global CONFIG
    device: str = CONFIG["device"]
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

        state, info = env.reset()
        state = tensor(state, dtype=torch.float32, device=device)

        finished: bool = False
        steps: int = 0      # episode steps
        steps_tot: int = 0  # global number of steps, for t_net update
        max_loss: float = 0.0
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
                next_state = tensor(obs, dtype=torch.float32, device=device)

            # reward as float to be used in TD target easily
            trans: Transition = Transition(state=state,
                                           action=tensor([action],
                                                               dtype=torch.long,
                                                               device=device),
                                           next_state=next_state,
                                           reward=tensor([reward],
                                                               dtype=torch.float,
                                                               device=device))
            # save in the experience replay buffer
            buffer.push(trans)

            loss: float = optimize_qnet(optimizer, q_net, t_net, buffer)
            max_loss = max(max_loss, loss)

            # clone Q into the target network at fixed steps interval,
            # regardless the relative position in the episode.
            steps_tot += 1
            if steps_tot % CONFIG["target_upd_steps"] == 0:
                t_net.load_state_dict(q_net.state_dict())


            # while loop conditions
            state = next_state
            steps += 1
            if steps >= CONFIG["max_steps"]:
                finished = True
        # while finished
        print(f"Terminated episode {episode} with {steps=}, {max_loss=}")
    # for episodes

#FIXME RESTORE
#    torch.save(q_net.state_dict(), CHECKPOINT)
#    print(f"Model saved as {CHECKPOINT}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=CONFIG['batch_size'])
    parser.add_argument('--checkpoint', type=str, default=CONFIG['checkpoint'])
    parser.add_argument('--discount', type=float, default=CONFIG['discount'])
    parser.add_argument('--episodes', type=int, default=CONFIG['episodes'])
    parser.add_argument('--epsilon', type=float, default=CONFIG['epsilon'])
    parser.add_argument('--learning_rate', type=float, default=CONFIG['learning_rate'])
    parser.add_argument('--max_steps', type=int, default=CONFIG['max_steps'])
    parser.add_argument('--replay_capacity', type=int, default=CONFIG['replay_capacity'])
    parser.add_argument('--target_upd_steps', type=int, default=CONFIG['target_upd_steps'])

    args = parser.parse_args()
    CONFIG.update(vars(args))

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

    print("Hyperparameters")
    pprint(CONFIG)
    CONFIG["device"] = device

    main()
