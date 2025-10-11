"""
A basic implementation of DQN for cartpole.

Deep Q Learning with NN with
- target network
- replay buffer
- epsilon-greedy exploration with constant espilon
- early stop after 5 time hit max number of steps

"""

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch import tensor
import random
import matplotlib.pyplot as plt
from statistics import mean
import pandas as pd
import numpy as np

from commons import QNetwork, ReplayBuffer, Transition
from commons import PoleLengthCurriculum, DefaultPoleLength
from pprint import pprint

CONFIG = {
    "checkpoint": "baseline-dqn",
    "replay_capacity": 200_000,
    "batch_size": 128,
    "learning_rate": 0.001,
    "epsilon": 0.3,
    "discount": 0.99,
    "episodes": 300, # FIXME deprecated
    "tot_steps": 120_000, # total steps for the training
    "max_steps": 1000,    # maximum steps for one episode
    "target_upd_steps": 10
}

def plot_rewards(episodes: list[int], rewards: list[int],
                 losses: list[float], lengths: list[float]) -> None:
    global CONFIG

    checkpoint: str = CONFIG["checkpoint"]

    fig, ax1 = plt.subplots()

#    ax1.plot(episodes, losses, 'b-', label='Loss')
#    ax1.set_xlabel("Episodes")
#    ax1.set_ylabel("Loss", color='b')
#    ax1.tick_params(axis='y', labelcolor='b')

    ax1.plot(episodes, lengths, 'b-', label='Pole Length')
    ax1.set_xlabel("Episodes")
    ax1.set_ylabel("Pole Lengths", color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.plot(episodes, rewards, 'r-', label='Episode Length')
    ax2.set_ylabel("Episode Length", color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    #FIXME add parameters in the title?
    plt.title(f"{checkpoint}")

    df = pd.DataFrame({"episode": episodes,
                       "poleLen": lengths,
                       "loss": losses,
                       "reward": rewards})
    df.to_csv("exports/" + checkpoint + ".csv")
    plt.savefig("charts/" + checkpoint + ".png")
    plt.show()
# plot_rewards

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
        targets = rewards.clone()
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
    loss = criterion(q, targets)

    # backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
# optimize_qnet

def behavior_policy(q_net: QNetwork, state: tensor, action_dim: int, steps: int) -> int:
    """
    Select an action based on epsilon-greedy exploration method.
    return action
    """
    global CONFIG
    assert state.dim() == 1

    action: int = 0
    p: float = random.random()

    epsilon: float = CONFIG["epsilon"]

    if p < epsilon:
        # a ~ A
        action = random.choice(range(action_dim))
    else:
        # a = argmax Q(s,a)
        with torch.no_grad():
            action = q_net(state.unsqueeze(0)).max(1).indices.item()

    assert type(action) is int
    assert 0 <= action <= 1
    return action
# behavior_policy

def train(pc: PoleLengthCurriculum):
    global CONFIG
    device: str = CONFIG["device"]
    print(f"Running on device {device}")
    print("Curriculum: ", type(pc))
    # setup the environment
    env = gym.make('CartPole-v1')
    #env = gym.make('CartPole-v1', render_mode="human")
    state_dim: int = env.observation_space.shape[0]
    action_dim: int = env.action_space.n

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

    # plot vectors
    episodes = []
    episodes_rewards = []
    episodes_losses = []
    episodes_lengths = []
    # global number of steps, for t_net update
    steps_tot: int = 0
    # count how many consecutive times the agent reach the "max_step"
    max_steps: int = CONFIG["max_steps"]

    # force random seed for repeatibility
    env.reset(seed=42)
    episode: int = 0
    while steps_tot < CONFIG["tot_steps"]:

        state, info = env.reset()
        pole_len: float = pc.set_pole_length(env, steps_tot)
        episodes_lengths.append(pole_len)

        state = tensor(state, dtype=torch.float32, device=device)

        finished: bool = False
        steps: int = 0      # episode steps
        losses: list[float] = []
        # run an episode until the agent fails or it reaches max_steps
        while not finished:
            action: int = behavior_policy(q_net, state, action_dim, steps_tot)
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
            losses.append(loss)

            # clone Q into the target network at fixed steps interval,
            # regardless the relative position in the episode.
            steps_tot += 1
            if steps_tot % CONFIG["target_upd_steps"] == 0:
                t_net.load_state_dict(q_net.state_dict())


            # while loop conditions
            state = next_state
            steps += 1
            if steps >= max_steps:
                finished = True
        # while finished

        # plot one episode information, not aggregation
        loss_mean: float = mean(losses)
        episodes.append(episode)
        episodes_rewards.append(steps)
        episodes_losses.append(loss_mean)
        print(f"Episode {episode}[len: {pole_len:.04f}], {steps=}, {steps_tot=}, {loss_mean=:.04f} ")

        episode += 1
    # for episodes

    plot_rewards(episodes, episodes_rewards, episodes_losses, episodes_lengths)

    checkpoint: str = "weights/" + CONFIG["checkpoint"] + ".pth"
    torch.save(q_net.state_dict(), checkpoint)
    print(f"Model saved as {checkpoint}")


def main(pc: PoleLengthCurriculum):
    import argparse
    import os

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=CONFIG['batch_size'])
    parser.add_argument('--checkpoint', type=str, default=CONFIG['checkpoint'])
    parser.add_argument('--discount', type=float, default=CONFIG['discount'])
    parser.add_argument('--episodes', type=int, default=CONFIG['episodes'])
    parser.add_argument('--tot_steps', type=int, default=CONFIG['tot_steps'])
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
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # create the output directories
    os.makedirs("exports", exist_ok=True)
    os.makedirs("charts", exist_ok=True)
    os.makedirs("weights", exist_ok=True)

    # if GPU is to be used
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    print("Hyperparameters")
    CONFIG["device"] = device
    CONFIG["curriculum"] = str(type(pc))
    pprint(CONFIG)
    with open(f"exports/{CONFIG['checkpoint']}.json", "w") as f:
        pprint(CONFIG, stream=f)

    train(pc)

if __name__ == '__main__':
    pc = DefaultPoleLength()
    main(pc)
