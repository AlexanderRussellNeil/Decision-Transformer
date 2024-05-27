import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import time
import gymnasium as gym
import numpy as np
import torch
import sys
from decision_transformer.models.decision_transformer import DecisionTransformer
import panda_gym

import os

def extract_numeric_values(data):
    numeric_values = []
    
    for item in data:
        for key, value in item.items():
            if key == 'is_success':
                continue  # Skip the 'is_success' key

            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, np.ndarray):
                        numeric_values.extend(subvalue.flatten().tolist())
            elif isinstance(value, np.ndarray):
                numeric_values.extend(value.flatten().tolist())
            elif isinstance(value, (float, int, np.number)):
                numeric_values.append(value)

    return np.array(numeric_values)


def extract_numeric_values_on_the_go(data):
    result = []
    for key, value in data.items():
        result.extend(value.tolist())

    return np.array(result)


def evaluate_episode(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        device='cuda',
        target_return=None,
        mode='normal',
        state_mean=0.,
        state_std=1.,
):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32)
    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return=target_return,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, truncated, info = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length


def evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
    ):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(np.array(state_mean)).to(device=device)
    state_std = torch.from_numpy(np.array(state_std)).to(device=device)

    state = env.reset()

    state = extract_numeric_values(state)

    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []

    episode_return, episode_length = 0, 0
    t = 0
    while True:

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, truncated, info = env.step(action)

        time.sleep(0.23)

        state = extract_numeric_values_on_the_go(state)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        if mode != 'delayed':
            pred_return = target_return[0,-1] - (reward/scale)
        else:
            pred_return = target_return[0,-1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1
        t += 1
        if t == 50:
            done = True
        if done:
            time.sleep(10)
            break

    return episode_return, episode_length


if __name__ == '__main__':
    env_name = 'PandaPickAndPlaceDense-v3'
    state_dim = 25  # Example state dimension
    act_dim = 4     # Example action dimension

    # Load the trained model
    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=20,  # Example max length
        max_ep_len=50,  # Example max episode length
        hidden_size=128,  # Example hidden size
        n_layer=3,  # Example number of layers
        n_head=1,  # Example number of heads
        n_inner=4*128,  # Example number of inner layers
        activation_function='relu',  # Example activation function
        n_positions=1024,  # Example number of positions
        resid_pdrop=0.1,  # Example residual dropout
        attn_pdrop=0.1,  # Example attention dropout
    )
    try:
        model_path = os.getcwd() + "/models checkpoints/ULTIMATE REV 4/"
        
        user_input = input("Which epoch to load? ")
        # Load the state dictionary of the trained model
        model.load_state_dict(torch.load(f'{model_path}4 datasets, Model iter = {user_input}'))

        # Test the model using evaluate_episode_rtg
        env = gym.make(env_name, render_mode='human')
        state_mean = np.zeros(state_dim)
        state_std = np.ones(state_dim)
        target_return = 1000
        episode_return, episode_length = evaluate_episode_rtg(
                env, state_dim, act_dim, model, max_ep_len=50, scale=1000.,
                state_mean=0.010294408580045152, state_std=1.1180083836623258, device='cuda',
                target_return=target_return, mode='normal'
            )

        print(f'Episode Return: {episode_return}, Episode Length: {episode_length}')
    except Exception as e:
        print(f"Error : {e}")

