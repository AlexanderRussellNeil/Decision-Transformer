import warnings
warnings.filterwarnings("ignore")

import gymnasium as gym
import numpy as np
import torch
import wandb
import argparse
import pickle
import random
import sys
from decision_transformer.evaluation.evaluate_episodes import evaluate_episode
from decision_transformer.evaluation.evaluate_episodes import evaluate_episode_rtg
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer
from datetime import datetime
import panda_gym
import pandas as pd
import re
from torch.utils.tensorboard import SummaryWriter


import os

def save_model(model, save_path='model_checkpoints', file_name='trained_model_35.pt'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file = os.path.join(save_path, file_name)
    torch.save(model.state_dict(), save_file)
    print(f'Trained model saved to {save_file}')


def extract_numeric_values(state_str):
    nums = []
    number = ""
    is_success = 0.0

    tokens = re.split(r'[\s\[\]\(\)\{\},\'=:]+', state_str)
    
    nums = []
    
    for token in tokens:
        try:
            num = float(token)
            nums.append(num)
        except ValueError:
            continue


    return nums


def convert_df_to_original_structure(df):
    trajectories = []
    episodes = df['episode'].unique()

    for episode in episodes:
        episode_df = df[df['episode'] == episode]

        # Calculate next_observations
        next_observations = episode_df['state'].shift(-1).ffill().values

        trajectory = {
            'observations': np.vstack(episode_df['observation'].apply(lambda x: np.array(x)).values),
            'next_observations': np.vstack(episode_df['observation'].shift(-1).ffill().apply(lambda x: np.array(x)).values),
            'actions': np.vstack(episode_df['action'].apply(lambda x: np.array(x)).values),
            'rewards': np.array(episode_df['reward'].values),
            'dones': np.array(episode_df['done'].values),
            'terminals': np.array(episode_df['truncated'].values)
        }
        trajectories.append(trajectory)

    return trajectories



def scale_action(action):
    pass


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum

def combine_observation_observation(row):
    state_values = extract_numeric_values(row['state'])
    return state_values

def combine_observation_action(row):
    state_values = extract_numeric_values(row['action'])
    return state_values

def experiment(exp_prefix, variant):
    print(f"Experiment Prefix: {exp_prefix}")
    print(f"Variant: {variant}")
    
    device = variant.get('device', 'cuda') # get the device to work on
    log_to_wandb = variant.get('log_to_wandb', False) # logger
    if log_to_wandb is True:
        wandb.init()
        wandb.log()


    env_name, dataset = variant['env'], variant['dataset'] 
    model_type = variant['model_type']
    group_name = f'{exp_prefix}-{env_name}-{dataset}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join('logs', exp_prefix))

    if env_name == 'PandaPickAndPlaceDense':
        env = gym.make('PandaPickAndPlaceDense-v3')#, render_mode = "human")
        max_ep_len = 50
        env_targets = [600, 300]  # evaluation conditioning targets
        scale = 1000.  # normalization for rewards/returns
    else:
        raise NotImplementedError


    # load dataset
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    dataset_paths = [
    'data/PandaPickAndPlaceDense-medium-v3-1.pkl',
    'data/PandaPickAndPlaceDense-medium-v3-2.pkl',
    'data/PandaPickAndPlaceDense-medium-v3-3.pkl',
    'data/PandaPickAndPlaceDense-medium-v3-4.pkl'
    ]

    max_episode = 0
    
    dataframes = []
    for dataset_path in dataset_paths:
        try:
            with open(dataset_path, 'rb') as f:
                trajectories = pickle.load(f)
            df = pd.DataFrame(trajectories)
            df['episode'] += max_episode
            max_episode = df['episode'].max() + 1
            dataframes.append(df)
        except Exception as e:
            print(f"Failed to load {dataset_path}: {e}")
    
    trajectories = pd.concat(dataframes, ignore_index=True)

    
    trajectories = trajectories.dropna(subset=['truncated'])
    trajectories['truncated'] = trajectories['truncated'].astype(bool)

    trajectories['observation'] = trajectories.apply(combine_observation_observation, axis=1)
    trajectories['action'] = trajectories.apply(combine_observation_action, axis=1)
    #print("lenghth:      ", trajectories['observation'].tail(1))

    g_trajectories = trajectories.groupby('episode')


    #print(trajectories.info())

    states, traj_lens, returns = [], [], []
    for episode, group in g_trajectories:
        observations = group['observation']
        rewards = group['reward'].sum()        
        states.append(observations)
        traj_lens.append(len(observations))
        returns.append(rewards)
    
    traj_lens, returns = np.array(traj_lens), np.array(returns)
    num_timesteps = sum(traj_lens)
    states = np.concatenate(states, axis=0)
    states = np.concatenate(states, axis=0)
    state_mean = np.mean(states, axis=0)
    state_std = np.std(states, axis=0) + 1e-6

    #print(trajectories.describe().T)
    #print(trajectories.info())

    print('=' * 100)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'State mean: {state_mean}, state std: {state_std}')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 100)

    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    pct_traj = variant.get('pct_traj', 1.)

    num_timesteps = max(int(pct_traj*num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(g_trajectories) - 2

    #print(ind)
    #print(timesteps)
    #print(traj_lens[sorted_inds[ind]])
    #print(num_timesteps)

    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])


    trajectories = convert_df_to_original_structure(trajectories)

    state_dim = len(trajectories[0]['observations'][0])
    act_dim = len(trajectories[0]['actions'][0])


    def get_batch(batch_size=256, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)

            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, d, rtg, timesteps, mask
    
    mode = variant.get('mode', 'normal')
    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths = [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    if model_type == 'dt':
                        ret, length = evaluate_episode_rtg(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            scale=scale,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                    else:
                        ret, length = evaluate_episode(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                returns.append(ret)
                lengths.append(length)
            return {
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                f'target_{target_rew}_length_std': np.std(lengths),
            }
        return fn

    
    model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
            tensorboard_log = "/home/21587102/Desktop/COMP6002/decision-transformer-master/gym/logs",
        )

    model = model.to(device=device)
    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )

    trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            eval_fns=[eval_episodes(tar) for tar in env_targets], # can exclude this one, but not preffered
        )
    
    curr_time = datetime.now()
    model_path = os.getcwd() + f"models checkpoints/{curr_time}"
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    print(model_path)

    for iter in range(variant['max_iters']):
        outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter+1, print_logs=True)

        # Log metrics to TensorBoard
        for key, value in outputs.items():
            if isinstance(value, dict):  # If the value is a dictionary, iterate over its items
                for sub_key, sub_value in value.items():
                    writer.add_scalar(f'{key}/{sub_key}', sub_value, iter)
            else:  # Otherwise, log the key-value pair directly
                writer.add_scalar(key, value, iter)
        
        save_model(model, save_path = model_path, file_name = f"4 datasets, Model iter = {iter}")

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='PandaPickAndPlaceDense') 
    parser.add_argument('--dataset', type=str, default='medium') # medium, medium-replay, medium-expert, expert 
    parser.add_argument('--mode', type=str, default='normal') # normal for standard setting, delayed for sparse 
    parser.add_argument('--K', type=int, default=20) # 
    parser.add_argument('--pct_traj', type=float, default=1.) # 
    parser.add_argument('--batch_size', type=int, default=64) # 
    parser.add_argument('--model_type', type=str, default='dt')  # dt for decision transformer, bc for behavior cloning 
    parser.add_argument('--embed_dim', type=int, default=128) # 
    parser.add_argument('--n_layer', type=int, default=3) # 
    parser.add_argument('--n_head', type=int, default=1) # 
    parser.add_argument('--activation_function', type=str, default='relu') # 
    parser.add_argument('--dropout', type=float, default=0.1) # 
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.0000031) # 
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4) # 
    parser.add_argument('--warmup_steps', type=int, default=1) # 
    parser.add_argument('--num_eval_episodes', type=int, default=200) # 
    parser.add_argument('--max_iters', type=int, default=1400) # changed from 10 to 50 BY ALEX 
    parser.add_argument('--num_steps_per_iter', type=int, default=10000) # 
    parser.add_argument('--device', type=str, default='cuda') # 
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False) # 
    
    args = parser.parse_args() 
    
    print(args)

    experiment('gym-experiment', variant=vars(args))
