# Standard library imports
import os
import time
import pickle
import json
import codecs
import pandas as pd

# Third-party library imports
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, A2C, DDPG
from stable_baselines3.common.vec_env import SubprocVecEnv

# Local application/library specific imports
import panda_gym

class CustomEnvWrapper(gym.Wrapper):
    def __init__(self, env, env_loc_id, interaction_dir):
        super().__init__(env)
        self.env_loc_id = env_loc_id
        self.logfile = f"{interaction_dir}"  # Generic logfile name for single env
        
        self.alex_current_state = None
        self.alex_episode_counter = 0
        self.alex_step_count = 0

        self.alex_transitions = pd.DataFrame(columns=['EP','ST_N','state', 'action', 'reward', 'done', 'truncated'])


    def step(self, action):

        observation, reward, done, truncated, info = super().step(action)

        new_transition = pd.DataFrame({
            'EP': [self.alex_episode_counter],
            'ST_N': [self.alex_step_count],
            'state': [self.alex_current_state],
            'action': [action],
            'reward': [reward],
            'done': [done],
            'truncated': [truncated]
        })

        self.alex_transitions = pd.concat([self.alex_transitions, new_transition], ignore_index=True)

        self.alex_current_state = observation
        self.alex_step_count += 1
        return observation, reward, done, truncated, info

    def log_data(self):
        self.alex_transitions.to_csv(f"{self.logfile}/id_{self.env_loc_id}_env_logs.csv", mode='a', header=False)
        self.alex_transitions = pd.DataFrame(columns=['EP','ST_N','state', 'action', 'reward', 'done', 'truncated'])

    def reset(self, **kwargs):
        #if self.transitions:
        #    self.log_data()
        #self.step_count = 0
        #self.current_state = self.define_new_obs(super().reset(**kwargs))
        if self.alex_episode_counter != 0:
            final_transition = pd.DataFrame({
                'EP': [self.alex_episode_counter],
                'ST_N': [self.alex_step_count],
                'state': [self.alex_current_state],
                'action': [None],
                'reward': [None],
                'done': [True],
                'truncated': [None]
            })
            self.alex_transitions = pd.concat([self.alex_transitions, final_transition], ignore_index=True)

        self.log_data()

        self.alex_current_state = super().reset(**kwargs)

        self.alex_episode_counter += 1
        self.alex_step_count = 0

        return self.alex_current_state


def make_env(env_id, interaction_dir):
    def _init():
        env = gym.make("PandaPickAndPlaceDense-v3")
        env = CustomEnvWrapper(env=env, env_loc_id=env_id, interaction_dir=interaction_dir)
        return env
    return _init

if __name__ == '__main__':
    file_path = os.getcwd()
    model_dir = os.path.join(file_path, "models/PPO/")
    logs_dir = os.path.join(file_path, "logs/")
    interaction_dir = os.path.join(file_path, "interactions/PPO/")
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(interaction_dir, exist_ok=True)

    num_envs = 10
    envs = SubprocVecEnv([make_env(env_id, interaction_dir) for env_id in range(1, num_envs + 1)])


    #env = gym.make("PandaPickAndPlace-v3", reward_type="dense") #''', render_mode="human",''' reward_type="dense")#, renderer="OpenGL")
    #env = CustomEnvWrapper(env = env, env_loc_id = 1, interaction_dir=interaction_dir)

    model = PPO(
        policy="MultiInputPolicy",                      # Policy type
        env=envs,                                       # Environment
        verbose=1,                                      # Verbosity level
        tensorboard_log=logs_dir,                       # Tensorboard log directory
        policy_kwargs=dict(net_arch=[512, 512]),        # Policy keyword arguments (neural network architecture)
        n_steps=2048,                                   # Steps per environment per update
        batch_size=128,                                 # Minibatch size
        n_epochs=128,                                   # Number of epochs
        learning_rate=3e-4,                             # Learning rate
        gamma=0.99,                                     # Discount factor
        gae_lambda=0.95,                                # GAE lambda
        clip_range=0.2,                                 # Clipping range
        vf_coef=0.5,                                    # Value function coefficient
        max_grad_norm=0.5                               # Max gradient norm
    )
    #model.load(os.path.join(model_dir, "4060000.zip"))
    #model = PPO(policy="MultiInputPolicy", env=env, verbose=1, tensorboard_log=logs_dir, policy_kwargs=policy_kwargs)#, policy_kwargs=policy_kwargs)

    training_steps = 20000000000
    for train_ep in range(1000):
        model.learn(training_steps, progress_bar=True, reset_num_timesteps=False, tb_log_name = "PPO_1")
        #model.save(f"{model_dir}/{time.time()}")

    envs.close()
