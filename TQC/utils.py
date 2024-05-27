
import panda_gym
import gymnasium

import os
import time
import pickle
import json
import codecs
import re
import imageio

import pybullet as p
import pybullet_data
import gymnasium

# Third-party library imports
import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO, A2C, DDPG
from stable_baselines3.common.vec_env import SubprocVecEnv
from gymnasium import spaces
from gymnasium.spaces import Box

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
        #self.step_count += 1
        #observation, reward, done, truncated, info = super().step(action)
        #observation = self.update_new_obs(observation)
        #self.transitions.append((self.my_current_state, action, reward, done, truncated))
        #self.my_current_state = observation

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
        
class CustomEnvWrapperTest(gym.Wrapper):
    def __init__(self, env, env_loc_id, interaction_dir):
        super().__init__(env)
        self.alex_frames = []
        self.interaction_dir = interaction_dir
        self.alex_episode_counter = 0


    def step(self, action):
        #time.sleep(0.1)
        frame = self.env.render()
        self.alex_frames.append(frame)
        return super().step(action)
        
    def reset(self, **kwargs):
        if self.alex_episode_counter != 0:
            # Ensure the directory exists
            os.makedirs(self.interaction_dir, exist_ok=True)
            file_path = os.path.join(self.interaction_dir, f"episode_{self.alex_episode_counter}.mp4")
            # Convert frames to uint8
            frames_uint8 = np.array(self.alex_frames, dtype=np.uint8)
            # Write frames to video file
            with imageio.get_writer(file_path, fps=30) as video_writer:
                for frame in frames_uint8:
                    video_writer.append_data(frame)
        self.alex_episode_counter += 1
        self.alex_frames = []  # Clear frames for the next episode
        return super().reset(**kwargs)


class CustomEnvWrapperTestHuman(gym.Wrapper):
    def __init__(self, env, env_loc_id, interaction_dir, window_position=(0, 0)):
        super().__init__(env)
        self.env_loc_id = env_loc_id
        self.interaction_dir = interaction_dir
        self.window_position = window_position
        self._set_window_position()

    def _set_window_position(self):
        os.environ["SDL_VIDEO_WINDOW_POS"] = f"{self.window_position[0]},{self.window_position[1]}"

    def step(self, action):
        return super().step(action)

    def reset(self, **kwargs):
        return super().reset(**kwargs)
        
class CustomEnvWrapperTestHuman_1ep(gym.Wrapper):
    def __init__(self, env, env_loc_id, interaction_dir):
        super().__init__(env)


    def step(self, action):
        time.sleep(0.12)
        return super().step(action)
        
    def reset(self, **kwargs):
        time.sleep(5)
        return super().reset(**kwargs)



def get_push_env(lateral_friction=1.0,spinning_friction=0.001,mass=1.0):
    def _init():
        env = gymnasium.make('PandaPickAndPlace-v3')
        env.unwrapped.sim.set_lateral_friction('object', -1, lateral_friction=lateral_friction)
        env.unwrapped.sim.set_spinning_friction('object', -1, spinning_friction=spinning_friction)
        block_uid = env.unwrapped.sim._bodies_idx['object']
        env.unwrapped.sim.physics_client.changeDynamics(bodyUniqueId=block_uid, linkIndex=-1, mass=mass)
        # wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward
        # change table's friction
        env.unwrapped.sim.set_lateral_friction('table', -1, lateral_friction=lateral_friction)
        env.unwrapped.sim.set_spinning_friction('table', -1, spinning_friction=spinning_friction)
        block_uid = env.unwrapped.sim._bodies_idx['object']
        print("Info of objects", env.unwrapped.sim.physics_client.getDynamicsInfo(bodyUniqueId=block_uid, linkIndex=-1))
        block_uid = env.unwrapped.sim._bodies_idx['table']
        print("Info of Table", env.unwrapped.sim.physics_client.getDynamicsInfo(bodyUniqueId=block_uid, linkIndex=-1))
        return env
    return _init
    
def get_push_env_test_nondense(lateral_friction=1.0,spinning_friction=0.001,mass=1.0):
    def _init():
        env = gymnasium.make('PandaPickAndPlace-v3', render_mode = 'rgb_array')
        env.unwrapped.sim.set_lateral_friction('object', -1, lateral_friction=lateral_friction)
        env.unwrapped.sim.set_spinning_friction('object', -1, spinning_friction=spinning_friction)
        block_uid = env.unwrapped.sim._bodies_idx['object']
        env.unwrapped.sim.physics_client.changeDynamics(bodyUniqueId=block_uid, linkIndex=-1, mass=mass)
        # wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward
        # change table's friction
        env.unwrapped.sim.set_lateral_friction('table', -1, lateral_friction=lateral_friction)
        env.unwrapped.sim.set_spinning_friction('table', -1, spinning_friction=spinning_friction)
        block_uid = env.unwrapped.sim._bodies_idx['object']
        print("Info of objects", env.unwrapped.sim.physics_client.getDynamicsInfo(bodyUniqueId=block_uid, linkIndex=-1))
        block_uid = env.unwrapped.sim._bodies_idx['table']
        print("Info of Table", env.unwrapped.sim.physics_client.getDynamicsInfo(bodyUniqueId=block_uid, linkIndex=-1))
        return env
    return _init

def get_push_dense_env(lateral_friction=1.0,spinning_friction=0.001,mass=1.0, my_env_index = 1, my_interaction_dir = "./"):
    def _init():
        env = gymnasium.make('PandaPickAndPlaceDense-v3')
        env.unwrapped.sim.set_lateral_friction('object', -1, lateral_friction=lateral_friction)
        env.unwrapped.sim.set_spinning_friction('object', -1, spinning_friction=spinning_friction)
        block_uid = env.unwrapped.sim._bodies_idx['object']
        env.unwrapped.sim.physics_client.changeDynamics(bodyUniqueId=block_uid, linkIndex=-1, mass=mass)
        # wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward
        # change table's friction
        env.unwrapped.sim.set_lateral_friction('table', -1, lateral_friction=lateral_friction)
        env.unwrapped.sim.set_spinning_friction('table', -1, spinning_friction=spinning_friction)
        block_uid = env.unwrapped.sim._bodies_idx['object']
        print("Info of objects", env.unwrapped.sim.physics_client.getDynamicsInfo(bodyUniqueId=block_uid, linkIndex=-1))
        block_uid = env.unwrapped.sim._bodies_idx['table']
        print("Info of Table", env.unwrapped.sim.physics_client.getDynamicsInfo(bodyUniqueId=block_uid, linkIndex=-1))
        env = CustomEnvWrapper(env, env_loc_id=my_env_index, interaction_dir=my_interaction_dir)
        return env
    return _init
    
def get_push_dense_env_test(lateral_friction=1.0,spinning_friction=0.001,mass=1.0, my_env_index = 1, my_interaction_dir = "./"):
    def _init():
        env = gymnasium.make('PandaPickAndPlaceDense-v3', render_mode = 'rgb_array')
        env.unwrapped.sim.set_lateral_friction('object', -1, lateral_friction=lateral_friction)
        env.unwrapped.sim.set_spinning_friction('object', -1, spinning_friction=spinning_friction)
        block_uid = env.unwrapped.sim._bodies_idx['object']
        env.unwrapped.sim.physics_client.changeDynamics(bodyUniqueId=block_uid, linkIndex=-1, mass=mass)
        # wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward
        # change table's friction
        env.unwrapped.sim.set_lateral_friction('table', -1, lateral_friction=lateral_friction)
        env.unwrapped.sim.set_spinning_friction('table', -1, spinning_friction=spinning_friction)
        block_uid = env.unwrapped.sim._bodies_idx['object']
        print("Info of objects", env.unwrapped.sim.physics_client.getDynamicsInfo(bodyUniqueId=block_uid, linkIndex=-1))
        block_uid = env.unwrapped.sim._bodies_idx['table']
        print("Info of Table", env.unwrapped.sim.physics_client.getDynamicsInfo(bodyUniqueId=block_uid, linkIndex=-1))
        env = CustomEnvWrapperTest(env, env_loc_id=my_env_index, interaction_dir=my_interaction_dir)
        return env
    return _init
    

def get_push_dense_env_test_human(lateral_friction=1.0,spinning_friction=0.001,mass=1.0, my_env_index = 1, my_interaction_dir = "./"):
    def _init():
        env = gymnasium.make('PandaPickAndPlaceDense-v3', render_mode = 'human')
        env.unwrapped.sim.set_lateral_friction('object', -1, lateral_friction=lateral_friction)
        env.unwrapped.sim.set_spinning_friction('object', -1, spinning_friction=spinning_friction)
        block_uid = env.unwrapped.sim._bodies_idx['object']
        env.unwrapped.sim.physics_client.changeDynamics(bodyUniqueId=block_uid, linkIndex=-1, mass=mass)
        # wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward
        # change table's friction
        env.unwrapped.sim.set_lateral_friction('table', -1, lateral_friction=lateral_friction)
        env.unwrapped.sim.set_spinning_friction('table', -1, spinning_friction=spinning_friction)
        block_uid = env.unwrapped.sim._bodies_idx['object']
        print("Info of objects", env.unwrapped.sim.physics_client.getDynamicsInfo(bodyUniqueId=block_uid, linkIndex=-1))
        block_uid = env.unwrapped.sim._bodies_idx['table']
        print("Info of Table", env.unwrapped.sim.physics_client.getDynamicsInfo(bodyUniqueId=block_uid, linkIndex=-1))
        env = CustomEnvWrapperTestHuman(env, env_loc_id=my_env_index, interaction_dir=my_interaction_dir)
        return env
    return _init
    
def get_push_dense_env_test_human_1ep(lateral_friction=1.0,spinning_friction=0.001,mass=1.0, my_env_index = 1, my_interaction_dir = "./"):
    def _init():
        env = gymnasium.make('PandaPickAndPlaceDense-v3', render_mode = 'human')
        env.unwrapped.sim.set_lateral_friction('object', -1, lateral_friction=lateral_friction)
        env.unwrapped.sim.set_spinning_friction('object', -1, spinning_friction=spinning_friction)
        block_uid = env.unwrapped.sim._bodies_idx['object']
        env.unwrapped.sim.physics_client.changeDynamics(bodyUniqueId=block_uid, linkIndex=-1, mass=mass)
        # wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward
        # change table's friction
        env.unwrapped.sim.set_lateral_friction('table', -1, lateral_friction=lateral_friction)
        env.unwrapped.sim.set_spinning_friction('table', -1, spinning_friction=spinning_friction)
        block_uid = env.unwrapped.sim._bodies_idx['object']
        print("Info of objects", env.unwrapped.sim.physics_client.getDynamicsInfo(bodyUniqueId=block_uid, linkIndex=-1))
        block_uid = env.unwrapped.sim._bodies_idx['table']
        print("Info of Table", env.unwrapped.sim.physics_client.getDynamicsInfo(bodyUniqueId=block_uid, linkIndex=-1))
        env = CustomEnvWrapperTestHuman_1ep(env, env_loc_id=my_env_index, interaction_dir=my_interaction_dir)
        return env
    return _init

# def get_push_joints_env(lateral_friction=1.0,spinning_friction=0.001,mass=1.0):
#     def _init():
#         env = gymnasium.make('PandaPushJoints-v3')
#         env.unwrapped.sim.set_lateral_friction('object', -1, lateral_friction=lateral_friction)
#         env.unwrapped.sim.set_spinning_friction('object', -1, spinning_friction=spinning_friction)
#         block_uid = env.unwrapped.sim._bodies_idx['object']
#         env.unwrapped.sim.physics_client.changeDynamics(bodyUniqueId=block_uid, linkIndex=-1, mass=mass)
#         # change table's friction
#         env.unwrapped.sim.set_lateral_friction('table', -1, lateral_friction=lateral_friction)
#         env.unwrapped.sim.set_spinning_friction('table', -1, spinning_friction=spinning_friction)
#         block_uid = env.unwrapped.sim._bodies_idx['object']
#         print("Info of objects", env.unwrapped.sim.physics_client.getDynamicsInfo(bodyUniqueId=block_uid, linkIndex=-1))
#         block_uid = env.unwrapped.sim._bodies_idx['table']
#         print("Info of Table", env.unwrapped.sim.physics_client.getDynamicsInfo(bodyUniqueId=block_uid, linkIndex=-1))
#         return env
#     return _init

# def get_push_joints_dense_env(lateral_friction=1.0,spinning_friction=0.001,mass=1.0):
#     def _init():
#         env = gymnasium.make('PandaPushJointsDense-v3')
#         env.unwrapped.sim.set_lateral_friction('object', -1, lateral_friction=lateral_friction)
#         env.unwrapped.sim.set_spinning_friction('object', -1, spinning_friction=spinning_friction)
#         block_uid = env.unwrapped.sim._bodies_idx['object']
#         env.unwrapped.sim.physics_client.changeDynamics(bodyUniqueId=block_uid, linkIndex=-1, mass=mass)
#         # change table's friction
#         env.unwrapped.sim.set_lateral_friction('table', -1, lateral_friction=lateral_friction)
#         env.unwrapped.sim.set_spinning_friction('table', -1, spinning_friction=spinning_friction)
#         block_uid = env.unwrapped.sim._bodies_idx['object']
#         print("Info of objects", env.unwrapped.sim.physics_client.getDynamicsInfo(bodyUniqueId=block_uid, linkIndex=-1))
#         block_uid = env.unwrapped.sim._bodies_idx['table']
#         print("Info of Table", env.unwrapped.sim.physics_client.getDynamicsInfo(bodyUniqueId=block_uid, linkIndex=-1))
#         return env
#     return _init


def get_pick_and_place_env(lateral_friction=1.0,spinning_friction=0.001,mass=1.0):
    def _init():
        env = gymnasium.make('PandaPickAndPlace-v3')
        # env.unwrapped.sim.set_lateral_friction('object', -1, lateral_friction=lateral_friction)
        # env.unwrapped.sim.set_spinning_friction('object', -1, spinning_friction=spinning_friction)
        block_uid = env.unwrapped.sim._bodies_idx['object']
        env.unwrapped.sim.physics_client.changeDynamics(bodyUniqueId=block_uid, linkIndex=-1, mass=mass)
        # change table's friction
        env.unwrapped.sim.set_lateral_friction('table', -1, lateral_friction=lateral_friction)
        env.unwrapped.sim.set_spinning_friction('table', -1, spinning_friction=spinning_friction)
        block_uid = env.unwrapped.sim._bodies_idx['object']
        print("Info of objects", env.unwrapped.sim.physics_client.getDynamicsInfo(bodyUniqueId=block_uid, linkIndex=-1))
        block_uid = env.unwrapped.sim._bodies_idx['table']
        print("Info of Table", env.unwrapped.sim.physics_client.getDynamicsInfo(bodyUniqueId=block_uid, linkIndex=-1))
        return env
    return _init
    
def get_pick_and_place_dense_env(lateral_friction=1.0,spinning_friction=0.001,mass=1.0):
    def _init():
        env = gymnasium.make('PandaPickAndPlaceDense-v3')
        # env.unwrapped.sim.set_lateral_friction('object', -1, lateral_friction=lateral_friction)
        # env.unwrapped.sim.set_spinning_friction('object', -1, spinning_friction=spinning_friction)
        block_uid = env.unwrapped.sim._bodies_idx['object']
        env.unwrapped.sim.physics_client.changeDynamics(bodyUniqueId=block_uid, linkIndex=-1, mass=mass)
        # change table's friction
        env.unwrapped.sim.set_lateral_friction('table', -1, lateral_friction=lateral_friction)
        env.unwrapped.sim.set_spinning_friction('table', -1, spinning_friction=spinning_friction)
        block_uid = env.unwrapped.sim._bodies_idx['object']
        print("Info of objects", env.unwrapped.sim.physics_client.getDynamicsInfo(bodyUniqueId=block_uid, linkIndex=-1))
        block_uid = env.unwrapped.sim._bodies_idx['table']
        print("Info of Table", env.unwrapped.sim.physics_client.getDynamicsInfo(bodyUniqueId=block_uid, linkIndex=-1))
        return env
    return _init

def get_reach_env(lateral_friction=1.0,spinning_friction=0.001,mass=1.0):
    # NO OBJECT!!!!!!!
    def _init():
        env = gymnasium.make('PandaReach-v3')
        # print(env.unwrapped.sim._bodies_idx.keys())
        # dict_keys(['panda', 'plane', 'table', 'target'])
        # env.unwrapped.sim.set_lateral_friction('object', -1, lateral_friction=lateral_friction)
        # env.unwrapped.sim.set_spinning_friction('object', -1, spinning_friction=spinning_friction)
        block_uid = env.unwrapped.sim._bodies_idx['object']
        env.unwrapped.sim.physics_client.changeDynamics(bodyUniqueId=block_uid, linkIndex=-1, mass=mass)
        return env
    return _init

def get_slide_env(lateral_friction=1.0,spinning_friction=0.001,mass=1.0):
    def _init():
        env = gymnasium.make('PandaSlide-v3')
        env.unwrapped.sim.set_lateral_friction('object', -1, lateral_friction=lateral_friction)
        env.unwrapped.sim.set_spinning_friction('object', -1, spinning_friction=spinning_friction)
        block_uid = env.unwrapped.sim._bodies_idx['object']
        env.unwrapped.sim.physics_client.changeDynamics(bodyUniqueId=block_uid, linkIndex=-1, mass=mass)
        # change table's friction
        env.unwrapped.sim.set_lateral_friction('table', -1, lateral_friction=lateral_friction)
        env.unwrapped.sim.set_spinning_friction('table', -1, spinning_friction=spinning_friction)
        block_uid = env.unwrapped.sim._bodies_idx['object']
        print("Info of objects", env.unwrapped.sim.physics_client.getDynamicsInfo(bodyUniqueId=block_uid, linkIndex=-1))
        block_uid = env.unwrapped.sim._bodies_idx['table']
        print("Info of Table", env.unwrapped.sim.physics_client.getDynamicsInfo(bodyUniqueId=block_uid, linkIndex=-1))
        return env
    return _init

def get_stack_env(lateral_friction=1.0,spinning_friction=0.001,mass=1.0):
    def _init():
        env = gymnasium.make('PandaStack-v3')
        env.unwrapped.sim.set_lateral_friction('object1', -1, lateral_friction=lateral_friction)
        env.unwrapped.sim.set_spinning_friction('object1', -1, spinning_friction=spinning_friction)
        env.unwrapped.sim.set_lateral_friction('object2', -1, lateral_friction=lateral_friction)
        env.unwrapped.sim.set_spinning_friction('object2', -1, spinning_friction=spinning_friction)
        block_uid1 = env.unwrapped.sim._bodies_idx['object1']
        env.unwrapped.sim.physics_client.changeDynamics(bodyUniqueId=block_uid1, linkIndex=-1, mass=mass)
        block_uid2 = env.unwrapped.sim._bodies_idx['object2']
        env.unwrapped.sim.physics_client.changeDynamics(bodyUniqueId=block_uid2, linkIndex=-1, mass=mass)
        # change table's friction
        env.unwrapped.sim.set_lateral_friction('table', -1, lateral_friction=lateral_friction)
        env.unwrapped.sim.set_spinning_friction('table', -1, spinning_friction=spinning_friction)
        block_uid = env.unwrapped.sim._bodies_idx['object1']
        print("Info of objects", env.unwrapped.sim.physics_client.getDynamicsInfo(bodyUniqueId=block_uid, linkIndex=-1))
        block_uid = env.unwrapped.sim._bodies_idx['table']
        print("Info of Table", env.unwrapped.sim.physics_client.getDynamicsInfo(bodyUniqueId=block_uid, linkIndex=-1))
        return env
    return _init


