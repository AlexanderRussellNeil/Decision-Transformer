from stable_baselines3 import HerReplayBuffer
from sb3_contrib import ARS, QRDQN, TQC, TRPO, RecurrentPPO
import gymnasium
import panda_gym
import numpy as np
import time
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from gymnasium import spaces
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    unwrap_vec_normalize,
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
import os


from model import TQCEnvSwitchWrapper
from utils import get_push_dense_env, get_push_dense_env_test, get_push_dense_env_test_human

import argparse
import numpy
import warnings


def train(args):

    warnings.filterwarnings("ignore")
    env_id = args.domain_name
    # log_dir = './panda_push_v3_tensorboard/'
    log_dir = './ARCHIEVE/LOGS/' + args.domain_name + '_tensorboard_tqc/'

    file_path = os.getcwd()
    interaction_dir = os.path.join(file_path, "ARCHIEVE/INTERACTIONS/")

    env_int = numpy.random.randint(low=args.random_int[0], high=args.random_int[1], size=4, dtype='l')

    # env for train
    env1 = get_push_dense_env(1.0, 0.001, env_int[0], 1, interaction_dir)
    env2 = get_push_dense_env(1.0, 0.001, env_int[1], 2, interaction_dir)
    env3 = get_push_dense_env(1.0, 0.001, env_int[2], 3, interaction_dir)
    env4 = get_push_dense_env(1.0, 0.001, env_int[3], 4, interaction_dir)
    # env for test
    env5 = get_push_dense_env(1.0, 0.001, args.test_mass)

    train_env = DummyVecEnv([env1,env2,env3,env4])
    test_env = DummyVecEnv([env5,env5,env5,env5])

    model = TQCEnvSwitchWrapper(policy = "MultiInputPolicy",
                            env = train_env,
                            batch_size=2048,
                            gamma=0.95,
                            learning_rate=1e-4,
                            train_freq=64,
                            gradient_steps=64,
                            tau=0.05,
                            replay_buffer_class=HerReplayBuffer,
                            replay_buffer_kwargs=dict(
                                n_sampled_goal=4,
                                goal_selection_strategy="future",
                            ),
                            policy_kwargs=dict(
                                net_arch=[512, 512, 512],
                                n_critics=10,
                            ),
                            learning_starts = 1000,
                            verbose=2,
                            tensorboard_log=log_dir)
    
    start_time = time.time()

    for i in range(100):
        model.learn(total_timesteps=args.time_step, reset_num_timesteps=False, progress_bar=True)
        model_name = "./ARCHIEVE/MODELS/TQC-" + str(args.domain_name + "-test-" + str(args.test_mass) + "h" + str(round(time.time() - start_time/3600,2)))
        model.save(model_name)
    train_env.close()

    
    model.save_replay_buffer('TQC-' + str(args.domain_name) + '-buffer' + "-test-" + str(args.test_mass))

    test_env.close()


def test():
    
    file_path = os.getcwd()
    interaction_dir = os.path.join(file_path, "VIDEOS/TQC/")
    
    env5 = get_push_dense_env_test(1.0, 0.001, 20, 5, interaction_dir)

    test_env = DummyVecEnv([env5])

    model = TQCEnvSwitchWrapper.load('TQC-PandaPickAndPlaceDense-v3',env=test_env)
    
    test_mean_reward, test_std_reward = evaluate_policy(model, test_env, 100)
    
    print(f"Test Mean reward = {test_mean_reward:.2f} +/- {test_std_reward:.2f}")

    test_env.close()

def test_human_view():
    
    file_path = os.getcwd()
    interaction_dir = os.path.join(file_path, "interactions/TQC/")
    
    env5 = get_push_dense_env_test_human(1.0, 0.001, 20, 5, interaction_dir)

    test_env = DummyVecEnv([env5])
    
    arch_path = os.getcwd() +  "/ARCHIEVE/MODELS/"

    model = TQCEnvSwitchWrapper.load(arch_path + 'TQC-PandaPickAndPlaceDense-v3',env=test_env)
    
    test_mean_reward, test_std_reward = evaluate_policy(model, test_env, 100)
    
    print(f"Test Mean reward = {test_mean_reward:.2f} +/- {test_std_reward:.2f}")

    test_env.close()

def retrain():
    env5 = get_push_dense_env(1.0, 0.001,20)
    test_env = DummyVecEnv([env5,env5,env5,env5])

    model = TQCEnvSwitchWrapper.load('TQC-PandaPush-v3',env=test_env)
    model.eval_env = True

    model.learn(total_timesteps=2_000_000,progress_bar=True)

    test_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain_name', default='PandaPickAndPlaceDense-v3')
    parser.add_argument('--random_int', default=[1, 2, 3, 4], nargs='+', type=int)
    parser.add_argument('--test_mass', default=1000, type=int)
    parser.add_argument('--time_step', default=204800, type=int)
    args = parser.parse_args()

    train(args)
    #test_human_view()
