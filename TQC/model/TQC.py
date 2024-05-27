from stable_baselines3 import HerReplayBuffer, SAC
from sb3_contrib import ARS, QRDQN, TQC, TRPO, RecurrentPPO
import gymnasium
import panda_gym
import numpy as np
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from gymnasium import spaces
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    unwrap_vec_normalize,
)

class TQCEnvSwitchWrapper(TQC):
    def __init__(self,**args):
        super().__init__(**args)
        self.eval_env = False
        #self.my_counter = 0

    # 重新设置环境
    def reset_env(self,env):
        # reset env
        env = self._wrap_env(env, self.verbose, monitor_wrapper=True)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.n_envs = env.num_envs
        self.env = env

        # get VecNormalize object if needed
        self._vec_normalize_env = unwrap_vec_normalize(env)
        
        if issubclass(self.replay_buffer_class, HerReplayBuffer):
            self.replay_buffer.set_env(env)
        # # Make a local copy as we should not pickle
        # # the environment when using HerReplayBuffer
        # replay_buffer_kwargs = self.replay_buffer_kwargs.copy()
        # if issubclass(self.replay_buffer_class, HerReplayBuffer):
        #     assert self.env is not None, "You must pass an environment when using `HerReplayBuffer`"
        #     replay_buffer_kwargs["env"] = self.env
        # self.replay_buffer = self.replay_buffer_class(
        #     self.buffer_size,
        #     self.observation_space,
        #     self.action_space,
        #     device=self.device,
        #     n_envs=self.n_envs,
        #     optimize_memory_usage=self.optimize_memory_usage,
        #     **replay_buffer_kwargs,  # pytype:disable=wrong-keyword-args
        # )
        
        self.eval_env = True
    
    # 切换环境时 变为不是随机采样
    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param n_envs:
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        # Select action randomly or according to policy
        if not self.eval_env:
            if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
                # Warmup phase
                unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
            else:
                # Note: when using continuous actions,
                # we assume that the policy uses tanh to scale the action
                # We use non-deterministic action in the case of SAC, for TD3, it does not matter
                unscaled_action, _ = self.predict(self._last_obs, deterministic=False)
        else:
            unscaled_action, _ = self.predict(self._last_obs, deterministic=False)
        
        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        #array = np.array(list(self._last_obs.values()))
        #print("LAST OBS: ", array.shape, " AND ACTIONS???? : ", action.shape)
        #print("C:", self.my_counter ,"TYYYYYYYYYYYYYYYYYYPPPPPPPPPPPPPOOOOOOO: ", self._last_obs)
        #self.my_counter += 1
        return action, buffer_action
