from gym import ObservationWrapper, spaces, ActionWrapper
import gym
import numpy as np
from collections import deque
# import cv2

class ClippedAction(ActionWrapper):
    def __init__(self, env:gym.Env, low=0, high=1):
        super().__init__(env)
        self.old_low = env.action_space.low
        self.old_high = env.action_space.high
        shape = env.action_space.shape
        self.action_space = spaces.Box(low=low, high=high, shape=shape)

    def action(self, action):
        l, h = self.action_space.low, self.action_space.high
        L, H = self.old_low, self.old_high

        return (action-l)/(h-l)*(H-L)+L

class NormalizeObservation(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        low = np.mean(self.observation_space.low)
        high = np.mean(self.observation_space.high)
        shape = env.observation_space.shape
        self.observation_space = spaces.Box(low, high, shape=shape, dtype='float32')

    @staticmethod
    def convert_obs(obs):
        return ((obs-127.5)/127.5).astype('float32')

    def observation(self, obs):
        return NormalizeObservation.convert_obs(obs)

class expand_dim_obs(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        shape = tuple([1] + list(self.observation_space.shape))
        low = self.observation_space.low
        high = self.observation_space.high
        self.observation_space = spaces.Box(low, high, shape=shape, dtype='float32')

    def observation(self, obs):
        return np.expand_dims(obs, axis=0)
    
# class expand_dim_act(ActionWrapper):
#     def __init__(self, env:gym.Env):
#         super().__init__(env)
#         self.action_space.shape = tuple([1] + list(self.action_space.shape))
        
#     def action(self, action):
#         return np.expand_dims(action, axis=0)