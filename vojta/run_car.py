from os import truncate
import gym
from gym.wrappers import RecordVideo
import numpy as np
from parser import create_parser
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from PPO import PPO
from wrappers import normalize_obs, BoundAction
from datetime import datetime

from gym.wrappers.monitoring.video_recorder import VideoRecorder

if __name__ == '__main__':
    args = create_parser().parse_args([] if "__file__" not in globals() else None)
    
    def make_env():
        # env = gym.make('CarRacing-v2', render_mode="human")
        env = gym.make('CarRacing-v2', render_mode='rgb_array')
        env = normalize_obs(env)
        env = BoundAction(env, low=0, high=1)
        return env

    single_env = make_env()
    env = gym.vector.SyncVectorEnv([lambda: single_env])
    
    ppo = PPO(observation_space = env.observation_space, 
              action_space = env.action_space, 
              entropy_coeff = args.entropy_coeff,
              gamma = args.gamma,
              gae_lambda = args.gae_lambda,
              learning_rate = args.learning_rate)

    ppo.load_w('marek_models/car')
    
    time1 = datetime.now()
    
    done = False
    obs, _ = env.reset()
    step = 0
    score = 0
    while not done:        
        # print(single_env.render())
        value, mu, sigma = ppo.model(obs)
        action, _, _ = ppo.act(obs)
        # print(action, mu, sigma)
        obs_, reward, terminated, truncated, _ = env.step(action.numpy())
        done = terminated or truncated
        score += reward
        obs = obs_
        # print(f"{obs=}")
        step += 1
    print(f"{score = }")
    print(f"time spent = {datetime.now() - time1}")
