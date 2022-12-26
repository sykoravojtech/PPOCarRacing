import os
# Report only TF errors by default
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from os import truncate
import gym
from gym.wrappers import RecordVideo
import numpy as np

from my_parser import create_parser
from PPO import PPO
from wrappers import NormalizeObservation, ClippedAction
from utils import print_info

from gym.wrappers.monitoring.video_recorder import VideoRecorder

RENDER = False
MODEL_PATH = "BEST/ep1330/weights"

if __name__ == '__main__':
    args = create_parser().parse_args([] if "__file__" not in globals() else None)
    
    def make_env():
        if RENDER:
            env = gym.make('CarRacing-v2', render_mode="human")
        else:
            env = gym.make('CarRacing-v2', render_mode='rgb_array')
            
        env = NormalizeObservation(env)
        env = ClippedAction(env, low=0, high=1)
        return env

    single_env = make_env()
    env = gym.vector.SyncVectorEnv([lambda: single_env])
    # print(f"obs shape {env.observation_space.shape}\nact space {env.action_space.shape}")
    
    # env = gym.make('CarRacing-v2', render_mode='rgb_array')
    # env = gym.vector.make('CarRacing-v2', num_envs=1,
    #                       wrappers=[normalize_obs, BoundAction])
    # env = gym.wrappers.RecordVideo(env, "recording")
    
    print_info(single_env, args)
    
    ppo = PPO(observation_space = env.observation_space, 
              action_space = env.action_space, 
              entropy_coeff = args.entropy_coeff,
              gamma = args.gamma,
              gae_lambda = args.gae_lambda,
              learning_rate = args.learning_rate,
              value_fun_coeff = args.vf_coeff)

    
    ppo.load_weights(MODEL_PATH)
    
    ppo.run(
        env,
        single_env, 
        num_of_episodes = 50,
        render = RENDER
        )
