import gym
import numpy as np

if __name__ == '__main__':
    from PPO import PPO
    from wrappers import LuminanceWrapper, StackObservation, normalize_obs, BoundAction

    env = gym.vector.make('CarRacing-v2', num_envs=6,
                          wrappers=[normalize_obs, BoundAction])
    ppo = PPO(env.observation_space, env.action_space, entropy_coeff=0.001)
    # ppo.load('models/car')
    def lr_schedule(x): return x * 7e-4
    # lr_schedule = lambda x: 0.0
    ppo.train(env, nepisodes=2000, steps_per_ep=2048, mb_size=512, epochs_per_ep=4,
              lr=lr_schedule, clip_range=0.2, model_name='Vmodels/car', start_from_ep=0)
