import gym
import numpy as np
from agent import Agent
from utils import *
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import tensorflow as tf
from random import random
from time import sleep

from my_parser import create_parser

# tf.get_logger().setLevel('INFO') 
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# conda activate newest


if __name__ == '__main__':
    args = create_parser().parse_args([] if "__file__" not in globals() else None)
    
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    
    sleep(random()) # so that the parallel executions dont save into the same folder
    args.models_dir = add_curr_time_to_dir(args.models_dir, microsecond=True)
    os.makedirs(args.models_dir)
    print(f"==== dir = {args.models_dir}")
    
    env = gym.make('CarRacing-v2', continuous=False)
    print_info(env, args)
    
    agent = Agent(n_actions = env.action_space.n, 
                  input_dims = env.observation_space.shape,
                  gamma = args.gamma,
                  alpha = args.alpha, 
                  gae_lambda = args.gae_lambda,
                  policy_clip = args.policy_clip,
                  batch_size = args.batch_size,
                  n_epochs = args.n_epochs
                  )
    
    print("loading models")
    agent.load_models(os.path.join("bestmodel", "actor"), os.path.join("bestmodel", "critic"))
    
    # best_score = env.reward_range[0]
    return_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0
    
    for episode in range(1, 10):
        observation, _ = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation.astype(np.int32))
            observation_, reward, done, truncated, info = env.step(action)
            done  = done or truncated
            n_steps += 1
            score += reward
            observation = observation_
        return_history.append(score)
        avg_score = np.mean(return_history[-100:])
        print('ep=', episode, ', ret=%.1f' % score, ', avg ret=%.1f' % avg_score,
              ', n_steps=', n_steps, ', learn_n=', learn_iters)



    # for episode in range(1, args.train_episodes + 1):
    #     observation, _ = env.reset()
    #     done = False
    #     score = 0
    #     while not done:
    #         action, prob, val = agent.choose_action(observation.astype(np.int32))
    #         observation_, reward, done, truncated, info = env.step(action)
    #         done  = done or truncated
    #         n_steps += 1
    #         score += reward
    #         agent.store_transition(observation, action,
    #                                prob, val, reward, done)
    #         if n_steps % args.learn_every == 0:
    #             agent.learn()
    #             learn_iters += 1
    #         observation = observation_
    #     return_history.append(score)
    #     avg_score = np.mean(return_history[-100:])

    #     # if avg_score > best_score:
    #     #     best_score = avg_score
    #     #     agent.save_models()

    #     print('ep=', episode, ', ret=%.1f' % score, ', avg ret=%.1f' % avg_score,
    #           ', n_steps=', n_steps, ', learn_n=', learn_iters)
        
    #     if episode % args.save_every == 0:
    #         save_all(return_history, agent, args, episode)
        
    save_all(return_history, agent, args)
