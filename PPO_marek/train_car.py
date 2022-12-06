# Python Libraries
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default
import gym
import numpy as np
import argparse
import tensorflow as tf

# My Libraries
from PPO import PPO
from wrappers import LuminanceWrapper, StackObservation, normalize_obs, BoundAction
from my_parser import create_parser, save_args
from utils import *

def main(env, args: argparse.Namespace) -> None:
    if args.seed is not None:
        tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    
    # create a specific folder for this training (usefull for parallel execution)
    # args.models_dir = create_dir_for_curr_runtime(args.models_dir)
    args.models_dir = create_subfolder(args.models_dir, f"horizon_{args.steps_per_ep}")

    ppo = PPO(observation_space = env.observation_space, 
              action_space = env.action_space, 
              entropy_coeff = args.entropy_coeff,
              gamma = args.gamma,
              gae_lambda = args.gae_lambda,
              learning_rate = args.learning_rate,
              value_fun_coeff = args.vf_coeff)
    
    if args.load_model != "":
        ppo.load_w(args.load_model)
    
    # TODO automate this
    def lr_schedule(x): return x * args.learning_rate
    # def lr_schedule(x): return x
    
    logger = get_logger(args.models_dir, args.tensorboard)
    with logger.as_default():
        tf.summary.text('arguments', str(args.__dict__), step=1)
        
    save_args(args, os.path.join(args.models_dir, 'args'))
    save_model_summary(args.models_dir, ppo.get_model())
    
    # ppo.model_summary()
    
    ppo.train(env = env,
              args = args, 
              nepisodes = args.training_episodes, 
              steps_per_ep = args.steps_per_ep, 
              mb_size = args.batch_size, 
              epochs_per_ep = args.epochs_per_ep,
              lr = lr_schedule, 
              clip_range = args.clip_range, 
              model_dir = args.models_dir, 
              start_from_ep = args.start_from_ep, 
              save_interval = args.save_every,
              print_freq = args.print_ep_info_freq,
              logger = logger)


if __name__ == '__main__':    
    args = create_parser().parse_args([] if "__file__" not in globals() else None)

    env = gym.vector.make('CarRacing-v2', num_envs=args.num_envs,
                          wrappers=[normalize_obs, BoundAction])
    
    print_info(env, args)
    
    main(env, args)
