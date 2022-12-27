import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
import json
import tensorflow as tf
from time import sleep
import random
from my_parser import *

def get_logger(model_dir, tensorboard=False):
    """
    tensorboard --logdir='/your/path/here'
    """
    if tensorboard:
        tensorboard_log_dir = model_dir if model_dir[-1]=="/" else model_dir + "/"
        summary_writer = tf.summary.create_file_writer(tensorboard_log_dir)
        return summary_writer
    return None

def print_info(env, args):
    print("==== ENV INFO ====")
    print(f"{env.action_space=}\n{env.observation_space=}")
    print("==================")
    print("====== ARGS ======")
    print(args.__dict__)
    print("==================")

def get_curr_datetime(microsecond=False):
    if microsecond:
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
def add_curr_time_to_dir(dir, microsecond=False):
    return os.path.join(dir, get_curr_datetime(microsecond))

def create_dir_for_curr_runtime(models_dir):
    """adds a subfolder in the models directory for this runtime

    Args:
        models_dir (string): path to directory with models

    Returns:
        string: models_dir with a new current datetime subfolder
    """
    sleep(random.random()) # so that the parallel executions dont save into the same folder
    models_dir = add_curr_time_to_dir(models_dir, microsecond=True)
    os.makedirs(models_dir)
    print(f"This runtime saved in {models_dir}")
    return models_dir

def save_pltgraph(avg_score_history, chkpt_dir, e, start_from_ep):
    fig_name = os.path.join(chkpt_dir, "mean_returns")
    
    plt.plot(
        np.linspace(start_from_ep, e, len(avg_score_history)), 
        avg_score_history)
    plt.title(f"Average Return for last 300 episodes")
    plt.xlabel("episode")
    plt.ylabel("average return")
    plt.savefig(f"{fig_name}.png")
    plt.clf()
    
def save_model_summary(model_dir, model):
    filename = os.path.join(model_dir, "model_summary.txt")
    with open(filename,'w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
        model.get_layer('CNN_model').summary(print_fn=lambda x: fh.write(x + '\n'))
        # model.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))

def create_subfolder(models_dir, addon=""):
    sleep(random.random()) # so that the parallel executions dont save into the same folder
    models_dir = os.path.join(models_dir, addon)
    os.makedirs(models_dir)
    print(f"This runtime saved in {models_dir}")
    return models_dir

################ OLD

def graph_make_save(env, args, curr_e, dir):
    returns = env._mean_episode_returns
    plt.plot(np.linspace(0, curr_e, len(returns)), returns)
    plt.title(f"batch{args.batch_size}_g{args.gamma}_a{args.learning_rate}")
    plt.xlabel("episode")
    plt.ylabel("Mean Return for last 100 episodes")
    fig_name = os.path.join(dir, "mean_returns")
    plt.savefig(f"{fig_name}.png")
    plt.clf()

# def plot_learning_curve(x, returns, figure_file):
#     """
#     @args 
#         figure_file ... path where to save it
#         returns
#     """
#     running_avg = np.zeros(len(returns))
#     for i in range(len(running_avg)):
#         running_avg[i] = np.mean(returns[max(0, i-100):(i+1)])
#     plt.plot(x, running_avg)
#     plt.title('Running average of previous 100 returns')
#     plt.savefig(figure_file)
#     plt.clf()
#     print(f"saving running_avg to {figure_file}")
        
def save_checkpoint(network, target_network, env, args, curr_episode=0):
    ep = str(curr_episode) if curr_episode != 0 else ''
    dir = os.path.join(args.models_dir, get_curr_datetime() + 'ep' + ep)
    os.makedirs(dir)
    print("==================")
    print(f"=> ep {len(env._episode_returns)} : SAVING ALL TO {dir}")
    
    if curr_episode == 0:
        curr_episode = args.trainfor
    graph_make_save(env, args, curr_episode, dir)
    
    network.save_weights(os.path.join(dir, 'NN/'))
    target_network.save_weights(os.path.join(dir, 'baselineNN/'))
    
    save_args(args, os.path.join(dir, 'args'))
    print("==================")
    