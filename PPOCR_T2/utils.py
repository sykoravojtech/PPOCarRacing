import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from my_parser import save_args
import os


def plot_learning_curve(x, returns, figure_file):
    """
    @args 
        figure_file ... path where to save it
        returns
    """
    running_avg = np.zeros(len(returns))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(returns[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 returns')
    plt.savefig(figure_file)
    print(f"saving running_avg to {figure_file}")
    
    
def print_info(env, args):
    print("==== ENV INFO ====")
    print(f"{env.action_space=}\n{env.observation_space=}\n{env.reward_range=}")
    print("==================")
    print("====== ARGS ======")
    print(args.__dict__)
    print("==================")
    
def get_curr_datetime(microsecond=False):
    if microsecond:
        return datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    
def add_curr_time_to_dir(dir, microsecond=False):
    return os.path.join(dir, get_curr_datetime(microsecond))
    
def save_all(returns, agent, args, curr_episode=0):
    ep = str(curr_episode) if curr_episode != 0 else ''
    dir = os.path.join(args.models_dir, get_curr_datetime() + '_ep' + ep)
    print("==================")
    print(f"=> ep {len(returns)} : SAVING ALL TO {dir}")
    os.makedirs(dir)
    x = [i for i in range(1, len(returns) + 1)]
    plot_learning_curve(x, returns, os.path.join(dir, 'reward_history'))
    agent.save_models(
        actor_path = os.path.join(dir, 'actor'),
        critic_path = os.path.join(dir, 'critic')
        )
    save_args(args, os.path.join(dir, 'args'))
    print("==================")
    
