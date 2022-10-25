import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, figure_file):
    """
    @args 
        figure_file ... path where to save it
        scores
    """
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)
    print(f"saving running_avg to {figure_file}")
    
    
def print_env_info(env):
    print("==== ENV INFO ====")
    print(f"{env.action_space=}\n{env.observation_space=}\n{env.reward_range=}")
    print("==================")