import argparse
import json

def create_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-entropy", "--entropy_coeff", default=0.01, type=float, help="Entropy coefficient (c2)") # 0.0
    parser.add_argument("-vf", "--vf_coeff", default=1, type=float, help="Value function coefficient (c1)") # 0.5 - 1
    parser.add_argument("-g", "--gamma", default=0.99, type=float, help="Discount factor - gamma.")
    parser.add_argument("-gae", "--gae_lambda", default=0.95, type=float, help="gae_lambda") # 0.9-1
    parser.add_argument("-lr", "--learning_rate", default=2.5e-4, type=float, help="Learning rate - alpha.")
    parser.add_argument("--constant_lr", default=False, action="store_true", help="Constant or discounted learning rate")
    # parser.add_argument("--lr_discount", default=7e-6, type=float, help="Learning rate discount factor")
    parser.add_argument("-t", "--training_episodes", default=4000, type=int, help="Training episodes.")
    parser.add_argument("-s", "-horizon", "--steps_per_ep", default=2250, type=int, help="Steps per episode - since we have multiple envs we don't end when done")
    parser.add_argument("-b", "--batch_size", default=128, type=int, help="Batch/minibatch size")
    parser.add_argument("-epochs", "--epochs_per_ep", default=3, type=int, help="Number of epochs per episode")
    parser.add_argument("-clip", "--clip_range", default=0.1, type=float, help="Clip range (1-clip, 1+clip)")
    parser.add_argument("-dir", "--models_dir", default="models/", help="Directory in which all models are saved")
    parser.add_argument("--start_from_ep", default=1, type=int, help="Starting episode number")
    parser.add_argument("--save_every", default=10, type=int, help="Save the model every N episodes. Make a checkpoint.")
    parser.add_argument("--seed", default=None, type=int, help="Random seed.")
    parser.add_argument("--threads", default=128, type=int, help="Maximum number of threads to use.")
    parser.add_argument("-load", "--load_model", default="", help="Load model from the given path")
    parser.add_argument("-log", "--tensorboard", default=True, action="store_false", help="Tensorboard logging")
    parser.add_argument("-print_freq", "--print_ep_info_freq", default=1, type=int, help="Each Nth episide print info about episode")
    parser.add_argument("--num_envs", default=6, type=int, help="Number of parallel environments")
    # parser.add_argument("--render_each", default=100, type=int, help="Render some episodes.")


    # parser.add_argument("--learn_every", default=20, type=int, help="N")
    # parser.add_argument("-e", "--epsilon", default=0.3, type=float, help="Exploration factor.")
    # parser.add_argument("--epsilon_final", default=0.001, type=float, help="Final exploration factor.")
    # parser.add_argument("--epsilon_final_at", default=2500, type=int, help="Training episodes.")
    return parser
    
def save_args(args, save_path):
    with open(save_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

def load_args(args, load_path):
    with open(load_path, 'r') as f:
        args.__dict__ = json.load(f)
        