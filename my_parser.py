import argparse
import json

def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-a", "--alpha", default=0.000003, type=float, help="Learning rate - alpha.")
    parser.add_argument("-t", "--train_episodes", default=1000, type=int, help="Training episodes.")
    parser.add_argument("-b", "--batch_size", default=64, type=int, help="Batch size")
    parser.add_argument("-g", "--gamma", default=0.99, type=float, help="Discount factor - gamma.")
    parser.add_argument("--gae_lambda", default=0.95, type=float, help="gae_lambda")
    parser.add_argument("--policy_clip", default=0.2, type=float, help="policy clip")
    parser.add_argument("--n_epochs", default=4, type=int, help="number of epochs")
    parser.add_argument("--learn_every", default=20, type=int, help="N")
    parser.add_argument("--save_every", default=10, type=int, help="N")
    parser.add_argument("--models_dir", default="models/", help="directory in which all models are saved")
    parser.add_argument("--load_model", default="", help="load model from the given path")
    parser.add_argument("--seed", default=None, type=int, help="Random seed.")
    parser.add_argument("--threads", default=2, type=int, help="Maximum number of threads to use.")

    # parser.add_argument("--render_each", default=100, type=int, help="Render some episodes.")
    
    return parser
    
def save_args(args, save_path):
    with open(save_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

def load_args(args, load_path):
    with open(load_path, 'r') as f:
        args.__dict__ = json.load(f)
        