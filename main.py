import gym
import numpy as np
from agent import Agent
from utils import plot_learning_curve, print_env_info

# conda activate newest


if __name__ == '__main__':
    env = gym.make('CarRacing-v2', continuous=False)
    print_env_info(env)
    
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size,
                  alpha=alpha, n_epochs=n_epochs,
                  input_dims=env.observation_space.shape)
    
    print("loading models")
    agent.load_models()
    
    figure_file = 'plots/CarRacing20.png'

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    n_episodes = 10
    for i in range(1, n_episodes + 1):
        observation, _ = env.reset()
        done = False
        score = 0
        while not done:
            # print("choosing action")
            action, prob, val = agent.choose_action(observation)
            # print("making step")
            observation_, reward, done, truncated, info = env.step(action)
            done  = done or truncated
            n_steps += 1
            score += reward
            # print("storing transition")
            agent.store_transition(observation, action,
                                   prob, val, reward, done)
            if n_steps % N == 0:
                # print(f"agent.learn() {n_steps=}")
                agent.learn()
                # print("learned")
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        # print(f"{avg_score=} >? {best_score=}")
        if avg_score > best_score:
            best_score = avg_score
            
            agent.save_models()

        print('episode=', i, ', score=%.1f' % score, ', avg score=%.1f' % avg_score,
              ', time_steps=', n_steps, ', learning_steps=', learn_iters)
              
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)