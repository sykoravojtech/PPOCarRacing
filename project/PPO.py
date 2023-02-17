import os
# Report only TF errors by default
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from collections import defaultdict

import tensorflow as tf
from tensorflow import keras
from keras.models import Model
import tensorflow_probability as tfp

import numpy as np

from actorcritic import get_ActorCritic_model
from utils import save_pltgraph

# import random

class PPO:
    def __init__(self, observation_space, action_space, entropy_coeff, gamma, gae_lambda, learning_rate, value_fun_coeff):
        self.gamma : float = gamma
        self.gae_lambda : float = gae_lambda
        self.model: Model = None
        self.action_space = action_space
        self.optim = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.entropy_coeff : float = entropy_coeff
        self.get_probdist = self.get_beta_probdist
        self.vf_coeff : float = value_fun_coeff

        if len(observation_space.shape) == 4:
            print("... building conv model ...")
            # passing just the pure one observation shape
            self.model = get_ActorCritic_model(
                observation_space.shape[1:], action_space)
        else:
            raise Exception(
                f'Unsupported observation space shape {observation_space.shape} ... only 4 is supported')

    def save_weights(self, filename='model'):
        # TODO remake for h5 version where weights are in one file
        self.model.save_weights(filename)

    def load_weights(self, filename='model'):
        self.model.load_weights(filename)
        
    def get_model(self):
        return self.model

    def model_summary(self):
        self.model.summary()
        self.model.get_layer("CNN_model").summary()

    @tf.function
    def choose_action(self, state):
        prob_dist, values = self.get_probdist(state)
        action = prob_dist.sample()
        return action, values, prob_dist.log_prob(action)

    def get_values(self, state):
        return self.get_probdist(state)[1]

    def get_beta_probdist(self, state):
        value, alpha, beta = self.model(state)
        prob_dist = tfp.distributions.Independent(
            tfp.distributions.Beta(alpha, beta),
            reinterpreted_batch_ndims=1)
        return prob_dist, tf.squeeze(value, axis=-1)

    @tf.function
    def get_loss_policy_clipped(self, advantage, logprobs, old_logprobs, clip):
        ratio = tf.math.exp(logprobs - old_logprobs)
        clipped_adv = tf.clip_by_value(ratio, 1 - clip, 1 + clip) * advantage
        loss_policy = -tf.reduce_mean(tf.minimum(ratio*advantage, clipped_adv))
        return loss_policy

    @tf.function
    def get_loss_critic_value(self, pred_value, returns):
        return tf.reduce_mean((pred_value - returns)**2)

    # @tf.function # gives an error
    def gradient(self, states, actions, returns, values, clip, old_logprobs):
        eps = 1e-8
        advantages = returns - values
        advantages = (advantages 
                      - tf.reduce_mean(advantages) / (tf.keras.backend.std(advantages) + eps))

        with tf.GradientTape() as tape:
            prob_dist, predicted_vals = self.get_probdist(states)
            logprobs = prob_dist.log_prob(actions)
            loss_policy_clipped = self.get_loss_policy_clipped(advantages, logprobs, old_logprobs, clip)
            loss_critic_value = self.get_loss_critic_value(predicted_vals, returns)
            entropy = prob_dist.entropy()

            # main loss function of PPO
            loss = loss_policy_clipped + loss_critic_value*self.vf_coeff - entropy * \
                self.entropy_coeff  
                
        # with tf.compat.v1.Session() as sess:
        #     a = tf.print(loss[0], loss_policy_clipped, loss_critic_value*self.vf_coeff, entropy[0] *self.entropy_coeff)
        #     print(f"\tLOSS = {a}")

        approx_kldiv = 0.5 * tf.reduce_mean(tf.square(old_logprobs-logprobs))

        vars = tape.watched_variables()
        grads = tape.gradient(loss, vars)
        
        return zip(grads, vars), loss_critic_value, loss_policy_clipped, entropy, approx_kldiv

    @tf.function
    def learn_on_single_batch(self, clip, lr, states, returns, actions, values, old_logprobs):
        states = tf.keras.backend.cast_to_floatx(states)
        grads_and_vars, loss_v, loss_policy_clipped, entropy, approx_kldiv = self.gradient(
            states, actions, returns, values, clip, old_logprobs)
        self.optim.learning_rate = lr
        self.optim.apply_gradients(grads_and_vars)

        return loss_policy_clipped, loss_v, entropy, approx_kldiv

    def get_returns(self, rewards, values, dones, last_values, last_dones, dtype):
        timesteps = len(rewards)
        num_of_envs = rewards[0].shape[0]
        advantages = np.zeros((timesteps, num_of_envs))
        curr_adv = 0
        
        for step in reversed(range(timesteps)):
            if step != timesteps-1:
                next_values = values[step+1]
                next_non_terminal = 1.0 - dones[step+1]
            else:
                next_values = last_values
                next_non_terminal = 1.0 - last_dones

            delta = rewards[step] + self.gamma * \
                next_values * next_non_terminal - values[step]
                
            curr_adv = delta + self.gamma * self.gae_lambda * next_non_terminal * curr_adv
            advantages[step] = curr_adv
            
        returns = (advantages + values).astype(dtype)
        return returns

    def learn(self, state, returns, actions, values, old_logprobs, clip, lr, batch_size, epochs):
        losses = defaultdict(list)
        num_batches = state.shape[0]
        indxs = np.arange(num_batches)
        for _ in range(epochs):
            np.random.shuffle(indxs)
            for start in range(0, num_batches, batch_size):
                end = start + batch_size
                batch_i = indxs[start:end]
                slices = (tf.constant(sravo[batch_i]) for sravo in (
                    state, returns, actions, values, old_logprobs))
                loss = self.learn_on_single_batch(
                    clip, lr, *slices)
                for k, v in zip(['policy_loss', 'value_loss', 'entropy', 'kl'], loss):
                    losses[k].append(v)

        return losses

    def train(
        self, env, args, num_of_episodes, steps_per_ep, epochs_per_ep,
        batch_size, clip_range, lr, save_interval,
        models_dir, starting_episode, print_freq, logger
    ):

        print(f"^^^^^ TRAINING for {num_of_episodes-starting_episode+1} episodes ^^^^^")
        dtype = tf.keras.backend.floatx()
        dones = np.zeros((env.num_envs,), dtype=dtype)
        overall_scores = np.zeros_like(dones)
        
        state, _ = env.reset()
        
        score_history = []
        avg_score_history = []

        step = 0
        for ep in range(starting_episode, num_of_episodes + 1):
            
            # initialize batches
            batch_states, batch_rewards, batch_actions, batch_values, batch_dones, batch_logprobs = [], [], [], [], [], []
            for _ in range(steps_per_ep):
                step += 1
                
                # choose action
                actions, values, logprobs = self.choose_action(state)
                actions = actions.numpy()

                # add newly obtained data to batches
                batch_states.append(state)
                batch_actions.append(actions)
                batch_values.append(values.numpy())
                batch_dones.append(dones)
                batch_logprobs.append(logprobs.numpy())

                # use the chosen action
                state, rewards, terminated, truncated, _ = env.step(actions)
                
                dones = terminated | truncated
                # if np.any(dones):
                #     print(f"{dones = }")
                dones = tf.cast(dones, tf.float32)

                # sort out rewards
                batch_rewards.append(rewards)
                overall_scores += rewards
                for i in range(rewards.shape[0]):  # for each env in vector env
                    if dones[i]:
                        # print("appending")
                        score_history.append(overall_scores[i])
                        # print(f"{score_history = }")
                        overall_scores[i] = 0

            final_values = self.get_values(state).numpy()

            # calculates the returns from this episode using advantage and values
            returns = self.get_returns(
                batch_rewards, batch_values, batch_dones, final_values, dones, dtype=dtype)

            # reshaping
            batch_states = np.concatenate(batch_states, axis=0)
            returns = np.concatenate(returns, axis=0)
            batch_actions = np.concatenate(batch_actions, axis=0)
            batch_values = np.concatenate(batch_values, axis=0)
            batch_logprobs = np.concatenate(batch_logprobs, axis=0)

            # choice between constant and decaying learning rate
            if args.constant_lr:
                lr_now = lr(1.0)
            else:
                lr_now = lr(1.0 - ep/num_of_episodes)

            # learn from the obtained batch of experiences
            self.learn(batch_states, returns, batch_actions, batch_values, batch_logprobs,
                       clip_range, lr_now, batch_size, epochs=epochs_per_ep)
            
            # calculate the mean score for a smooth graph for comparing
            avg_score = np.mean(score_history[-300:] if len(score_history) > 300 else score_history)
            avg_score_history.append(avg_score)

            # print information about the episode
            if ep % print_freq == 0:
                print(f'==> episode: {ep}/{num_of_episodes} step={step}, avg score: {avg_score:.3f}')

            # save weights to be able to load the model later
            if ep % save_interval == 0:
                chkpt_dir = os.path.join(models_dir, f"ep{ep}")
                weights_dir = os.path.join(chkpt_dir, "weights")
                print(f"    ... Saving model ep={ep} ...")
                self.save_weights(weights_dir)
                save_pltgraph(avg_score_history, chkpt_dir, ep, starting_episode)

            # tensorboard for clear overview of average score over all episodes
            if args.tensorboard:
                with logger.as_default():
                    tf.summary.scalar('average score', avg_score, step=step)
                    tf.summary.scalar('learning rate', lr_now, step=step)
                    tf.summary.scalar('episode', ep, step=step)

        # end of all episodes
        self.save_weights(os.path.join(models_dir, "FINAL"))

    def run(self, env, single_env, num_of_episodes = 0, render = False):
        if num_of_episodes == 0:
            num_of_episodes = 999
            
        print(f" ^ ^ ^ Running for {num_of_episodes} episodes ^ ^ ^ ")
            
        for ep in range(num_of_episodes):
            done = False
            state, _ = env.reset()
            step = 0
            score = 0
            
            while not done:
                if render:
                    single_env.render()
                action, _, _ = self.choose_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action.numpy())
                done = terminated or truncated
                score += reward[0]
                state = next_state
                step += 1
                
            print(f"episode {ep}: {score = :.0f}")
