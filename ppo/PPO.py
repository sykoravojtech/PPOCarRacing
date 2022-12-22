import tensorflow as tf
from keras.models import Model
from tensorflow import keras
from tensorflow_probability import distributions
import numpy as np
from gym.spaces import Box
from collections import defaultdict
import os
# Report only TF errors by default
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from actorcritic import get_ActorCritic_model
from utils import save_pltgraph

# import random
# import os

class PPO:
    def __init__(self, observation_space, action_space, entropy_coeff, gamma, gae_lambda, learning_rate, value_fun_coeff):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.model: Model = None
        self.action_space = action_space
        self.optim = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.entropy_coeff = entropy_coeff
        self.get_pd = self.get_beta_pd
        self.vf_coeff = value_fun_coeff

        print(f"{observation_space = }\n{action_space = }")

        if len(observation_space.shape) == 4:
            print("... building conv model ...")
            # passing just the pure one observation shape
            self.model = get_ActorCritic_model(
                observation_space.shape[1:], action_space)
        else:
            raise Exception(
                f'Unsupported observation space shape {observation_space.shape} ... add 1 dimension to mimic vectorenv')

    def get_model(self):
        return self.model

    def model_summary(self):
        self.model.summary()
        self.model.get_layer("CNN_model").summary()

    @tf.function
    def act(self, state):
        pd, value = self.get_pd(state)
        action = pd.sample()
        return action, value, pd.log_prob(action)

    def value(self, state):
        return self.get_pd(state)[1]

    def get_beta_pd(self, state):
        value, alpha, beta = self.model(state)
        pd = distributions.Independent(
            distributions.Beta(alpha, beta),
            reinterpreted_batch_ndims=1)
        return pd, tf.squeeze(value, axis=-1)

    def probas(self, state):
        value, pi = self.model(state)
        return tf.math.softmax(pi)

    @tf.function
    def get_loss_policy_clipped(self, advantage, logp, old_logp, clip):
        ratio = tf.math.exp(logp - old_logp)
        clipped_adv = tf.clip_by_value(ratio, 1 - clip, 1 + clip) * advantage
        loss_policy = -tf.reduce_mean(tf.minimum(ratio*advantage, clipped_adv))
        return loss_policy

    @tf.function
    def get_loss_critic(self, pred_value, returns):
        return tf.reduce_mean((pred_value - returns)**2)

    # @tf.function # gives an error
    def grad(self, state, clip, returns, values, actions, old_logp):
        epsilon = 1e-8
        adv = returns - values
        adv = (adv-tf.reduce_mean(adv)/(tf.keras.backend.std(adv)+epsilon))

        with tf.GradientTape() as tape:
            pd, pred_value = self.get_pd(state)
            logp = pd.log_prob(actions)
            loss_pi = self.get_loss_policy_clipped(adv, logp, old_logp, clip)
            loss_critic = self.get_loss_critic(pred_value, returns)
            entropy = pd.entropy()

            loss = loss_pi + loss_critic*self.vf_coeff - entropy * \
                self.entropy_coeff  # main loss function of PPO

        approxkl = .5 * tf.reduce_mean(tf.square(old_logp-logp))

        vars = tape.watched_variables()
        grads = tape.gradient(loss, vars)
        grads_and_vars = zip(grads, vars)
        return grads_and_vars, loss_pi, loss_critic, entropy, approxkl

    @tf.function
    def learn_on_batch(self, lr, cliprange, state, returns, actions, values, old_logp):
        state = tf.keras.backend.cast_to_floatx(state)
        grads_and_vars, loss_pi, loss_v, entropy, approxkl = self.grad(
            state, cliprange, returns, values, actions, old_logp)
        self.optim.learning_rate = lr
        self.optim.apply_gradients(grads_and_vars)

        return loss_pi, loss_v, entropy, approxkl

    def calculate_returns(self, rewards, values, dones, last_values, last_dones, dtype):
        last_adv = 0
        steps = len(rewards)
        nenvs = rewards[0].shape[0]
        adv = np.zeros((steps, nenvs))
        for t in reversed(range(steps)):
            if t == steps-1:
                next_non_terminal = 1.0 - last_dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - dones[t+1]
                next_values = values[t+1]

            delta = rewards[t] + self.gamma * \
                next_values * next_non_terminal - values[t]
            last_adv = delta + self.gamma * self.gae_lambda * next_non_terminal * last_adv
            adv[t] = last_adv
        return (adv + values).astype(dtype)

    def learn(self, state, returns, actions, values, old_logp, cliprange, lr, mb_size=32, epochs=1):
        # losses = {'policy_loss':}
        losses = defaultdict(list)
        nbatch = state.shape[0]
        inds = np.arange(nbatch)
        for epoch in range(epochs):
            np.random.shuffle(inds)
            for start in range(0, nbatch, mb_size):
                end = start + mb_size
                mb_idx = inds[start:end]
                slices = (tf.constant(arr[mb_idx]) for arr in (
                    state, returns, actions, values, old_logp))
                loss = self.learn_on_batch(
                    lr, cliprange, *slices)
                for k, v in zip(['policy_loss', 'value_loss', 'entropy', 'kl'], loss):
                    losses[k].append(v)

        return losses

    def train(
        self, env, args, nepisodes, steps_per_ep=2048, epochs_per_ep=4,
        mb_size=256, clip_range=0.2, lr=3e-4, save_interval=10,
        model_dir='model', start_from_ep=1, print_freq=1, logger=None
    ):

        print(f"^^^^^ TRAINING for {nepisodes-start_from_ep+1} episodes ^^^^^")
        dtype = tf.keras.backend.floatx()
        state, _ = env.reset()
        dones = np.zeros((env.num_envs,), dtype=dtype)
        scores = np.zeros_like(dones)
        score_history = []
        if type(lr) == float:
            def lr(x): return lr

        avg_score_history = []

        step = 0
        for e in range(start_from_ep, nepisodes+1):
            mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_logp = [], [], [], [], [], []
            for _ in range(steps_per_ep):
                step += 1
                actions, values, logp = self.act(state)
                actions = actions.numpy()
                values = values.numpy()
                logp = logp.numpy()

                mb_obs.append(state)
                mb_actions.append(actions)
                mb_values.append(values)
                mb_dones.append(dones)
                mb_logp.append(logp)

                state, rewards, terminated, truncated, _ = env.step(actions)
                dones = terminated | truncated
                # if np.any(dones):
                #     print(f"{dones = }")
                dones = tf.cast(dones, tf.float32)

                mb_rewards.append(rewards)
                scores += rewards
                for i in range(rewards.shape[0]):  # for each env in vector env
                    if dones[i]:
                        # print("appending")
                        score_history.append(scores[i])
                        # print(f"{score_history = }")
                        scores[i] = 0

            last_values = self.value(state).numpy()

            returns = self.calculate_returns(
                mb_rewards, mb_values, mb_dones, last_values, dones, dtype=dtype)

            mb_obs = np.concatenate(mb_obs, axis=0)
            returns = np.concatenate(returns, axis=0)
            mb_actions = np.concatenate(mb_actions, axis=0)
            mb_values = np.concatenate(mb_values, axis=0)
            mb_logp = np.concatenate(mb_logp, axis=0)

            if args.constant_lr:
                lr_now = lr(1.0)
            else:
                lr_now = lr(1.0 - e/nepisodes)

            self.learn(mb_obs, returns, mb_actions, mb_values, mb_logp,
                       clip_range, lr_now, mb_size, epochs=epochs_per_ep)
            
            avg_score = np.mean(score_history[-300:] if len(score_history) > 300 else score_history)
            avg_score_history.append(avg_score)

            if e % print_freq == 0:
                # , score: {score_history[-10:] if score_history else []}
                print(
                    f'==> episode: {e}/{nepisodes} step={step}, avg score: {avg_score:.3f}')
                # print(f"{returns = } len={len(returns)}")

            if e % save_interval == 0:
                chkpt_dir = os.path.join(model_dir, f"ep{e}")
                weights_dir = os.path.join(chkpt_dir, "weights")
                print(f"    ... Saving model ep={e} ...")
                self.save_w(weights_dir)
                save_pltgraph(avg_score_history, chkpt_dir, e, start_from_ep)

            if args.tensorboard:
                with logger.as_default():
                    tf.summary.scalar('average score', avg_score, step=step)
                    tf.summary.scalar('learning rate', lr_now, step=step)
                    tf.summary.scalar('episode', e, step=step)

        # end of all episodes
        self.save_w(os.path.join(model_dir, "FINAL"))

    def save_w(self, filename='model'):
        self.model.save_weights(filename)

    def load_w(self, filename='model'):
        self.model.load_weights(filename)
