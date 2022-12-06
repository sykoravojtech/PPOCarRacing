import os
# Report only TF errors by default
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from collections import defaultdict
from gym.spaces import Box
import numpy as np
from tensorflow_probability import distributions
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Conv2D, Flatten, Input, Lambda
import tensorflow as tf

from utils import save_pltgraph


# import random
# import os

# https://github.com/MarekPokropinski/PPO/tree/1d6d3ab20ae3ec81fd135a00e23247fb52149e83

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


class PPO:
    def __init__(self, observation_space, action_space, entropy_coeff, gamma, gae_lambda, learning_rate):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.model: Model = None
        self.action_space = action_space
        self.optim = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.entropy_coeff = entropy_coeff
        self.get_pd = self.get_normal_pd
        
        print(f"{observation_space = }\n{action_space = }")

        if len(observation_space.shape) == 4:
            print("... building conv model ...")
            self.model = build_conv_model(observation_space.shape[1:], action_space) # passing just the pure one observation shape
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
        logp = pd.log_prob(action)
        return action, value, logp

    def value(self, state):
        pd, value = self.get_pd(state)
        return value

    def get_normal_pd(self, obs):
        value, b0, b1 = self.model(obs)
        pd = distributions.Beta(b0, b1)
        pd = distributions.Independent(pd, reinterpreted_batch_ndims=1)
        return pd, tf.squeeze(value, axis=-1)

    def probas(self, obs):
        value, pi = self.model(obs)
        return tf.math.softmax(pi)

    @tf.function
    def get_loss_pi(self, adv, logp, old_logp, clip):
        ratio = tf.math.exp(logp-old_logp)
        clipped_adv = tf.clip_by_value(ratio, 1-clip, 1+clip)*adv
        loss_pi = -tf.reduce_mean(tf.minimum(ratio*adv, clipped_adv))
        return loss_pi

    @tf.function
    def get_loss_value(self, pred_value, returns):
        return tf.reduce_mean((pred_value-returns)**2)

    # @tf.function # gives an error
    def grad(self, obs, clip, returns, values, actions, old_logp):
        epsilon = 1e-8
        adv = returns - values
        adv = (adv-tf.reduce_mean(adv)/(tf.keras.backend.std(adv)+epsilon))

        with tf.GradientTape() as tape:
            pd, pred_value = self.get_pd(obs)
            logp = pd.log_prob(actions)
            loss_pi = self.get_loss_pi(adv, logp, old_logp, clip)
            loss_v = self.get_loss_value(pred_value, returns)
            entropy = pd.entropy()

            loss = loss_pi+loss_v*0.5-entropy*self.entropy_coeff

        approxkl = .5 * tf.reduce_mean(tf.square(old_logp-logp))

        vars = tape.watched_variables()
        grads = tape.gradient(loss, vars)
        grads_and_vars = zip(grads, vars)
        # self.optim.apply_gradients(zip(grads, vars))
        return grads_and_vars, loss_pi, loss_v, entropy, approxkl

    @tf.function
    def learn_on_batch(self, lr, cliprange, obs, returns, actions, values, old_logp):
        obs = tf.keras.backend.cast_to_floatx(obs)
        grads_and_vars, loss_pi, loss_v, entropy, approxkl = self.grad(
            obs, cliprange, returns, values, actions, old_logp)
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

    def learn(self, obs, returns, actions, values, old_logp, cliprange, lr, mb_size=32, epochs=1):
        # losses = {'policy_loss':}
        losses = defaultdict(list)
        nbatch = obs.shape[0]
        inds = np.arange(nbatch)
        for epoch in range(epochs):
            np.random.shuffle(inds)
            for start in range(0, nbatch, mb_size):
                end = start + mb_size
                mb_idx = inds[start:end]
                slices = (tf.constant(arr[mb_idx]) for arr in (
                    obs, returns, actions, values, old_logp))
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
        obs, _ = env.reset()
        dones = np.zeros((env.num_envs,), dtype=dtype)
        scores = np.zeros_like(dones)
        score_history = []
        if type(lr) == float:
            def lr(x): return lr
            
        avg_score_history = []

        for e in range(start_from_ep, nepisodes+1):
            mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_logp = [], [], [], [], [], []
            for _ in range(steps_per_ep):
                actions, values, logp = self.act(obs)
                actions = actions.numpy()
                values = values.numpy()
                logp = logp.numpy()

                mb_obs.append(obs)
                mb_actions.append(actions)
                mb_values.append(values)
                mb_dones.append(dones)
                mb_logp.append(logp)

                obs, rewards, terminated, truncated, _ = env.step(actions)
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

            last_values = self.value(obs).numpy()

            returns = self.calculate_returns(
                mb_rewards, mb_values, mb_dones, last_values, dones, dtype=dtype)

            mb_obs = np.concatenate(mb_obs, axis=0)
            returns = np.concatenate(returns, axis=0)
            mb_actions = np.concatenate(mb_actions, axis=0)
            mb_values = np.concatenate(mb_values, axis=0)
            mb_logp = np.concatenate(mb_logp, axis=0)

            lr_now = lr(1.0 - e/nepisodes)
            # lr_now lr(1.0)
            self.learn(mb_obs, returns, mb_actions, mb_values, mb_logp,
                       clip_range, lr_now, mb_size, epochs=epochs_per_ep)
            avg_score = np.mean(
                score_history[-300:] if len(score_history) > 300 else score_history)
            avg_score_history.append(avg_score)
            
            if e % print_freq == 0:
                print(f'==> episode: {e}/{nepisodes}, avg score: {avg_score:.3f}')  # , score: {score_history[-10:] if score_history else []}
                # print(f"{returns = } len={len(returns)}")

            if e % save_interval == 0:
                chkpt_dir = os.path.join(model_dir, f"ep{e}")
                weights_dir = os.path.join(chkpt_dir, "weights") 
                print(f"    ... Saving model ep={e} ...")
                self.save_w(weights_dir)
                save_pltgraph(avg_score_history, chkpt_dir, e, start_from_ep)
                
            if args.tensorboard:
                with logger.as_default():
                    tf.summary.scalar('average score', avg_score, step=e)
                    tf.summary.scalar('learning rate', lr_now, step=e)
                
        # end of all episodes
        self.save_w(os.path.join(model_dir, "FINAL"))

    def save_w(self, filename='model'):
        self.model.save_weights(filename)

    def load_w(self, filename='model'):
        self.model.load_weights(filename)

def build_conv_model(input_shape, action_space) -> Model:
    input = Input(shape=input_shape, name='input_cnn')
    cnn1 = Conv2D(filters=32, kernel_size=8, strides=4, activation='relu', name='conv2D_1')(input)
    cnn2 = Conv2D(64, 4, 2, activation='relu', name='conv2D_2')(cnn1)
    cnn3 = Conv2D(64, 4, 2, activation='relu', name='conv2D_3')(cnn2)
    flat = Flatten(name='flatten')(cnn3)
    latent = Dense(512, activation='relu', name='dense_1')(flat) # this will be expanded to get beta distribution for actors decision
    value = Dense(1, activation='linear', name='value_critic')(latent) # critic
    # print(f"{value = }\n{latent = }")
    # value = <KerasTensor: shape=(None, 1) dtype=float32 (created by layer 'dense_1')>
    # latent = <KerasTensor: shape=(None, 512) dtype=float32 (created by layer 'dense')>
    
    model = Model(input, [value, latent], name='CNN_model')
        
    model_input_shape = model.input_shape
    input_tensor = Input(shape=model_input_shape[1:], name='input_PPO')
    value, latent = model(input_tensor)

    # print(f"{value = }\n{latent = }")
    # value = <KerasTensor: shape=(None, 1) dtype=float32 (created by layer 'model')>
    # latent = <KerasTensor: shape=(None, 512) dtype=float32 (created by layer 'model')>

    # print(f"{input_shape=}\n{model_input_shape=}\n{input=}\n{input_tensor=}")
    # input_shape=(96, 96, 3)
    # model_input_shape=(None, 96, 96, 3)
    # input=<KerasTensor: shape=(None, 96, 96, 3) dtype=float32 (created by layer 'input_1')>
    # input_tensor=<KerasTensor: shape=(None, 96, 96, 3) dtype=float32 (created by layer 'input_2')>
    
    size = action_space.shape[1]

    # for beta distribution
    b0 = Dense(size, activation='softplus', name='dense_b0')(latent)
    b0 = Lambda(lambda x: 1+x, name='lambda_b0')(b0)
    
    # for beta distribution
    b1 = Dense(size, activation='softplus', name='dense_b1')(latent)
    b1 = Lambda(lambda x: 1+x, name='lambda_b1')(b1)
    
    model = Model(input_tensor, [value, b0, b1], name='ActorCriticPPO')

    return model
