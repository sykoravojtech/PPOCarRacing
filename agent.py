import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam
import tensorflow_probability as tfp
from memory import PPOMemory
from networks import ActorNetwork, CriticNetwork
# keras.backend.set_floatx('float32')

class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003,
                 gae_lambda=0.95, policy_clip=0.2, batch_size=64,
                 n_epochs=10, chkpt_dir='models/'):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.chkpt_dir = chkpt_dir
        self.input_dims = input_dims

        self.actor = ActorNetwork(n_actions)
        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic = CriticNetwork()
        self.critic.compile(optimizer=Adam(learning_rate=alpha))
        self.memory = PPOMemory(batch_size)

    def store_transition(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save(self.chkpt_dir + 'actor')
        self.critic.save(self.chkpt_dir + 'critic')

    def load_models(self):
        print('... loading models ...')
        self.actor = keras.models.load_model(self.chkpt_dir + 'actor')
        self.critic = keras.models.load_model(self.chkpt_dir + 'critic')

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])
        # print(f"--><choose_action> {state.shape=} {state.dtype}")

        probs = self.actor(state)
        dist = tfp.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(state)

        action = action.numpy()[0]
        value = value.numpy()[0]
        log_prob = log_prob.numpy()[0]

        return action, log_prob, value

    def learn(self):
        for epoch in range(self.n_epochs):
            # print(f"{epoch = }")
            state_arr, action_arr, old_prob_arr, vals_arr,\
                reward_arr, dones_arr, batches = \
                self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            # print("a_t")
            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1] * (
                        1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t

            for batch in batches:
                self.do_batch(state_arr, old_prob_arr,action_arr, batch, advantage, values)

        self.memory.clear_memory()

    @tf.function
    def do_batch(self, state_arr, old_prob_arr, action_arr, batch, advantage, values):
        # use tf.gather is like indexing using an array
        with tf.GradientTape(persistent=True) as tape:
            states = tf.gather(state_arr, indices=batch)
            old_probs = tf.gather(old_prob_arr, indices=batch)
            actions = tf.gather(action_arr, indices=batch)

            probs = self.actor(states)
            dist = tfp.distributions.Categorical(probs)
            new_probs = dist.log_prob(actions)

            critic_value = self.critic(states)

            critic_value = tf.squeeze(critic_value, 1)

            prob_ratio = tf.math.exp(new_probs - old_probs)
            weighted_probs = tf.gather(advantage, indices=batch) * prob_ratio
            clipped_probs = tf.clip_by_value(prob_ratio,
                                                1-self.policy_clip,
                                                1+self.policy_clip)
            weighted_clipped_probs = clipped_probs * tf.gather(advantage, indices=batch)
            actor_loss = -tf.math.minimum(weighted_probs,
                                            weighted_clipped_probs)
            actor_loss = tf.math.reduce_mean(actor_loss)

            returns = tf.gather(advantage, indices=batch) + tf.gather(values, indices=batch)
            # critic_loss = tf.math.reduce_mean(tf.math.pow(
            #                                  returns-critic_value, 2))
            critic_loss = keras.losses.MSE(critic_value, returns)

        actor_params = self.actor.trainable_variables
        actor_grads = tape.gradient(actor_loss, actor_params)
        critic_params = self.critic.trainable_variables
        critic_grads = tape.gradient(critic_loss, critic_params)
        self.actor.optimizer.apply_gradients(
                zip(actor_grads, actor_params))
        self.critic.optimizer.apply_gradients(
                zip(critic_grads, critic_params))





    # original func
    # def learn(self):
    #         for epoch in range(self.n_epochs):
    #             # print(f"{epoch = }")
    #             state_arr, action_arr, old_prob_arr, vals_arr,\
    #                 reward_arr, dones_arr, batches = \
    #                 self.memory.generate_batches()

    #             values = vals_arr
    #             advantage = np.zeros(len(reward_arr), dtype=np.float32)

    #             # print("a_t")
    #             for t in range(len(reward_arr)-1):
    #                 discount = 1
    #                 a_t = 0
    #                 for k in range(t, len(reward_arr)-1):
    #                     a_t += discount*(reward_arr[k] + self.gamma*values[k+1] * (
    #                         1-int(dones_arr[k])) - values[k])
    #                     discount *= self.gamma*self.gae_lambda
    #                 advantage[t] = a_t

    #             for batch in batches:
    #                 # do this into a tf function
    #                 # print("batch")
    #                 with tf.GradientTape(persistent=True) as tape:
    #                     states = tf.convert_to_tensor(state_arr[batch])
    #                     old_probs = tf.convert_to_tensor(old_prob_arr[batch])
    #                     actions = tf.convert_to_tensor(action_arr[batch])

    #                     probs = self.actor(states)
    #                     dist = tfp.distributions.Categorical(probs)
    #                     new_probs = dist.log_prob(actions)

    #                     critic_value = self.critic(states)

    #                     critic_value = tf.squeeze(critic_value, 1)

    #                     prob_ratio = tf.math.exp(new_probs - old_probs)
    #                     weighted_probs = advantage[batch] * prob_ratio
    #                     clipped_probs = tf.clip_by_value(prob_ratio,
    #                                                     1-self.policy_clip,
    #                                                     1+self.policy_clip)
    #                     weighted_clipped_probs = clipped_probs * advantage[batch]
    #                     actor_loss = -tf.math.minimum(weighted_probs,
    #                                                 weighted_clipped_probs)
    #                     actor_loss = tf.math.reduce_mean(actor_loss)

    #                     returns = advantage[batch] + values[batch]
    #                     # critic_loss = tf.math.reduce_mean(tf.math.pow(
    #                     #                                  returns-critic_value, 2))
    #                     critic_loss = keras.losses.MSE(critic_value, returns)

    #                 actor_params = self.actor.trainable_variables
    #                 actor_grads = tape.gradient(actor_loss, actor_params)
    #                 critic_params = self.critic.trainable_variables
    #                 critic_grads = tape.gradient(critic_loss, critic_params)
    #                 self.actor.optimizer.apply_gradients(
    #                         zip(actor_grads, actor_params))
    #                 self.critic.optimizer.apply_gradients(
    #                         zip(critic_grads, critic_params))

    #         self.memory.clear_memory()

