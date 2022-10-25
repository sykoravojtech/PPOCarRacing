import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
# keras.backend.set_floatx('float32')

INPUT_DIMS = (96, 96, 3)

class ActorNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=256, fc2_dims=256):
        super(ActorNetwork, self).__init__()

        self.conv1 = Conv2D(32, 8, 4, input_shape = INPUT_DIMS, activation = tf.nn.relu)
        self.conv2 = Conv2D(64,4,2, activation = tf.nn.relu)
        self.conv3 = Conv2D(64,4,2, activation = tf.nn.relu)
        self.flatten = Flatten()
        
        # self.conv1 = Conv2D(8, (4,4), input_shape = INPUT_DIMS, activation = tf.nn.relu)
        # self.pool1 = MaxPooling2D(2,2)
        # self.conv2 = Conv2D(16, (3,3), activation = tf.nn.relu)
        # self.pool2 = MaxPooling2D(2,2)
        # self.conv3 = Conv2D(32, (3, 3), activation = tf.nn.relu)
        # self.pool3 = MaxPooling2D(2,2)
        # self.flatten = Flatten()

        self.fc1 = Dense(512, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.fc3 = Dense(n_actions, activation='softmax')

    @tf.function
    def call(self, state):
        state = tf.cast(state, tf.float32)
        # print(f"->networks.ActorCall {state.shape=} {state.dtype}")
        x = self.conv1(state)
        # x = self.pool1(x)
        x = self.conv2(x)
        # x = self.pool2(x)
        x = self.conv3(x)
        # x = self.pool3(x)
        x = self.flatten(x)

        x = self.fc1(x)
        # x = self.fc2(x)
        x = self.fc3(x)

        return x


class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=256, fc2_dims=256):
        super(CriticNetwork, self).__init__()

        self.conv1 = Conv2D(32, 8, 4, input_shape = INPUT_DIMS, activation = tf.nn.relu)
        self.conv2 = Conv2D(64,4,2, activation = tf.nn.relu)
        self.conv3 = Conv2D(64,4,2, activation = tf.nn.relu)
        self.flatten = Flatten()
        
        # self.conv1 = Conv2D(8, (4,4), input_shape = INPUT_DIMS, activation = tf.nn.relu)
        # self.pool1 = MaxPooling2D(2,2)
        # self.conv2 = Conv2D(16, (3,3), activation = tf.nn.relu)
        # self.pool2 = MaxPooling2D(2,2)
        # self.conv3 = Conv2D(32, (3, 3), activation = tf.nn.relu)
        # self.pool3 = MaxPooling2D(2,2)
        # self.flatten = Flatten()
        
        self.fc1 = Dense(512, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.q = Dense(1, activation=None)

    @tf.function
    def call(self, state):
        state = tf.cast(state, tf.float32)
        # print(f"->networks.CriticCall {state.shape=} {state.dtype}")
        x = self.conv1(state)
        # x = self.pool1(x)
        x = self.conv2(x)
        # x = self.pool2(x)
        x = self.conv3(x)
        # x = self.pool3(x)
        x = self.flatten(x)
        
        x = self.fc1(x)
        # x = self.fc2(x)
        q = self.q(x)

        return q
    
    
    
    
    
# BACKUP
    
# class ActorNetwork(keras.Model):
#     def __init__(self, n_actions, fc1_dims=256, fc2_dims=256):
#         super(ActorNetwork, self).__init__()

#         self.fc1 = Dense(fc1_dims, activation='relu')
#         self.fc2 = Dense(fc2_dims, activation='relu')
#         self.fc3 = Dense(n_actions, activation='softmax')

#     def call(self, state):
#         x = self.fc1(state)
#         x = self.fc2(x)
#         x = self.fc3(x)

#         return x


# class CriticNetwork(keras.Model):
#     def __init__(self, fc1_dims=256, fc2_dims=256):
#         super(CriticNetwork, self).__init__()
#         self.fc1 = Dense(fc1_dims, activation='relu')
#         self.fc2 = Dense(fc2_dims, activation='relu')
#         self.q = Dense(1, activation=None)

#     def call(self, state):
#         x = self.fc1(state)
#         x = self.fc2(x)
#         q = self.q(x)

#         return q