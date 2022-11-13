import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
# keras.backend.set_floatx('float32')

INPUT_DIMS = (96, 96, 3)

class ActorNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=256, fc2_dims=256):
        super(ActorNetwork, self).__init__()
        
        # v3 deeprl mff
        self.resize = tf.keras.layers.Resizing(48,48,interpolation='nearest')
        self.standardize = tf.keras.layers.Lambda(lambda x: tf.image.per_image_standardization(x))
        
        self.conv1 = tf.keras.layers.Conv2D(32, (3,3), activation='gelu', input_shape=(48, 48, 3))
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2))

        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='gelu')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.pool2 = tf.keras.layers.MaxPooling2D((2, 2))
        
        # v2
        # self.conv1 = Conv2D(32, 8, 4, input_shape = INPUT_DIMS, activation = tf.nn.relu)
        # self.conv2 = Conv2D(64,4,2, activation = tf.nn.relu)
        # self.conv3 = Conv2D(64,4,2, activation = tf.nn.relu)
        # self.flatten = Flatten()
        
        # v1
        # self.conv1 = Conv2D(8, (4,4), input_shape = INPUT_DIMS, activation = tf.nn.relu)
        # self.pool1 = MaxPooling2D(2,2)
        # # self.conv2 = Conv2D(16, (3,3), activation = tf.nn.relu)
        # self.pool2 = MaxPooling2D(2,2)
        # # self.conv3 = Conv2D(32, (3, 3), activation = tf.nn.relu)
        # self.pool3 = MaxPooling2D(2,2)
        
        
        # end
        self.flatten = Flatten()

        self.fc1 = Dense(512, activation='gelu')
        self.fc2 = Dense(fc2_dims, activation='gelu')
        self.fc3 = Dense(n_actions, activation='softmax')

    @tf.function
    def call(self, state):
        state = tf.cast(state, tf.float32)
        # print(f"->networks.ActorCall {state.shape=} {state.dtype}")
        
        # v3
        dims = [-1] +  list(INPUT_DIMS)
        x = tf.reshape(state, dims)
        x = self.resize(x)   
        x = self.standardize(x)
 
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        
        # v1 v2
        # x = self.conv1(state)
        # x = self.pool1(x)
        # x = self.conv2(x)
        # x = self.pool2(x)
        # x = self.conv3(x)
        # x = self.pool3(x)
        
        
        # end
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=256, fc2_dims=256):
        super(CriticNetwork, self).__init__()

       # v3 deeprl mff
        self.resize = tf.keras.layers.Resizing(48,48,interpolation='nearest')
        self.standardize = tf.keras.layers.Lambda(lambda x: tf.image.per_image_standardization(x))
        
        self.conv1 = tf.keras.layers.Conv2D(32, (3,3), activation='gelu', input_shape=(48, 48, 3))
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2))

        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='gelu')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.pool2 = tf.keras.layers.MaxPooling2D((2, 2))
        
        # v2
        # self.conv1 = Conv2D(32, 8, 4, input_shape = INPUT_DIMS, activation = tf.nn.relu)
        # self.conv2 = Conv2D(64,4,2, activation = tf.nn.relu)
        # self.conv3 = Conv2D(64,4,2, activation = tf.nn.relu)
        # self.flatten = Flatten()
        
        # v1
        # self.conv1 = Conv2D(8, (4,4), input_shape = INPUT_DIMS, activation = tf.nn.relu)
        # self.pool1 = MaxPooling2D(2,2)
        # # self.conv2 = Conv2D(16, (3,3), activation = tf.nn.relu)
        # self.pool2 = MaxPooling2D(2,2)
        # # self.conv3 = Conv2D(32, (3, 3), activation = tf.nn.relu)
        # self.pool3 = MaxPooling2D(2,2)
        
        
        # end
        self.flatten = Flatten()

        self.fc1 = Dense(512, activation='gelu')
        self.fc2 = Dense(fc2_dims, activation='gelu')
        self.q = Dense(1, activation=None)

    @tf.function
    def call(self, state):
        state = tf.cast(state, tf.float32)
        # print(f"->networks.ActorCall {state.shape=} {state.dtype}")
        
        # v3
        dims = [-1] +  list(INPUT_DIMS)
        x = tf.reshape(state, dims)
        x = self.resize(x)   
        x = self.standardize(x)
 
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        
        # v1 v2
        # x = self.conv1(state)
        # x = self.pool1(x)
        # x = self.conv2(x)
        # x = self.pool2(x)
        # x = self.conv3(x)
        # x = self.pool3(x)
        
        
        # end
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        q = self.q(x)

        return q
    
    
def get_actor(n_actions):
    model = keras.Sequential([
        Conv2D(32, 8, 4, input_shape = INPUT_DIMS, activation = "gelu"),
        Conv2D(64, 4, 2, activation = "gelu"),
        Conv2D(64, 4, 2, activation = "gelu"),
        Flatten(),
        Dense(512, activation='gelu'),
        Dense(n_actions, activation='softmax')
    ])
    return model

def get_critic():
    model = keras.Sequential([
        Conv2D(32, 8, 4, input_shape = INPUT_DIMS, activation = "gelu"),
        Conv2D(64, 4, 2, activation = "gelu"),
        Conv2D(64, 4, 2, activation = "gelu"),
        Flatten(),
        Dense(512, activation='gelu'),
        Dense(1, activation='softmax')
    ])
    return model
    
    
# BACKUP

# def get_actor(n_actions):
#     model = keras.Sequential([
#         Conv2D(32, 8, 4, input_shape = INPUT_DIMS, activation = tf.nn.relu),
#         Conv2D(64, 4, 2, activation = tf.nn.relu),
#         Conv2D(64, 4, 2, activation = tf.nn.relu),
#         Flatten(),
#         Dense(512, activation='relu'),
#         Dense(n_actions, activation='softmax')
#     ])
#     return model

# def get_critic():
#     model = keras.Sequential([
#         Conv2D(32, 8, 4, input_shape = INPUT_DIMS, activation = tf.nn.relu),
#         Conv2D(64, 4, 2, activation = tf.nn.relu),
#         Conv2D(64, 4, 2, activation = tf.nn.relu),
#         Flatten(),
#         Dense(512, activation='relu'),
#         Dense(1, activation='softmax')
#     ])
#     return model
    
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