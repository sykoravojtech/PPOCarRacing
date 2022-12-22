from keras.layers import Dense, Conv2D, Flatten, Input, Lambda
from keras.models import Model

def get_ActorCritic_model(input_shape, action_space) -> Model:
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
    alpha = Dense(size, activation='softplus', name='dense_alpha')(latent)
    alpha = Lambda(lambda x: 1+x, name='lambda_alpha')(alpha)
    
    # for beta distribution
    beta = Dense(size, activation='softplus', name='dense_beta')(latent)
    beta = Lambda(lambda x: 1+x, name='lambda_beta')(beta)
    
    model = Model(input_tensor, [value, alpha, beta], name='ActorCriticPPO')

    return model
