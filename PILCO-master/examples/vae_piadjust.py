
from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

import numpy as np
import gym
from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
from pilco.rewards import ExponentialReward
import tensorflow as tf
from keras.models import load_model

from examples.utils import rollout, load_pilco
import continuous_cartpole
import pickle
from score_logger import ScoreLogger
import utils
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dense, Dropout
import pickle

import  sklearn

from encoautoencod import *

import warnings
warnings.filterwarnings("ignore")


def models(name):
    input_shape = (9,)
    intermediate_dim = 7
    batch_size = 128
    latent_dim = 5
    epochs = 500

    # VAE model = encoder + decoder
    # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Dense(intermediate_dim, activation='relu')(inputs)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()
    # plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim, activation='relu')(latent_inputs)
    outputs = Dense(9, activation='sigmoid')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    # plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')
    vae.load_weights(name, by_name=True)

    return(vae,encoder,decoder)


#TODO### BEGIN EXPERIMENTSSSSSSSSS

import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import pickle
import numpy as np
import gym
from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
from pilco.rewards import ExponentialReward
import tensorflow as tf

from examples.utils import rollout, load_pilco
import continuous_cartpole
import pickle
from score_logger import ScoreLogger
import utils
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dense, Dropout
import pickle

import  sklearn

import warnings
warnings.filterwarnings("ignore")

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

# ENV_NAME_T = "CartPole-v1"
# ENV_NAME_T = "CartPole-v99"



from score_logger import ScoreLogger

# ENV_NAME = "Acrobot-v1"


GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01

from test import *

np.random.seed(73)

np.random.seed(0)

ITERATION_TO_LOAD = 7

SUBS=3
bf = 10
maxiter=50
max_action=1.0
target = np.array([0., 0., 0., 0.]) # TODO review if correct
weights = np.diag([0.5, 0.1, 0.5, 0.25])
# m_init = np.reshape([-1.0, -1.0, 0., 0.0], (1,4))
# S_init = np.diag([0.01, 0.05, 0.01, 0.05])
T = 400
T_sim = T
# J = 4
# N = 8
# restarts = 2

state_dim = 4
control_dim = 1

N = 6

import utils

def loader(name):
    vae_s, enc_s, dec_s = models('vae_source.h5')
    vae_t, enc_t, dec_t = models('vae_target.h5')

    envS = gym.make('continuous-cartpole-v1')
    env = gym.make('continuous-cartpole-v99')
    env.seed(73)
    controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=bf,
                               max_action=max_action)
    R = ExponentialReward(state_dim=state_dim, t=target, W=weights)
    pilco = load_pilco('saved/pilco-continuous-cartpole-{:s}'.format(name), controller=controller, reward=R,
                       sparse=False)



    score_logger = ScoreLogger('PI ADJUST ANALYSISSSSSSS')
    run = 0
    while True:
        run += 1
        state = env.reset()
        # print(state)
        # input()
        step = 0
        while True:
            step += 1
            env.render()



            #TODO RUN PI ADJUST
            u_action =  utils.policy(env, pilco, state, False)
            state_copy = state

            a = np.ndarray.tolist(state_copy)
            a.extend(np.ndarray.tolist(u_action))
            action = pi_adjust.predict(np.array(a).reshape(1, -1))
            action = action[0]
            # TODO RUN PI ADJUST COMMENT THE NEXT LINE

            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            state = state_next
            if terminal:
                print(
                    "Run: "  + ", score: " + str(step))
                score_logger.add_score(step, run)
                break

    env.env.close()



if __name__ == "__main__":
    # cartpole()
    loader('2')
# env.env.close()

if __name__ == "__main__":
    vae_s,enc_s,dec_s = models('vae_source.h5')
    vae_t, enc_t, dec_t = models('vae_target.h5')

    source, target = mat_loader()
    sample = source[20].reshape((1,-1))
    mean,_,_ = enc_s.predict([sample])
    pred = dec_s.predict(mean)
    print(pred[0][4],sample[0][4])
    print(sample-pred)

