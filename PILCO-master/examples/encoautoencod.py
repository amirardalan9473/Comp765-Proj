from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Input, Dense
from keras.models import Model

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

import warnings
warnings.filterwarnings("ignore")

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

from test import *
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





def sampler(pi, env, samples_n,sigma, trials=1,render=False):
    D = None

    for t in range(trials):
        state = env.reset()
        for i in range(samples_n):
            if render: env.render()

            action = utils.policy(env, pi, state, False) + np.random.normal(0, sigma)
            state_next, reward, terminal, info = env.step(action)

            if D is not None:
                D = np.vstack((D, [state, action, state_next]))
            elif D is None:
                D = np.array([state, action, state_next]).reshape(1, -1)

            state = state_next

            if terminal:
                break

    return D




def dataset(name='5'):

    env = gym.make('continuous-cartpole-v99')
    env.seed(73)
    controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=bf,
                               max_action=max_action)
    R = ExponentialReward(state_dim=state_dim, t=target, W=weights)
    pilco = load_pilco('saved/pilco-continuous-cartpole-{:s}'.format(name), controller=controller, reward=R,
                       sparse=False)

    env_S = gym.make('continuous-cartpole-v0')
    env_S.seed(73)

    env_T = gym.make('continuous-cartpole-v99')
    env_T.seed(73)


    score_logger_S = ScoreLogger('PI ADJUST ANALYSISSSSSSS')


    #TODO IMplement Pi adjust
    D_S = sampler(pilco, env_S, samples_n=30,sigma=1e-10, trials=50)
    D_S2= np.concatenate((D_S,sampler(pilco, env_S, samples_n=30,sigma=1e-8, trials=50)))
    D_S3 =np.concatenate((D_S2,sampler(pilco, env_S, samples_n=30,sigma=1e-6, trials=50)))
    D_S4= np.concatenate((D_S3,sampler(pilco, env_S, samples_n=30,sigma=1e-4, trials=50)))
    D_S5= np.concatenate((D_S4,sampler(pilco, env_S, samples_n=30,sigma=1e-2, trials=50)))

    save_object(D_S5, 'VAE_S.pkl')


    D_S = sampler(pilco, env_T, samples_n=30,sigma=1e-10, trials=50)
    D_S2= np.concatenate((D_S,sampler(pilco, env_T, samples_n=30,sigma=1e-8, trials=50)))
    D_S3 =np.concatenate((D_S2,sampler(pilco, env_T, samples_n=30,sigma=1e-6, trials=50)))
    D_S4= np.concatenate((D_S3,sampler(pilco, env_T, samples_n=30,sigma=1e-4, trials=50)))
    D_S5= np.concatenate((D_S4,sampler(pilco, env_T, samples_n=30,sigma=1e-2, trials=50)))

    save_object(D_S5, 'VAE_T.pkl')



    print('D_S sampling done')


    env_S.env.close()
    env_T.env.close()


def mat_loader(s='VAE_S.pkl',t='VAE_T.pkl'):

    with open(s, 'rb') as S:
        source = pickle.load(S)

    with open(t,'rb') as T:
        target = pickle.load(T)


    x = np.array([ np.ndarray.tolist(xi) for xi in source[:, 0] ])
    x2= np.array([ np.ndarray.tolist(xi) for xi in source[:, 1] ])
    x3 = np.array([ np.ndarray.tolist(xi) for xi in source[:, 2] ])
    x4 = np.hstack((x,x2))
    source=np.hstack((x4,x3))

    x = np.array([np.ndarray.tolist(xi) for xi in target[:, 0]])
    x2 = np.array([np.ndarray.tolist(xi) for xi in target[:, 1]])
    x3 = np.array([np.ndarray.tolist(xi) for xi in target[:, 2]])
    x4 = np.hstack((x, x2))
    target = np.hstack((x4, x3))

    print(source.shape,target.shape)
    return(source,target)


#TODO AUTOENCOD BEGIN################################################################################



# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


source,target = mat_loader()
np.random.shuffle(source)
np.random.shuffle(target)

x_train = target[0:6000,:]
x_test = target[6000:,:]


original_dim = x_train.shape[1]
x_train = np.reshape(x_train, [-1, original_dim])
x_test = np.reshape(x_test, [-1, original_dim])


# network parameters
input_shape = (original_dim, )
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
outputs = Dense(original_dim, activation='sigmoid')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
# plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m",
                        "--mse",
                        help=help_, action='store_true')
    args = parser.parse_args()
    models = (encoder, decoder)

    # VAE loss = mse_loss or xent_loss + kl_loss
    if args.mse:
        reconstruction_loss = mse(inputs, outputs)
    else:
        reconstruction_loss = binary_crossentropy(inputs,
                                                  outputs)

    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()
    # plot_model(vae,
    #            to_file='vae_mlp.png',
    #            show_shapes=True)

    if args.weights:
        vae.load_weights(args.weights)
    else:
        # train the autoencoder
        history = vae.fit(x_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, None))
        vae.save_weights('vae_target.h5')


        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('VAEtarget.png')
