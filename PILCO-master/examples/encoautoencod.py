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





def sampler(pi, env, samples_n,sigma, trials=1,render=True):
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

    print(D_S5.shape)
    input()


    print('D_S sampling done')

    D_T = None
    i = 0
    pi_adj = pilco

    while i< NT:
        D_adj = []

        if i ==0:
            D_i_T = sampler(pilco, env_T,10)

        elif i!= 0:
            D_i_T = sampler_adj(pi_adj,pilco, env_T, 10)

        if D_T is not None:
            # print(D_i_T.shape, D_T.shape)
            D_T = np.concatenate((D_i_T,D_T))
        elif  D_T is  None:
            D_T = D_i_T


        print('Goin for inverse dyn')
        gpr = inverse_dyn(D_T)
        print('inverse dyn done')

        for samp in D_S:


            x_s = np.ndarray.tolist(samp[0])
            x_s1 = np.ndarray.tolist(samp[2])
            u_t_S = samp[1]
            # print(u_t_S)

            a=np.ndarray.tolist(samp[0])

            a.extend( np.ndarray.tolist(samp[2]))
            # print( np.array(a).reshape(1, 8)  )
            # print(a.shape, '\n\n\n')
            u_t_T = gpr.predict( np.array(a).reshape(1, 8), return_std=False)



            D_adj.append((x_s, u_t_S, u_t_T))


        # print(i, '    ',x_s, u_t_S, u_t_T)
        print('Goin for L3')
        pi_adj = L3(D_adj)
        print('L3 Done')
        # x_s.append(u_t_S)
        # print(pi_adj.predict(np.array(x_s).reshape(1,-1)))
        print(i)
        i = i + 1
        if (i%1==0):
            save_object(pi_adj, str(i)+'_pi_adj.pkl')

    env_S.env.close()
    env_T.env.close()

    return(pi_adj)


#TODO AUTOENCOD BEGIN################################################################################

x = Input(batch_shape=(batch_size, original_dim))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_sigma = Dense(latent_dim)(h)






def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(z_log_sigma) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
# so you could write `Lambda(sampling)([z_mean, z_log_sigma])`
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])



decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)



vae = Model(x, x_decoded_mean)

# encoder, from inputs to latent space
encoder = Model(x, z_mean)

# generator, from latent space to reconstructed inputs
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)



def vae_loss(x, x_decoded_mean):
    xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
    return xent_loss + kl_loss

vae.compile(optimizer='rmsprop', loss=vae_loss)




(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

vae.fit(x_train, x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, x_test))


x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.show()



if __name__ == "__main__":
    dataset()