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

SIGMA=0.00000000001

def load_and_run_model(env, name):
    controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=bf, max_action=max_action)
    R = ExponentialReward(state_dim=state_dim, t=target, W=weights)
    pilco = load_pilco('saved/pilco-continuous-cartpole-{:s}'.format(name), controller=controller, reward=R, sparse=False)

    print('Running {:s}'.format(name))
    rollout(env, pilco, timesteps=T_sim, verbose=False, SUBS=SUBS)





def loader(name):
    env = gym.make('continuous-cartpole-v99')
    env.seed(73)
    controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=bf,
                               max_action=max_action)
    R = ExponentialReward(state_dim=state_dim, t=target, W=weights)
    pilco = load_pilco('saved/pilco-continuous-cartpole-{:s}'.format(name), controller=controller, reward=R,
                       sparse=False)

    print('WE HEREEE')
    # input()

    score_logger = ScoreLogger('PI ADJUST ANALYSISSSSSSS')
    observation_space = env.observation_space.shape[0]
    run = 0
    while True:
        run += 1
        state = env.reset()
        step = 0
        while True:
            step += 1
            env.render()

            #TODO RUN PI ADJUST
            action = utils.policy(env, pilco, state, False)
            # TODO RUN PI ADJUST COMMENT THE NEXT LINE

            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            state = state_next
            if terminal:
                print(
                    "Run: " + str(run) + ", score: " + str(step))
                score_logger.add_score(step, run)
                break

    env.env.close()


#TODO################################################DO POLICY ADJUSTMENT HERE


def noiser(D,ind,mu=0,sigma=0.0000001):
    y = D
    noise = np.random.normal(mu,sigma,y.shape)
    # print(noise)
    y[:,ind]= y[:,ind] + noise[:,ind]

    return(y)


def sampler(pi, env, samples_n):
    D = None
    observation_space = env.observation_space.shape[0]

    state = env.reset()
    # print(state)
    # input()

    for i in range(samples_n):

        action = utils.policy(env, pi, state, False)+ np.random.normal(0,SIGMA)

        state_next, reward, terminal, info = env.step(action)
        if D is not None:
            D= np.vstack((D,[state,action,state_next]))
        elif D is None:
            D=np.array([state,action,state_next]).reshape(1,-1)
        state = state_next

        if terminal:
            state = env.reset()


    # y = np.array([np.array(xi) for xi in D])
    # print(D,'DD',D.shape)
    # input()
    return(D)


def inverse_dyn(D_T):
    kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(0.0, 10.0)) * RBF(length_scale=0.5,
                                                                                         length_scale_bounds=(0.0, 10.0))\
             + RBF(length_scale=2.0, length_scale_bounds=(0.0, 10.0))


    # print(np.concatenate(( D_T[:,0],D_T[:,2]),axis=0))
    # print(D_T[:,1])

    x = np.array([ np.ndarray.tolist(xi) for xi in D_T[:, 0] ])
    x2= np.array([ np.ndarray.tolist(xi) for xi in D_T[:, 2] ])
    # print(x)
    # print(x2)
    # input()
    x3 = np.hstack((x,x2))
    # print(x3)
    # input()
    # print(x3.shape)
    # print(x3,x3.shape)
    # input()
    # print(x3,'\n\n\n',x3.shape)
    gpr = GaussianProcessRegressor(kernel=kernel,random_state = 0).\
        fit(x3
            , np.array(D_T[:,1]).reshape(-1,1))
    # print(x3)
    return(gpr)

def L3(D_adj):
    kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(0.0, 10.0)) * RBF(length_scale=0.5,
                                                                                         length_scale_bounds=(0.0, 10.0))\
             + RBF(length_scale=2.0, length_scale_bounds=(0.0, 10.0))


    # print(np.concatenate(( D_T[:,0],D_T[:,2]),axis=0))
    # print(D_T[:,1])
    x =  np.array([ xi[0] for xi in D_adj ])
    x2 = np.array([ xi[1] for xi in D_adj ])
    y = np.array([ xi[2] for xi in D_adj ])

    # print(x.shape,'\n\n')
    # print(x2.reshape(-1,1).shape)

    x3 = np.concatenate((x,x2.reshape(-1,1)),axis=1)
    # print(x3.shape)
    # print(x3)
    # input()
    print('L3 gpr')
    gpr = GaussianProcessRegressor(kernel=kernel,random_state = 0).\
        fit(x3
            , y.reshape(-1,1))
    # print(x3)
    return(gpr)

def sampler_adj(pi_adj, pi_s, env, samples_n):
    D = list()
    observation_space = env.observation_space.shape[0]

    state = env.reset()
    # state = np.reshape(state, [1, observation_space])


    for i in range(samples_n):

        # u_action = pi_s.act(state)
        u_action = utils.policy(env, pi_s, state, False)
        state_copy = state
        fuck = np.append(state_copy[0], u_action)

        a = np.ndarray.tolist(state_copy)
        a.extend(np.ndarray.tolist(u_action))
        action = pi_adj.predict(np.array(a).reshape(1,-1))
        action=action[0]
        # if action!=u_action:
        #     print('DIIFFFFF')
        # print(action, u_action)
        # input()
        if action+u_action+ np.random.normal(0,SIGMA)>1:
            act=np.array([1])
        elif action+u_action+ np.random.normal(0,SIGMA)<-1:
            act=np.array([-1])
        else:
            act=action+u_action+ np.random.normal(0,SIGMA)
        state_next, reward, terminal, info = env.step(act)
        # state_next = np.reshape(state_next, [1, observation_space])

        D.append([state,action,state_next])
        state = state_next

        if terminal:
            state = env.reset()
            # state = np.reshape(state, [1, observation_space])


    y = np.array([np.array(xi) for xi in D])

    return(y)


def piadjust(NT,name):

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
    D_S = sampler(pilco,env_S,10)
    # D_S = noiser(D_S, [0,2])
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


            # print('\n\n', dqn_solver.act(  np.array(a[0:4]).reshape([1,4] ) ))
            # print( np.array(a[0:4]).reshape([1,4] )
            # print(i, '    ', D_adj)
            D_adj.append((x_s, u_t_S, u_t_T-u_t_S))


        # print(i, '    ',x_s, u_t_S, u_t_T)
        print('Goin for L3')
        pi_adj = L3(D_adj)
        print('L3 Done')
        # x_s.append(u_t_S)
        # print(pi_adj.predict(np.array(x_s).reshape(1,-1)))
        print(i)
        i = i + 1
        if (i%1==0):
            save_object(pi_adj, str(i)+'true_dyn_pi_adj.pkl')

    env_S.env.close()
    env_T.env.close()

    return(pi_adj)


if __name__ == "__main__":
    # cartpole()
    # loader('5')
    piadjust(10,'5')
# env.env.close()