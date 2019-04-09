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
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings("ignore")

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

# ENV_NAME_T = "CartPole-v1"
# ENV_NAME_T = "CartPole-v99"



from score_logger import ScoreLogger
import matplotlib.pyplot as plt
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
    Rs = np.empty(10).reshape(1,10)
    env = gym.make('continuous-cartpole-v99')
    env.seed(73)
    controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=bf,
                               max_action=max_action)
    R = ExponentialReward(state_dim=state_dim, t=target, W=weights)
    pilco = load_pilco('saved/pilco-continuous-cartpole-{:s}'.format(name), controller=controller, reward=R,
                       sparse=False)

    for pick in range(1,11):
        env = gym.make('continuous-cartpole-v99')

        with open(str(pick)+'_pi_adj.pkl', 'rb') as inp2:
            pi_adjust = pickle.load(inp2)


        score_logger = ScoreLogger('PI ADJUST ANALYSISSSSSSS')
        run = 0
        avg_reward = 0
        while run!=101:
            run += 1
            if (run%20==0):
                print('run:  ',run)
            state = env.reset()
            # print(state)
            # input()
            step = 0
            while True:
                step += 1
                #env.render()



                #TODO RUN PI ADJUST
                u_action =  utils.policy(env, pilco, state, False)
                state_copy = state

                a = np.ndarray.tolist(state_copy)
                a.extend(np.ndarray.tolist(u_action))
                action = pi_adjust.predict(np.array(a).reshape(1, -1))
                action = action[0]
                if action[0]>1:
                    action[0]=1
                elif action[0]<-1:
                    action[0]=-1
                # TODO RUN PI ADJUST COMMENT THE NEXT LINE

                state_next, reward, terminal, info = env.step(action)
                reward = reward if not terminal else -reward
                state = state_next
                if terminal:
                    # print("Run: "  + ", score: " + str(step))
                    score_logger.add_score(step, run)
                    avg_reward = avg_reward + step
                    break
        avg_reward=avg_reward/run
        env.env.close()
        Rs[0][pick-1]=avg_reward
    return(Rs)
def souce_loader(name):
        Rs = np.empty(10).reshape(1, 10)
        env = gym.make('continuous-cartpole-v99')
        env.seed(73)
        controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=bf,
                                   max_action=max_action)
        R = ExponentialReward(state_dim=state_dim, t=target, W=weights)
        pilco = load_pilco('saved/pilco-continuous-cartpole-{:s}'.format(name), controller=controller, reward=R,
                           sparse=False)

        env = gym.make('continuous-cartpole-v99')

        pi_adjust = None

        score_logger = ScoreLogger('PI ADJUST ANALYSISSSSSSS')
        run = 0
        avg_reward = 0
        while run != 101:
            run += 1
            if (run % 20 == 0):
                print('run:  ', run)
            state = env.reset()
            # print(state)
            # input()
            step = 0
            while True:
                step += 1
                # env.render()

                # TODO RUN PI ADJUST
                u_action = utils.policy(env, pilco, state, False)
                state_copy = state


                # TODO RUN PI ADJUST COMMENT THE NEXT LINE

                state_next, reward, terminal, info = env.step(u_action)
                reward = reward if not terminal else -reward
                state = state_next
                if terminal:
                    # print("Run: "  + ", score: " + str(step))
                    score_logger.add_score(step, run)
                    avg_reward = avg_reward + step
                    break
        avg_reward = avg_reward / run
        env.env.close()
        return (avg_reward)


def plot3D(name):
    fil = np.loadtxt(name,delimiter=',')
    x=np.array([0]*10+[1]*10+[2]*10+[3]*10+[4]*10+[5]*10)
    print(x)
    y=np.arange(60)
    y=y%10
    z=fil.flatten()

    print(x.shape,y.shape,z.shape)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('PILCO iteration', fontsize=8)
    ax.set_ylabel('Policy Adjust iteration', fontsize=8)
    ax.set_zlabel('100 eposiode average', fontsize=8)

    #TODO BELOW is the avergae of 100 episodes for source PILCO in target enviroment
    plane0 = 63.04950495049505
    plane1=67.72277227722772
    plane2=146.9108910891089
    plane3=55.37623762376238
    plane4=58.336633663366335
    plane5=56.82178217821782

    ax.scatter3D(x, y, z, c=z)
    plt.show()
    plt.savefig('3d.png')

if __name__ == "__main__":
    # cartpole()
    # reward = np.empty(60).reshape(6,10)
    #
    # for i in range(6):
    #     print(i)
    #     rew = loader(str(i))
    #     reward[i,:] = rew
    # print(reward)
    # np.savetxt('rewards.out',reward, delimiter=',')
        #


    for i in range(6):
        rew = souce_loader(str(i))
        print(rew)


# plot3D('rewards.out')

# env.env.close()
