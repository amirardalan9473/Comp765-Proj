import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import pickle

# ENV_NAME_S = "CartPole-v1"
ENV_NAME_T = "CartPole-v99"



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




def loader():
    with open('GOODv1.pkl ', 'rb') as input:
        dqn_solver = pickle.load(input)

    with open('pi_adj.pkl', 'rb') as input:
        pi_adjust = pickle.load(input)

    env = gym.make(ENV_NAME_T)
    score_logger = ScoreLogger('PI ADJUST ANALYSISSSSSSS')
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    run = 0
    while True:
        run += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        # print(state)
        # input()
        step = 0
        while True:
            step += 1
            env.render()



            #TODO RUN PI ADJUST
            # u_action = dqn_solver.act(state)
            # state_copy = state
            # fuck = np.append(state_copy[0], u_action)
            # action = pi_adjust.predict(np.array(fuck).reshape(1, -1))
            # action = action[0][0]
            # action = int(round(action))

            # TODO RUN PI ADJUST COMMENT THE NEXT LINE

            action = dqn_solver.act(state)


            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_space])
            state = state_next
            if terminal:
                print(
                    "Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
                score_logger.add_score(step, run)
                break

    env.env.close()



if __name__ == "__main__":
    # cartpole()
    loader()
# env.env.close()
