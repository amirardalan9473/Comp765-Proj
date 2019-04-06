import gym
import numpy as np

from sklearn.base import  BaseEstimator
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import RBF

GPU = False
NUM_EPOCHS = 10
CONVERGENCE_THRESHOLD = 1e-3
SYSTEM_NOISE = 0

env = gym.make('CartPole-v1')

class multivar_gp(BaseEstimator):
    def __init__(self, num_units):
        self.num_units = num_units
        self.gps = [gaussian_process.GaussianProcessRegressor(RBF(), alpha=1e-4) for _ in range(num_units)]

    def fit(self, x, y):
        if x.shape[0] == 0: return self.gps

        for i, gp in enumerate(self.gps):
            gp.fit(x, y[:,i])

        return self.gps

    def predict(self, x):
        y = np.empty((x.shape[0], self.num_units))
        for i, gp in enumerate(self.gps):
            y[:,i] = self.gps[i].predict(x)
        return y

def pilco(epochs):
    # initialize
    dynamics = np.empty((0,9)) # 4 current state, 1 action, 4 next state
    gps = multivar_gp(4)
    policy = (np.random.rand(1,4), np.random.rand(1))

    # 1. TODO apply random signal

    for _ in range(epochs):
        observation = env.reset()
        # 2. learn GP dynamics model
        gps.fit(dynamics[:,:5], dynamics[:,5:])

        error = CONVERGENCE_THRESHOLD + 1
        while error > CONVERGENCE_THRESHOLD:
            params = policy
            # 3. approximate inference for policy eval
            # 4. policy improvement
            # 5. update params

            error = 1e-5

        policy = params

#####
# dynamics = np.empty((0,9)) # 4 current state, 1 action, 4 next state
# gps = multivar_gp(4)
# policy = (np.random.rand(1,4), np.random.rand(1))
#
# print(policy)
#
# for i_episode in range(NUM_EPOCHS):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         print(t)
#
#         gps.fit(dynamics[:,:5], dynamics[:,5:])
#
#         prev_observation = np.array(env.state)
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         delta_observation = observation - prev_observation
#
#         mu_pred, var_pred = gps.predict(np.array(np.concatenate((prev_observation, [1. * action]))).reshape(1, -1))
#         err = np.linalg.norm(mu_pred - delta_observation)
#         print(err)
#
#         example = np.concatenate((prev_observation, [1. * action], delta_observation))
#         dynamics = np.vstack([dynamics, example])
#
#         print(env.action_space.sample)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
#
# gps.fit(dynamics[:,:5], dynamics[:,5:])
# gps.predict(dynamics[:,:5])
# print('hi')
