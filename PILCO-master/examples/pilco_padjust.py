import numpy as np
import gym
import tensorflow as tf

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

from examples.utils import load_pilco
from pilco.controllers import RbfController
from pilco.rewards import ExponentialReward

import continuous_cartpole

def sample_trajectories(env, controller, num_trajectories=10, num_timesteps=40):
    D = []
    for _ in range(num_trajectories):
        x = env.reset()

        for _ in range(num_timesteps):
            env.render()

            u = controller.act(x)
            x_new, _, done, _ = env.step(u)

            D.append(np.hstack((x, u, x_new)))
            x = x_new

            if done: break

    return np.array(D)


def inverse_dyn(D_T):
    kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(0.0, 10.0)) \
             * RBF(length_scale=0.5, length_scale_bounds=(0.0, 10.0)) \
             + RBF(length_scale=2.0, length_scale_bounds=(0.0, 10.0))

    x = np.hstack([D_T[:,:4], D_T[5:]])
    u = D_T[:,4]

    gpr = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(x,u)
    return (gpr)


def padjust(env_s, env_t, pi_s, num_trials=10):
    X_T = []; Y_T = []
    pi_adj = pi_s

    D_T = None
    D_S = sample_trajectories(env_s, pi_s)

    for i in range(num_trials):
        print('Iteration {:d}'.format(i))

        D_adj = []

        this_D_T = sample_trajectories(env_t, pi_adj)
        D_T = this_D_T if not D_T else np.vstack([D_T, this_D_T])

        g = inverse_dyn(D_T)


if __name__ == '__main__':
    bf = 10
    max_action = 1.0
    target = np.array([0., 0., 0., 0.])
    weights = np.diag([0.5, 0.1, 0.5, 0.25])
    T = 400

    state_dim = 4
    control_dim = 1

    with tf.Session() as sess:
        env = gym.make('continuous-cartpole-v0')

        controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=bf, max_action=max_action)
        R = ExponentialReward(state_dim=state_dim, t=target, W=weights)
        pilco = load_pilco('saved/pilco-continuous-cartpole-5', controller=controller, reward=R, sparse=False)

        pilco.act = lambda x: pilco.compute_action(x[None, :])[0, :]

        env_s = gym.make('continuous-cartpole-v0')
        env_t = gym.make('continuous-cartpole-v1')
        padjust(env_s, env_t, pilco)
