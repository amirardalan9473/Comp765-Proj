import numpy as np
import gym
from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
from pilco.rewards import ExponentialReward
import tensorflow as tf

from examples.utils import rollout, load_pilco
import continuous_cartpole

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


def load_and_run_model(env, name):
    controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=bf, max_action=max_action)
    R = ExponentialReward(state_dim=state_dim, t=target, W=weights)
    pilco = load_pilco('saved/pilco-continuous-cartpole-{:s}'.format(name), controller=controller, reward=R, sparse=False)

    print('Running {:s}'.format(name))
    rollout(env, pilco, timesteps=T_sim, verbose=True, SUBS=SUBS)

with tf.Session() as sess:
    # env = myPendulum()
    env = gym.make('continuous-cartpole-v0')

    load_and_run_model(env, 'initial')
    for i in range(N):
        load_and_run_model(env, str(i))

    # controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=bf, max_action=max_action)
    # R = ExponentialReward(state_dim=state_dim, t=target, W=weights)
    # pilco = load_pilco('saved/pilco-continuous-cartpole-5', controller=controller, reward=R, sparse=False)
    #
    # # print('Run initial')
    # rollout(env, pilco, timesteps=T_sim, verbose=True, SUBS=SUBS)
    #
    # for i in range(N):
    #     controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=bf,
    #                                max_action=max_action)
    #     R = ExponentialReward(state_dim=state_dim, t=target, W=weights)
    #     pilco = load_pilco('saved/pilco-continuous-cartpole-{:d}'.format(1), controller=controller, reward=R, sparse=False)
    #     for model in pilco.mgpr.models:
    #         model.likelihood.variance = 0.001
    #         model.likelihood.variance.trainable = False
    #
    #     print('Run iteration {:d}'.format(i))
    #     rollout(env, pilco, timesteps=T_sim, verbose=True, SUBS=SUBS)


    # input('About to run first pilco (Press Enter)')
    # rollout(env, pilco, timesteps=T_sim, verbose=True, SUBS=SUBS)
    # input('About to run pilco #6 (Press Enter)')
    # rollout(env, pilco_1, timesteps=T_sim, verbose=True, SUBS=SUBS)
