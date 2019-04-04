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
weights = np.diag([2.0, 2.0, 0.3, 0.3])
# m_init = np.reshape([-1.0, -1.0, 0., 0.0], (1,4))
# S_init = np.diag([0.01, 0.05, 0.01, 0.05])
T = 1000
T_sim = T
# J = 4
# N = 8
# restarts = 2

state_dim = 4
control_dim = 1

with tf.Session() as sess:
    # env = myPendulum()

    print('Load PILCO')
    controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=bf, max_action=max_action)
    R = ExponentialReward(state_dim=state_dim, t=target, W=weights)
    pilco = load_pilco('saved/pilco-continuous-cartpole-{:d}'.format(1), controller=controller, reward=R)

    controller_1 = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=bf,
                               max_action=max_action)
    R_1 = ExponentialReward(state_dim=state_dim, t=target, W=weights)
    pilco_1 = load_pilco('saved/pilco-continuous-cartpole-{:d}'.format(7), controller=controller_1, reward=R_1)

    print('Run an episode')
    # env=gym.make('continuous-cartpole-v0')
    # rollout(env, pilco, timesteps=T_sim, verbose=True, SUBS=SUBS)
    # print('another')
    # rollout(env, pilco_1, timesteps=T_sim, verbose=True, SUBS=SUBS)

    print(pilco.mgpr.models[0].X)
    print(pilco_1.mgpr.models[0].X)
