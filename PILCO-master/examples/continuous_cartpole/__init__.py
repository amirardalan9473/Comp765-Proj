from gym.envs.registration import register

import continuous_cartpole.envs as e

print(dir(e))

register(
    id='continuous-cartpole-v0',
    entry_point='envs.continuous_cartpole:ContinuousCartPoleEnv',
    max_episode_steps=1000
)