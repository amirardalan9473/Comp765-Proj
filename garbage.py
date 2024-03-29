import gym
import tensorflow as tf
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from IPython.display import clear_output
from gym import wrappers



# base code from udacity-deep-learning/reinforcement/Q-learning-cart.ipynb

# Create class QNetwork
class QNetwork:
    def __init__(self,learning_rate=0.01, state_size=4,action_size=2,hidden_size=10,hidden_layers=2,alpha=0.,name='QNetwork'):
        # create Q Network
        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(tf.float32,[None, state_size],name='inputs')

            # placeholder for actions, to be one-hot encoded next
            self.actions_ = tf.placeholder(tf.int32,[None],name='actions')

            # one hot encode actions
            one_hot_actions = tf.one_hot(self.actions_,action_size)

            # placeholder for target Qs
            self.targetQs_ = tf.placeholder(tf.float32,[None],name='target')

            # ReLU hidden layers
            self.fc1 = tf.layers.dense(self.inputs_,hidden_size,activation=None,kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.fc1 = tf.maximum(alpha * self.fc1, self.fc1)

            self.fc2 = tf.layers.dense(self.fc1, hidden_size,activation=None,kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.fc2 = tf.maximum(alpha * self.fc2, self.fc2)

            out_layer = self.fc2

            # Linear output layer
            self.output = tf.layers.dense(out_layer, action_size,activation=None,kernel_initializer=tf.contrib.layers.xavier_initializer())

            ### Train with loss (targetQ - Q)^2
            # output has length 2, for two actions. This next line chooses
            # one value from output (per row) according to the one-hot encoded actions.
            self.Q = tf.reduce_sum(tf.multiply(self.output, one_hot_actions), axis=1)

            self.loss = tf.reduce_mean(tf.square(self.targetQs_ - self.Q))
            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)


# create memory class for storing previous experiences
class Memory():
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)),
                               size=batch_size,
                               replace=False)
        return [self.buffer[ii] for ii in idx]


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N


def normalize_state(x):
    normalizer = [2., 3., 0.3, 2.]
    y = x / normalizer
    return y


def initialize_memory_rand_states(memory_size=1000, pretrain_length=32):
    # Initialize the simulation
    # Make a random action
    env.reset()
    action = env.action_space.sample()
    state, reward, done, _ = env.step(action)
    state = normalize_state(state)

    memory = Memory(max_size=memory_size)

    # Make a bunch of random actions and store the experiences
    ii = 0
    while ii < pretrain_length or not done:

        # Make a random action
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)

        if done:
            # The simulation fails so no next state
            next_state = np.zeros(state.shape)

            # Add experience to memory
            memory.add((state, action, reward, next_state))

            # Start new episode
            state = env.reset()
            state = normalize_state(state)

        else:
            # Add experience to memory
            memory.add((state, action, reward, next_state))
            state = next_state

        ii = ii + 1

    return memory


def train_q_network(mainQN,
                    memory,
                    train_episodes=1500,
                    gamma=0.99,
                    explore_start=1.0,
                    explore_stop=0.01,
                    decay_rate=0.0001,
                    batch_size=32,
                    max_steps=500,
                    verbose=True):
    state = env.reset()
    state = normalize_state(state)

    # Now train with experiences
    saver = tf.train.Saver()
    rewards_list = []
    with tf.Session() as sess:
        # Initialize variables
        sess.run(tf.global_variables_initializer())

        step = 0
        steps_list = []

        for ep in range(train_episodes):
            total_reward = 0
            t = 0

            while t < max_steps:
                step += 1

                # Explore or Exploit
                explore_p = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * step)
                if explore_p > np.random.rand():
                    # Make a random action
                    action = env.action_space.sample()
                else:
                    # Get action from Q-network
                    state_normalizer = [2., 3., 0.3, 3.]
                    feed = {mainQN.inputs_: state.reshape((1, *state.shape))}
                    Qs = sess.run(mainQN.output, feed_dict=feed)
                    action = np.argmax(Qs)

                # Take action, get new state and reward
                next_state, reward, done, _ = env.step(action)
                next_state = normalize_state(next_state)

                total_reward += reward

                if done:
                    t = t + 1
                    # the episode ends so no next state
                    next_state = np.zeros(state.shape)
                    steps_list.append(t)
                    t = max_steps

                    # Add experience to memory
                    memory.add((state, action, reward, next_state))
                    state = env.reset()
                    state = normalize_state(state)
                else:
                    # Add experience to memory
                    memory.add((state, action, reward, next_state))
                    state = next_state
                    t += 1

                # Sample mini-batch from memory
                batch = memory.sample(batch_size)
                states = np.array([each[0] for each in batch])
                actions = np.array([each[1] for each in batch])
                rewards = np.array([each[2] for each in batch])
                next_states = np.array([each[3] for each in batch])

                # Train network
                target_Qs = sess.run(mainQN.output, feed_dict={mainQN.inputs_: next_states})

                # Set target_Qs to 0 for states where episode ends
                episode_ends = (next_states == np.zeros(states[0].shape)).all(axis=1)
                target_Qs[episode_ends] = (0, 0)

                targets = rewards + gamma * np.max(target_Qs, axis=1)

                loss, _ = sess.run([mainQN.loss, mainQN.opt],
                                   feed_dict={mainQN.inputs_: states,
                                              mainQN.targetQs_: targets,
                                              mainQN.actions_: actions})

            rewards_list.append((ep, total_reward))
            runningMean = np.mean(steps_list[-100:])
            if verbose:
                print('Episode: {}'.format(ep),
                      'Total reward: {}'.format(total_reward),
                      'Training loss: {:.4f}'.format(loss),
                      'Explore P: {:.4f}'.format(explore_p),
                      'RunMean : {:.4f}'.format(runningMean))

            if runningMean > 495.:
                saver.save(sess, "checkpoints/cartpole.ckpt")
                return rewards_list, mainQN, saver

        saver.save(sess, "checkpoints/cartpole.ckpt")
        return rewards_list, mainQN, saver


def plot_rewards(rewards_list):
    eps, rews = np.array(rewards_list).T
    smoothed_rews = running_mean(rews, 10)
    plt.plot(eps[-len(smoothed_rews):], smoothed_rews)
    plt.plot(eps, rews, color='grey', alpha=0.3)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')


def generate_and_train_qnetwork(train_episodes=1500,
                                gamma=0.99,
                                explore_start=1.0,
                                explore_stop=0.01,
                                decay_rate=0.0001,
                                hidden_size=128,
                                hidden_layers=2,
                                learning_rate=0.0001,
                                memory_size=10000,
                                batch_size=32,
                                render=False,
                                alpha=0.1,
                                verbose=True):
    # reset graph
    tf.reset_default_graph()

    mainQN = QNetwork(name='main', hidden_size=hidden_size,
                      hidden_layers=hidden_layers, learning_rate=learning_rate, alpha=alpha)

    memory = initialize_memory_rand_states(memory_size=memory_size, pretrain_length=batch_size)

    # train q-network
    rewards_list, mainQN, saver = train_q_network(mainQN,
                                                  memory,
                                                  train_episodes=train_episodes,
                                                  gamma=gamma,
                                                  explore_start=explore_start,
                                                  explore_stop=explore_stop,
                                                  decay_rate=decay_rate,
                                                  batch_size=batch_size,
                                                  verbose=verbose)

    if verbose:
        # plot training
        plot_rewards(rewards_list)

    avg_train_rewards = np.sum([each[1] for each in rewards_list]) / len(rewards_list)
    if verbose:
        print('average training reward = ', avg_train_rewards)

    return avg_train_rewards, mainQN, saver, len(rewards_list)


# Create the environment
env = gym.make('CartPole-v1')

# Start monitor
env = wrappers.Monitor(env, '/tmp/cartpole-experiment-1', force=True)

# Train network
avg_train_rewards, mainQN, saver, number_of_episodes = generate_and_train_qnetwork(train_episodes=1000, verbose=False)
print('average test reward = ', avg_train_rewards, '   number of trials = ', number_of_episodes)

# Close environment
env.close