import numpy as np
import gym
import tensorflow as tf

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

env = gym.make('CartPole-v0')

dim = max(np.shape(env.observation_space))
dim_actions = env.action_space.n

num_nodes = 20

sess = tf.InteractiveSession()

state = tf.placeholder(tf.float32, shape=[None, dim])
action_choice = tf.placeholder(tf.float32, shape=[None, dim_actions])
reward_signal = tf.placeholder(tf.float32, shape=[None, 1])

W1 = weight_variable([dim, num_nodes])
b1 = bias_variable([num_nodes])
a1 = tf.nn.relu(tf.matmul(state, W1) + b1)

Wo = weight_variable([num_nodes, dim_actions])
bo = bias_variable([dim_actions])
ao = tf.nn.softmax(tf.matmul(a1, Wo) + bo)

n_timesteps = 2
tau = np.zeros((n_timesteps, dim + dim_actions + 1))
tau[0, :] = [-.5, 1, 0, .1, 0, 1, 1]
tau[1, :] = [-.2, .6, 10, .01, .1, 0, 3]

rewards = np.sum(tau, axis=0)[dim+dim_actions]*np.ones((n_timesteps, 1))

states = tau[:, 0:dim]
actions = tau[:, dim: dim + dim_actions]


#loss = tf.log(tf.matmul(action_choice, ao))*reward_signal
loss = tf.matmul(ao,tf.transpose(action_choice))


init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

env.render(close=True)

loss_val = sess.run(loss, feed_dict={state: states, action_choice: actions, reward_signal: rewards})

print sess.run(ao, feed_dict={state: states, action_choice: actions, reward_signal: rewards})
print sess.run(action_choice + 0.0, feed_dict={state: states, action_choice: actions, reward_signal: rewards})
#loss_val = sess.run(tf.matmul(action_choice, ao), feed_dict={state: states[0, :], action_choice: actions[0,:]})
print loss_val


