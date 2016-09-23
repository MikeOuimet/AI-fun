import numpy as np
import gym
import tensorflow as tf
import matplotlib.pyplot as plt

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.2, shape=shape)
  return tf.Variable(initial)


#Initial state and NN
env = gym.make('CartPole-v0')

dim = max(np.shape(env.observation_space))
dim_actions = env.action_space.n

num_nodes = 100

num_gradients = 1
maxsteps = 300
num_runs = 500

sess = tf.InteractiveSession()

state = tf.placeholder(tf.float32, shape=[None, dim])
action_choice = tf.placeholder(tf.float32, shape=[None, dim_actions])
reward_signal = tf.placeholder(tf.float32, shape=(None,1) )
n_timesteps = tf.placeholder(tf.float32, shape = ())


W1 = weight_variable([dim, num_nodes])
b1 = bias_variable([num_nodes])
a1 = tf.nn.relu(tf.matmul(state, W1) + b1)

Wo = weight_variable([num_nodes, dim_actions])
bo = bias_variable([dim_actions])
ao = tf.nn.softmax(tf.matmul(a1, Wo) + bo)


log_prob = tf.log(tf.diag_part(tf.matmul(ao,tf.transpose(action_choice))))# fix this so it doesn't need diag
log_prob = tf.reshape(log_prob, (1,-1))
loss =  tf.matmul(log_prob, reward_signal)
loss = -tf.reshape(loss, [-1])#/n_timesteps
train_step = tf.train.AdamOptimizer().minimize(loss)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


timestep_learning = np.zeros((num_runs,1))
for run in range(num_runs):

    states = np.zeros((maxsteps,dim), dtype='float32')
    actions = np.zeros((maxsteps,dim_actions), dtype='float32')
    rewards = np.zeros((maxsteps,1), dtype='float32')
    timestep =0
    observation = env.reset()
    observation = np.reshape(observation,(1,dim))
    done = False
    
    while not done and timestep < maxsteps:
        #env.render()
        action_prob = sess.run(ao, feed_dict={state: observation})
        action = np.argmax(np.random.multinomial(1, action_prob[0]))
        new_observation, reward, done, info = env.step(action)
        
        states[timestep, :]= observation
    
        actions[timestep, action] = 1
        rewards[timestep,:] = reward
        timestep +=1
        
        observation[:] = new_observation

    states = states[:timestep, :]
    actions = actions[:timestep, :]
    #rewards = np.cumsum(rewards[::-1])[::-1] # this should help implement sum of later rewards
    rewards = rewards[:timestep,:]
    rewards = np.sum(rewards)*rewards # fix this so it includes sum of costs after that action

    if run % 50 == 0:
        print 'run #: ', run
        print 'Time lasted: ', timestep
    #print 'time lasted', timestep
    #print sess.run(loss, feed_dict={state: states, action_choice: actions, reward_signal: rewards, n_timesteps: timestep})
   
    for i in range(num_gradients):
        sess.run(train_step, feed_dict={state: states, action_choice: actions, reward_signal: rewards, n_timesteps: timestep})
        #print sess.run(loss, feed_dict={state: states, action_choice: actions, reward_signal: rewards, n_timesteps: timestep})
    #print sess.run(ao, feed_dict={state: states})
    #print sess.run(loss, feed_dict={state: states, action_choice: actions, reward_signal: rewards, n_timesteps: timestep})
    #print ''
    timestep_learning[run] = timestep

env.render(close=True)
plt.plot(timestep_learning)
plt.show()




#a=  sess.run(log_prob, feed_dict={state: states, action_choice: actions, reward_signal: rewards})
#b= sess.run(reward_signal+0.0, feed_dict={state: states, action_choice: actions, reward_signal: rewards})
#print sess.run(loss, feed_dict={state: states, action_choice: actions, reward_signal: rewards})


'''   Test data
tau = np.zeros((n_timesteps, dim + dim_actions + 1))
tau[0, :] = [-.5, 1, 0, .1, 0, 1, 1]
tau[1, :] = [-.2, .6, 10, .01, 1, 0, 3]
tau[2, :] = [-.1, -.6, -10, -.01, 0, 1, 3]

rewards = np.sum(tau, axis=0)[dim+dim_actions]*np.ones((n_timesteps, 1))

states = tau[:, 0:dim]
actions = tau[:, dim: dim + dim_actions]
'''

