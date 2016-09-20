import numpy as np
import gym
import tensorflow as tf

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


#Initial state and NN
env = gym.make('CartPole-v0')

dim = max(np.shape(env.observation_space))
dim_actions = env.action_space.n

num_nodes = 50

num_gradients =200
maxsteps = 400
num_runs = 1

#sess = tf.InteractiveSession()

state = tf.placeholder(tf.float32, shape=[None, dim])
action_choice = tf.placeholder(tf.float32, shape=[None, dim_actions])
reward_signal = tf.placeholder(tf.float32, shape=(None,1) )
#n_timesteps = tf.placeholder(tf.int16, shape = ())


W1 = weight_variable([dim, num_nodes])
b1 = bias_variable([num_nodes])
a1 = tf.nn.relu(tf.matmul(state, W1) + b1)

Wo = weight_variable([num_nodes, dim_actions])
bo = bias_variable([dim_actions])
ao = tf.nn.softmax(tf.matmul(a1, Wo) + bo)

#blah = tf.matmul(ao,tf.transpose(action_choice))

log_prob1 = tf.matmul(ao,tf.transpose(action_choice))
log_prob2 = tf.diag_part(tf.matmul(ao,tf.transpose(action_choice)))
log_prob = tf.log(tf.diag_part(tf.matmul(ao,tf.transpose(action_choice))))# fix this so it doesn't need diag
log_prob = tf.reshape(log_prob, (1,-1)) # how can I matrix multiply without hardcode reshaping
loss =  -tf.matmul(log_prob, reward_signal)
loss = tf.reshape(loss, [-1])
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)



for run in range(num_runs):

    states = np.zeros((maxsteps,dim), dtype='float32')
    actions = np.zeros((maxsteps,dim_actions), dtype='float32')
    rewards = np.zeros((maxsteps,1), dtype='float32')
    timestep =0
    observation = env.reset()
    observation = np.reshape(observation,(1,dim))
    done = False
    
    while not done and timestep < maxsteps:
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
rewards = rewards[:timestep,:]


print sess.run(loss, feed_dict={state: states, action_choice: actions, reward_signal: rewards})

for i in range(num_gradients):
    sess.run(train_step, feed_dict={state: states, action_choice: actions, reward_signal: rewards})
    print sess.run(loss, feed_dict={state: states, action_choice: actions, reward_signal: rewards})
#print sess.run(ao, feed_dict={state: states})

env.render(close=True)



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

