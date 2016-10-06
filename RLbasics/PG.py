import numpy as np
import gym
import tensorflow as tf
import matplotlib.pyplot as plt
import gym_pull




def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.2, shape=shape)
  return tf.Variable(initial)


#Initial state and NN
#env = gym.make('ppaquette/DoomBasic-v0')
env = gym.make('CartPole-v1')
#env.monitor.start('/tmp/cartpole-experiment-1', force=True)


dim = max(np.shape(env.observation_space))
dim_actions = env.action_space.n

num_nodes = 100

num_nodes_value = 100
discount_factor = .999

num_gradients = 1
maxsteps = 500
num_runs = 1000

sess = tf.InteractiveSession()

# State placeholders
state = tf.placeholder(tf.float32, shape=[None, dim])
action_choice = tf.placeholder(tf.float32, shape=[None, dim_actions])
reward_signal = tf.placeholder(tf.float32, shape=(None, 1))
advantage_signal = tf.placeholder(tf.float32, shape=(None, 1))


# Value Network - uses state and reward_signal
W1_value = weight_variable([dim, num_nodes_value])
b1_value = bias_variable([num_nodes_value])
a1_value = tf.nn.relu(tf.matmul(state, W1_value) + b1_value)

Wo_value = weight_variable([num_nodes_value, 1])
bo_value = bias_variable([1])
ao_value = tf.matmul(a1_value, Wo_value) + bo_value

loss_value = tf.nn.l2_loss(ao_value - reward_signal)
train_step_value = tf.train.AdamOptimizer().minimize(loss_value)



# Policy Network
W1 = weight_variable([dim, num_nodes])
b1 = bias_variable([num_nodes])
a1 = tf.nn.relu(tf.matmul(state, W1) + b1)

Wo = weight_variable([num_nodes, dim_actions])
bo = bias_variable([dim_actions])
ao = tf.nn.softmax(tf.matmul(a1, Wo) + bo)


log_prob = tf.log(tf.diag_part(tf.matmul(ao, tf.transpose(action_choice))))# fix this so it doesn't need diag
log_prob = tf.reshape(log_prob, (1,-1))
loss = tf.matmul(log_prob, advantage_signal)
#loss = tf.matmul(log_prob, reward_signal)
loss = -tf.reshape(loss, [-1])
train_step = tf.train.AdamOptimizer().minimize(loss)




init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


reward_discount = np.ones(maxsteps)
for rs in range(1,maxsteps):
	reward_discount[rs] = discount_factor*reward_discount[rs-1] #generate discounting vector to multiply rewards by

timestep_learning = np.zeros((num_runs,1))
for run in range(num_runs):

    states = np.zeros((maxsteps,dim), dtype='float32')
    actions = np.zeros((maxsteps,dim_actions), dtype='float32')
    rewards = np.zeros((maxsteps,1), dtype='float32')
    weighted_sum_rewards = np.zeros((maxsteps,1), dtype='float32')
    final_rewards = np.zeros((maxsteps,1), dtype='float32')
    timestep =0
    observation = env.reset()
    observation = np.reshape(observation, (1, dim))
    done = False
    
    while not done and timestep < maxsteps:
        if run % 50 == 0:
            env.render()
        action_prob = sess.run(ao, feed_dict={state: observation})
        action = np.argmax(np.random.multinomial(1, action_prob[0]))
        new_observation, reward, done, info = env.step(action)
        
        states[timestep, :] = observation
    
        actions[timestep, action] = 1
        rewards[timestep, :] = reward
        timestep += 1
        
        observation[:] = new_observation

    
    timestep_learning[run]=timestep
    timestep_of_run = timestep
    states = states[:timestep_of_run, :]
    actions = actions[:timestep_of_run, :]
    rewards = rewards[:timestep_of_run,:]
    #summed_rewards[:, 0] = np.cumsum(rewards[::-1])[::-1]
    
    for step in range(timestep_of_run):
    	weighted_sum_rewards[step,0] = rewards[step:,0].dot(reward_discount[:timestep_of_run - step])
    weighted_sum_rewards= np.reshape(weighted_sum_rewards[:timestep_of_run,0], (timestep_of_run, 1))
    
    final_rewards = weighted_sum_rewards
    current_values = sess.run(ao_value, feed_dict={state: states})

    advantages = final_rewards - current_values




    #print 'value function loss'
    #print sess.run(loss_value, feed_dict={state: states, reward_signal: final_rewards})
    for i in range(num_gradients):
        sess.run(train_step_value, feed_dict={state: states, action_choice: actions, reward_signal: final_rewards})
    #    print sess.run(loss_value, feed_dict={state: states, reward_signal: final_rewards})
    #print ''


    

    if run % 50 == 0:
        print 'run #: ', run
        print 'Time lasted: ', timestep
        print 'value function loss', sess.run(loss_value, feed_dict={state: states, reward_signal: final_rewards})
        print 'policy function loss', sess.run(loss, feed_dict={state: states, action_choice: actions, advantage_signal: advantages})
        print ''
    #print 'policy function loss'
    #print sess.run(loss, feed_dict={state: states, action_choice: actions, advantage_signal: advantages})
    for i in range(num_gradients):
        sess.run(train_step, feed_dict={state: states, action_choice: actions, advantage_signal: advantages})
    #    print sess.run(loss, feed_dict={state: states, action_choice: actions, advantage_signal: advantages})
    #print ''



timestep_learning[run] = timestep

#env.monitor.close()
env.render(close=True)
plt.plot(timestep_learning)
plt.show()

#env.monitor.close()

