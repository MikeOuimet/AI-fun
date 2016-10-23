import numpy as np
import gym
import tensorflow as tf
import matplotlib.pyplot as plt
import gym_pull

### 0 = shoot
### 10 = right
### 11 = left



def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.2, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


#Initial state and NN
env = gym.make('ppaquette/DoomBasic-v0')
#env.monitor.start('/tmp/doom', force=True)

observation = env.reset()

image_shape = np.shape(observation)

#dim = np.shape(env.observation_space)
#dim_actions = env.action_space.n

#print dim
#print dim_actions


#num_nodes = 100

#num_nodes_value = 100
discount_factor = .999

patch_size = 20

num_gradients = 5
maxsteps = 10
num_runs = 1
nskips = 4

sess = tf.InteractiveSession()

# State placeholders
state = tf.placeholder(tf.float32, shape=[None, 480, 640, 3])
action_choice = tf.placeholder(tf.float32, shape=[None, 3])  #potentially fix
reward_signal = tf.placeholder(tf.float32, shape=(None, 1))
advantage_signal = tf.placeholder(tf.float32, shape=(None, 1))

#Shared network
W_conv1 = weight_variable([patch_size, patch_size, 3, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(state, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([patch_size, patch_size, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([patch_size, patch_size, 64, 128])
b_conv3 = bias_variable([128])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

W_fc1 = weight_variable([80*128, 1024])
b_fc1 = bias_variable([1024])

h_pool3_flat = tf.reshape(h_pool3, [-1, 80*128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

#keep_prob = tf.placeholder(tf.float32)
#h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#Policy End of network
W_fc2 = weight_variable([1024, 3])
b_fc2 = bias_variable([3])
ao = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)
log_prob = tf.log(tf.diag_part(tf.matmul(ao, tf.transpose(action_choice))))# fix this so it doesn't need diag
log_prob = tf.reshape(log_prob, (1,-1))
loss = tf.matmul(log_prob, advantage_signal)
loss = -tf.reshape(loss, [-1])
train_step = tf.train.AdamOptimizer().minimize(loss)

# Value End of network
W_fc2_value = weight_variable([1024, 1])
b_fc2_value = bias_variable([1])
ao_value = tf.matmul(h_fc1, W_fc2_value) + b_fc2_value
loss_value = tf.nn.l2_loss(ao_value - reward_signal)
train_step_value = tf.train.AdamOptimizer().minimize(loss_value)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


reward_discount = np.ones(maxsteps)
for rs in range(1,maxsteps):
    reward_discount[rs] = discount_factor*reward_discount[rs-1] #generate discounting vector to multiply rewards by

for run in range(num_runs):

    states = np.zeros((maxsteps, 480, 640, 3), dtype='uint8')
    actions = np.zeros((maxsteps,3), dtype='uint8')
    rewards = np.zeros((maxsteps,1), dtype='float32')
    weighted_sum_rewards = np.zeros((maxsteps,1), dtype='float32')
    final_rewards = np.zeros((maxsteps,1), dtype='float32')
    timestep =0
    observation = env.reset()
    observation = np.reshape(observation, (1, 480, 640, 3))
    done = False
    
    #for i in range(10):
    while not done and timestep < maxsteps:
        env.render()
        action_prob = sess.run(ao, feed_dict={state: observation})
        #print action_prob
        action_vec = np.zeros(np.shape(env.action_space))


        action = np.argmax(np.random.multinomial(1, action_prob[0]))
        action_vec = np.zeros(np.shape(env.action_space))
        if action ==0:
            action_vec[0] = 1 #shoot
        elif action ==1:
            action_vec[10] =1 #right
        elif action ==2:
            action_vec[11] = 1 #left
        #print action_vec
        #new_observation, reward, done, info = env.step(action_step)
        for skip in range(nskips):
            observation, reward, done, info = env.step(action_vec)
        observation = np.reshape(observation, (1, 480, 640, 3))
        
        states[timestep, :] = observation
    
        actions[timestep, action] = 1
        rewards[timestep, :] = reward
        timestep += 1
        
        #observation[:] = new_observation


    #timestep_learning[run]=timestep
    timestep_of_run = timestep
    print timestep_of_run
    states = states[:timestep_of_run, :]
    actions = actions[:timestep_of_run, :]
    rewards = rewards[:timestep_of_run,:]
    print rewards
    #summed_rewards[:, 0] = np.cumsum(rewards[::-1])[::-1]

    for step in range(timestep_of_run):
        weighted_sum_rewards[step,0] = rewards[step:,0].dot(reward_discount[:timestep_of_run - step])
    weighted_sum_rewards= np.reshape(weighted_sum_rewards[:timestep_of_run,0], (timestep_of_run, 1))
    print weighted_sum_rewards
   
    final_rewards = weighted_sum_rewards
    current_values = sess.run(ao_value, feed_dict={state: states})
    print current_values
    print ''
    advantages = final_rewards - current_values
    print advantages

    print 'value function loss'
    print sess.run(loss_value, feed_dict={state: states, reward_signal: final_rewards})
    for i in range(num_gradients):
        sess.run(train_step_value, feed_dict={state: states, action_choice: actions, reward_signal: final_rewards})
        print sess.run(loss_value, feed_dict={state: states, reward_signal: final_rewards})
    #print ''


    
'''
    if run % 50 == 0:
        print 'run #: ', run
        print 'Time lasted: ', timestep
        print 'value function loss', sess.run(loss_value, feed_dict={state: states, reward_signal: final_rewards})
        print 'policy function loss', sess.run(loss, feed_dict={state: states, action_choice: actions, advantage_signal: advantages})
        print ''
    print 'policy function loss'
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

'''