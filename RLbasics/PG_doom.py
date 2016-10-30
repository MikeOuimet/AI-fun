import numpy as np
import gym
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import ndimage
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

def mean_pool_nxm(x,n,m):
  return tf.nn.avg_pool(x, ksize=[1, n, n, 1],
                        strides=[1, m, m, 1], padding='SAME')

def max_pool_nxm(x,n,mx, my):
  return tf.nn.max_pool(x, ksize=[1, n, n, 1],
                        strides=[1, mx, my, 1], padding='SAME')

def mean_subtraction(pooled_picture):
    avg0 = np.round(np.mean(np.mean(pooled_picture[:,:,:,0], axis = 1), axis = 1)[0])
    avg1 = np.round(np.mean(np.mean(pooled_picture[:,:,:,1], axis = 1), axis = 1)[0])
    avg2 = np.round(np.mean(np.mean(pooled_picture[:,:,:,2], axis = 1), axis = 1)[0])

    pooled_picture[0,:, :, 0] = pooled_picture[0,:, :, 0] - avg0
    pooled_picture[0,:, :, 1] = pooled_picture[0,:, :, 1] - avg1
    pooled_picture[0,:, :, 2] = pooled_picture[0,:, :, 2] - avg2
    pooled_picture[pooled_picture < 0] =0
    pooled_picture[pooled_picture > 255] = 255
    return pooled_picture


#Initial state and NN
env = gym.make('ppaquette/DoomBasic-v0')
#env.monitor.start('/tmp/doom5000-2')

observation = env.reset()

image_shape = np.shape(observation)

#dim = np.shape(env.observation_space)
#dim_actions = env.action_space.n

#print dim
#print dim_actions


#num_nodes = 100

#num_nodes_value = 100
discount_factor = .98

patch_size = 2

num_gradients = 1
maxsteps =200
num_runs = 4000
nskips = 4

height_start = 180
height_end = 250
width_start = 0
width_end = 640

# State placeholders
original_state = tf.placeholder(tf.float32, shape=[None, height_end-height_start, width_end-width_start, 3])
state = tf.placeholder(tf.float32, shape=[None, 7, 128, 3]) # (35,270,3)
action_choice = tf.placeholder(tf.float32, shape=[None, 3])  #potentially fix
reward_signal = tf.placeholder(tf.float32, shape=(None, 1))
advantage_signal = tf.placeholder(tf.float32, shape=(None, 1))
length_of_episode = tf.placeholder(tf.float32, shape = (1,1))

#Pooling pre-processing
h_prepool1 = max_pool_nxm(original_state, 4, 10, 5)
#h_prepool2 = mean_pool_nxm(h_prepool1, 2 , 1)
#h_prepool3 = mean_pool_nxm(h_prepool2, 2 , 1)


#Shared network
W_conv1 = weight_variable([patch_size, patch_size, 3, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(state, W_conv1) + b_conv1)


W_conv2 = weight_variable([patch_size, patch_size, 32, 32])
b_conv2 = bias_variable([32])
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
h_flat = tf.reshape(h_conv2, [-1, 32*2*32])

W_fc1 = weight_variable([32*2*32, 10])
b_fc1 = bias_variable([10])
h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)



#Policy End of network
W_fc2 = weight_variable([10, 3])
b_fc2 = bias_variable([3])
ao = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)
log_prob = tf.log(tf.diag_part(tf.matmul(ao, tf.transpose(action_choice))))# fix this so it doesn't need diag
log_prob = tf.reshape(log_prob, (1,-1))
loss = tf.matmul(log_prob, advantage_signal)
loss = tf.reshape(loss, [-1])/length_of_episode 
train_step = tf.train.RMSPropOptimizer(1e-5).minimize(-loss)
#train_step = tf.train.AdamOptimizer().minimize(-loss)

# Value End of network
W_fc1_value = weight_variable([10, 1])
b_fc1_value = bias_variable([1])
ao_value = tf.matmul(h_fc1, W_fc1_value) + b_fc1_value
loss_value = tf.nn.l2_loss(ao_value - reward_signal)
train_step_value = tf.train.RMSPropOptimizer(1e-5).minimize(loss_value)
#train_step_value = tf.train.AdamOptimizer().minimize(loss_value)


init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

observation = env.reset()
observation = np.reshape(observation, (1, 480, 640, 3))
done = False
cropped_observation = observation[:,height_start:height_end,width_start:width_end,:]
pooled_picture = sess.run(h_prepool1, feed_dict={original_state: cropped_observation})
#


reward_discount = np.ones(maxsteps)
for rs in range(1,maxsteps):
    reward_discount[rs] = discount_factor*reward_discount[rs-1] #generate discounting vector to multiply rewards by

reward_vec = np.zeros(num_runs)
for run in range(num_runs):
    #original_state = np.zeros((1, height_end-height_start, width_end-width_start, 3), dtype='uint8')
    cropped_observation = np.zeros((1,height_end-height_start,width_end-width_start,3))
    states = np.zeros((maxsteps, 7, 128, 3), dtype='float32')
    actions = np.zeros((maxsteps,3), dtype='uint8')
    rewards = np.zeros((maxsteps,1), dtype='float32')
    weighted_sum_rewards = np.zeros((maxsteps,1), dtype='float32')
    final_rewards = np.zeros((maxsteps,1), dtype='float32')
    timestep = 0
    observation = env.reset()
    observation = np.reshape(observation, (1, 480, 640, 3))
    done = False
    cropped_observation = observation[:,height_start:height_end,width_start:width_end,:]
    pooled_picture = sess.run(h_prepool1, feed_dict={original_state: cropped_observation})
    pooled_picture = mean_subtraction(pooled_picture)/255.0




    #print np.shape(pooled_picture) #(1,35, 270,3)
    #plt.imshow(pooled_picture[0,:,:,:])
    #plt.imsave(fname='pic.jpg', arr= -pooled_picture[0,:,:,:], format='jpg')

    if run % 50 ==0:
        np.save('rewards.npy', reward_vec)

        #for i in range(10):
    skip_count = 0
    while not done and timestep < maxsteps:
        if run % 50 == 0:
            env.render()

        states[timestep,:] = pooled_picture
        if skip_count <= 0:
            action_prob = sess.run(ao, feed_dict={state: pooled_picture})[0]
            action_prob[:] = action_prob/1.01 #try to kill multinomial error
            action_vec = np.zeros(np.shape(env.action_space))
            #print np.max(action_prob)
            if np.max(action_prob)<.98:
                action = np.argmax(np.random.multinomial(1, action_prob))
            else:
                action = np.argmax(action_prob)
            if action == 0:
                action_vec[0] = 1 #shoot
            elif action == 1:
                action_vec[10] = 1 #right
            elif action == 2:
                action_vec[11] = 1 #left
        #   print 'NEW ACTION'
        #print '{} = skip count'.format(skip_count)
        skip_count += 1
        if skip_count >= nskips:
            skip_count = 0
        #print 'The action is {}'.format(action)
        #print 'The timestep is {}'.format(timestep)
        #print ''

        observation, reward, done, info = env.step(action_vec) # need to add all the data of the skips into history
        observation = np.reshape(observation, (1, 480, 640, 3))
        cropped_observation[:] = observation[:,height_start:height_end,width_start:width_end,:]
        pooled_picture = sess.run(h_prepool1, feed_dict={original_state: cropped_observation})
        pooled_picture = mean_subtraction(pooled_picture)/255.0
        

        actions[timestep, action] = 1
        rewards[timestep, :] = reward
        timestep = timestep + 1
    timestep_of_run = timestep
    timestep_for_mean = np.reshape(timestep_of_run, [1,1])

    
    states = states[:timestep_of_run, :]
    actions = actions[:timestep_of_run, :]
    
    original_rewards = rewards
    reward_vec[run] =  np.sum(original_rewards[:,0])

    rewards = rewards[:timestep_of_run,:]/150.0

    
    #print actions
    #summed_rewards[:, 0] = np.cumsum(rewards[::-1])[::-1]

    for step in range(timestep_of_run):
        weighted_sum_rewards[step,0] = rewards[step:,0].dot(reward_discount[:timestep_of_run - step])
    weighted_sum_rewards= np.reshape(weighted_sum_rewards[:timestep_of_run,0], (timestep_of_run, 1))
    #print weighted_sum_rewards

    final_rewards = weighted_sum_rewards
    
    #print np.shape(states)
    current_values = sess.run(ao_value, feed_dict={state: states})

    advantages = (final_rewards - current_values)
    #print advantages




    if run % 5 ==0:
        print 'run #:', run
        print 'The run lasted {} timesteps'.format(timestep_of_run)
        print 'reward', reward_vec[run]
    #    print ''

    for i in range(num_gradients):
        sess.run(train_step_value, feed_dict={state: states, action_choice: actions, reward_signal: final_rewards})

    for i in range(num_gradients):
        sess.run(train_step, feed_dict={state: states, action_choice: actions, advantage_signal: advantages, length_of_episode: timestep_for_mean})
