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

patch_size = 2

num_gradients = 2
maxsteps =400
num_runs = 1000
nskips = 4

height_start = 180
height_end = 250
width_start = 50
width_end = 590

# State placeholders
original_state = tf.placeholder(tf.float32, shape=[None, height_end-height_start, width_end-width_start, 3])
state = tf.placeholder(tf.float32, shape=[None, 35, 270, 3])
action_choice = tf.placeholder(tf.float32, shape=[None, 3])  #potentially fix
reward_signal = tf.placeholder(tf.float32, shape=(None, 1))
advantage_signal = tf.placeholder(tf.float32, shape=(None, 1))

#Pooling pre-processing
h_prepool1 = mean_pool_nxm(original_state, 1, 2)
h_prepool2 = mean_pool_nxm(h_prepool1, 2 , 1)
#h_prepool3 = mean_pool_nxm(h_prepool2, 2 , 1)


#Shared network
W_conv1 = weight_variable([patch_size, patch_size, 3, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(state, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([patch_size, patch_size, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)  # ends at (None, 3, 17, 64)



W_fc1 = weight_variable([3264, 1024])
b_fc1 = bias_variable([1024])

h_pool3_flat = tf.reshape(h_pool2, [-1, 3264])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

#Policy End of network
W_fc2 = weight_variable([1024, 3])
b_fc2 = bias_variable([3])
ao = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)
log_prob = tf.log(tf.diag_part(tf.matmul(ao, tf.transpose(action_choice))))# fix this so it doesn't need diag
log_prob = tf.reshape(log_prob, (1,-1))
loss = tf.matmul(log_prob, advantage_signal)
loss = tf.reshape(loss, [-1]) # is abs right?
train_step = tf.train.RMSPropOptimizer(1e-4).minimize(loss)

# Value End of network
W_fc2_value = weight_variable([1024, 1])
b_fc2_value = bias_variable([1])
ao_value = tf.matmul(h_fc1, W_fc2_value) + b_fc2_value
loss_value = tf.nn.l2_loss(ao_value - reward_signal)
train_step_value = tf.train.AdamOptimizer().minimize(loss_value)


init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

print h_prepool1
print h_prepool2
#print h_prepool3
#print ''
print h_pool1
print h_pool2


reward_discount = np.ones(maxsteps)
for rs in range(1,maxsteps):
    reward_discount[rs] = discount_factor*reward_discount[rs-1] #generate discounting vector to multiply rewards by

for run in range(num_runs):
	#original_state = np.zeros((1, height_end-height_start, width_end-width_start, 3), dtype='uint8')
	cropped_observation = np.zeros((1,height_end-height_start,width_end-width_start,3))
	states = np.zeros((maxsteps, 35, 270, 3), dtype='float32')
	actions = np.zeros((maxsteps,3), dtype='uint8')
	rewards = np.zeros((maxsteps,1), dtype='float32')
	weighted_sum_rewards = np.zeros((maxsteps,1), dtype='float32')
	final_rewards = np.zeros((maxsteps,1), dtype='float32')
	timestep = 0
	observation = env.reset()
	observation = np.reshape(observation, (1, 480, 640, 3))
	done = False
	cropped_observation = observation[:,height_start:height_end,width_start:width_end,:]
	pooled_picture = sess.run(h_prepool2, feed_dict={original_state: cropped_observation})
	#print np.shape(pooled_picture) #(1,35, 270,3)
    #plt.imshow(pooled_picture[0,:,:,:])
    #plt.imsave(fname='pic.jpg', arr= -pooled_picture[0,:,:,:], format='jpg')
   
    #for i in range(10):
	skip_count = 0
	while not done and timestep < maxsteps:
		states[timestep,:] = pooled_picture
		if skip_count <= 0:
			action_prob = sess.run(ao, feed_dict={state: pooled_picture})
			action_vec = np.zeros(np.shape(env.action_space))
			#print np.max(action_prob)
			if np.max(action_prob)<.98:
				action = np.argmax(np.random.multinomial(1, action_prob[0]))
			else:
				action = np.argmax(action_prob)
			if action == 0:
				action_vec[0] = 1 #shoot
			elif action == 1:
				action_vec[10] = 1 #right
			elif action == 2:
				action_vec[11] = 1 #left
		#	print 'NEW ACTION'
		#print '{} = skip count'.format(skip_count)
		skip_count += 1
		if skip_count >= nskips:
			skip_count = 0
		#print 'The action is {}'.format(action)
		#print 'The timestep is {}'.format(timestep)
		#print ''

		observation, reward, done, info = env.step(action_vec) # need to add all the data of the skips into history
		observation = np.reshape(observation, (1, 480, 640, 3))
		cropped_observation[:] = observation[:,height_start:height_end,width_start:width_end,:]/255.0
		pooled_picture = sess.run(h_prepool2, feed_dict={original_state: cropped_observation})
		

		actions[timestep, action] = 1
		rewards[timestep, :] = reward
		timestep = timestep + 1
	timestep_of_run = timestep
	print 'The run lasted {} timesteps'.format(timestep_of_run)
	states = states[:timestep_of_run, :]
	actions = actions[:timestep_of_run, :]
	rewards = rewards[:timestep_of_run,:]/200
	#print actions
	#summed_rewards[:, 0] = np.cumsum(rewards[::-1])[::-1]

	for step in range(timestep_of_run):
	    weighted_sum_rewards[step,0] = rewards[step:,0].dot(reward_discount[:timestep_of_run - step])
	weighted_sum_rewards= np.reshape(weighted_sum_rewards[:timestep_of_run,0], (timestep_of_run, 1))
	#print weighted_sum_rewards

	final_rewards = weighted_sum_rewards
	current_values = sess.run(ao_value, feed_dict={state: states})
	#print current_values
	#print ''
	advantages = (final_rewards - current_values)
	#print advantages

	print 'value function loss'
	print sess.run(loss_value, feed_dict={state: states, reward_signal: final_rewards})
	for i in range(num_gradients):
		sess.run(train_step_value, feed_dict={state: states, action_choice: actions, reward_signal: final_rewards})
		print sess.run(loss_value, feed_dict={state: states, reward_signal: final_rewards})
	#print ''


    

    #if run % 50 == 0:
    #    print 'run #: ', run
    #    print 'Time lasted: ', timestep
    #    print 'value function loss', sess.run(loss_value, feed_dict={state: states, reward_signal: final_rewards})
    #    print 'policy function loss', sess.run(loss, feed_dict={state: states, action_choice: actions, advantage_signal: advantages})
    #    print ''
	print 'policy function loss'
	print sess.run(loss, feed_dict={state: states, action_choice: actions, advantage_signal: advantages})
	#print sess.run(ao, feed_dict={state: states, action_choice: actions, advantage_signal: advantages})
	#print actions
	#print sess.run(log_prob, feed_dict={state: states, action_choice: actions, advantage_signal: advantages})
	for i in range(num_gradients):
		sess.run(train_step, feed_dict={state: states, action_choice: actions, advantage_signal: advantages})
		print sess.run(loss, feed_dict={state: states, action_choice: actions, advantage_signal: advantages})
	print ''

'''
#timestep_learning[run] = timestep

#env.monitor.close()
#env.render(close=True)
#plt.plot(timestep_learning)
#plt.show()

#env.monitor.close()
'''
