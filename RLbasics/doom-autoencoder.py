import numpy as np
import gym
import tensorflow as tf
import matplotlib.pyplot as plt
#from scipy import ndimage

#Questions: what does it take to autoencode full frame vs downsampled vs blurred?

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.2, shape=shape)
  return tf.Variable(initial)

observations = np.load(file = 'observations.npy')
stride = 10
num_pics = 100
downsampled = observations[0,:num_pics,::stride,::stride,:]/255.0


plt.imshow(downsampled[0])
plt.show()



input_dim = np.shape(downsampled) #640/stride*480/stride*3



downsampled = np.reshape(downsampled, [input_dim[0],-1])


flattened_dim = 640/stride*480/stride*3 #np.shape(downsampled)[1]#input_dim[0]*input_dim[1]*input_dim[2]



num_nodes1 = 500
num_nodes2 = 100
num_nodes3 = 500
num_batches = 50
num_gradients = 5


sess = tf.InteractiveSession()


state = tf.placeholder(tf.float32, shape=[None, flattened_dim])




W1 = weight_variable([flattened_dim, num_nodes1])
b1 = bias_variable([num_nodes1])
a1 = tf.nn.relu(tf.matmul(state, W1) + b1)

W2 = weight_variable([num_nodes1, num_nodes2])
b2 = bias_variable([num_nodes2])
a2 = tf.nn.relu(tf.matmul(a1, W2) + b2)

W3 = weight_variable([num_nodes2, num_nodes3])
b3 = bias_variable([num_nodes3])
a3 = tf.nn.relu(tf.matmul(a2, W3) + b3)

Wo = weight_variable([num_nodes3, flattened_dim])
bo = bias_variable([flattened_dim])
ao = tf.nn.relu(tf.matmul(a3, Wo) + bo)


loss = tf.nn.l2_loss(ao - state)
train_step = tf.train.AdamOptimizer().minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


#print sess.run(a1, feed_dict={state: downsampled})
#print sess.run(ao, feed_dict={state: downsampled})
for cycles in range(num_batches):
	print 'cycle is:', cycles
	for pic in range(num_pics):
		#print 'pic is ', pic
		batch = np.reshape(downsampled[pic], [1,flattened_dim])
		#print sess.run(loss, feed_dict={state: batch})
		for i in range(num_gradients):
			sess.run(train_step, feed_dict={state: batch})
			#if i %  ==0:
			#	print i
			#	print sess.run(loss, feed_dict={state: batch})
			#	print ''
	if cycles % 1 == 0:
		pic = 0
		batch = np.reshape(downsampled[pic], [1,flattened_dim])
		final_answer = sess.run(ao, feed_dict={state: batch})
		final_image = np.reshape(final_answer, input_dim[1:])
		plt.imshow(final_image)
		plt.show()




