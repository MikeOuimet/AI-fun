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
stride = 50
downsampled = observations[0,0,::stride,::stride,:]

plt.imshow(downsampled)
plt.show()

input_dim = np.shape(downsampled)

downsampled = np.reshape(downsampled, [1,-1])



flattened_dim = np.shape(downsampled)[1]#input_dim[0]*input_dim[1]*input_dim[2]


num_nodes = 10000
num_gradients = 500


sess = tf.InteractiveSession()


state = tf.placeholder(tf.float32, shape=[None, flattened_dim])




W1 = weight_variable([flattened_dim, num_nodes])
b1 = bias_variable([num_nodes])
a1 = tf.nn.relu(tf.matmul(state, W1) + b1)

W2 = weight_variable([num_nodes, num_nodes])
b2 = bias_variable([num_nodes])
a2 = tf.nn.relu(tf.matmul(a1, W2) + b2)

W3 = weight_variable([num_nodes, num_nodes])
b3 = bias_variable([num_nodes])
a3 = tf.nn.relu(tf.matmul(a2, W3) + b3)

Wo = weight_variable([num_nodes, flattened_dim])
bo = bias_variable([flattened_dim])
ao = tf.nn.relu(tf.matmul(a3, Wo) + bo)


loss = tf.nn.l2_loss(ao - state)
train_step = tf.train.AdamOptimizer().minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


print sess.run(loss, feed_dict={state: downsampled})

for i in range(num_gradients):
        sess.run(train_step, feed_dict={state: downsampled})
 	if i % 50 ==0:
 		print i
 		print sess.run(loss, feed_dict={state: downsampled})
 		print ''


final_answer = sess.run(ao, feed_dict={state: downsampled})
final_image = np.reshape(final_answer, input_dim)




plt.imshow(final_image)
plt.show()
