# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 21:36:05 2016

@author: mike
"""
import numpy as np
import gym
import matplotlib.pyplot as plt
env = gym.make('Pong-v0')

def reduce_image(_obs):
    new_obs = np.sum(_obs, 2)/(3.0*255)
    return new_obs[30:195,0:160]

env.reset()
for i_episode in range(1):
    observation = env.reset()
    done = False
    new_observation = reduce_image(observation)
    print np.shape(new_observation)
    plt.pcolormesh(new_observation)
    plt.show()

    
    for i in range(100):
        observation = reduce_image(observation)
        plt.pcolormesh(observation)
        plt.show()
 
        env.render()
        #print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        #print done
#env.render(close=True)

'''
x = data[:, 0:3]
y = data[:, 3]


NUM_HIDDEN_NODES = 100
NUM_EXAMPLES = len(x)
TRAIN_SPLIT = .9
MINI_BATCH_SIZE = 100
NUM_EPOCHS = 10

train_size = int(NUM_EXAMPLES*TRAIN_SPLIT)
trainX = x[:train_size]
validX = x[train_size:]
trainY = y[:train_size]
validY = y[train_size:]
trainY = trainY.reshape(len(trainY), 1)
validY = validY.reshape(len(validY), 1)



X = tf.placeholder(tf.float32, [None, 3])
Y = tf.placeholder(tf.float32, [None, 1])


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


W_l1 = weight_variable([3, NUM_HIDDEN_NODES])
b_l1 = bias_variable([1, NUM_HIDDEN_NODES])
y_l1 = tf.nn.relu(tf.matmul(X, W_l1) + b_l1)

W_l2 = weight_variable([NUM_HIDDEN_NODES, NUM_HIDDEN_NODES])
b_l2 = bias_variable([1, NUM_HIDDEN_NODES])
y_l2 = tf.nn.relu(tf.matmul(y_l1, W_l2) + b_l2)

W_l3 = weight_variable([NUM_HIDDEN_NODES, NUM_HIDDEN_NODES])
b_l3 = bias_variable([1, NUM_HIDDEN_NODES])
y_l3 = tf.nn.relu(tf.matmul(y_l2, W_l3) + b_l3)

W_o = weight_variable([NUM_HIDDEN_NODES,1])
b_o = bias_variable([1,1])

yhat = tf.matmul(y_l3, W_o) + b_o

train_op = tf.train.AdamOptimizer().minimize(tf.nn.l2_loss(yhat - Y))

sess = tf.Session()
sess.run(tf.initialize_all_variables())

#correct_prediction = tf.equal(tf.round(yhat), validY) #validY)
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

errors = []
percent_classified =[]  # todo Fix percent_classified now that I'm sending raw actions
for i in range(NUM_EPOCHS):
    for start, end in zip(range(0, len(trainX), MINI_BATCH_SIZE), range(MINI_BATCH_SIZE, len(trainX), MINI_BATCH_SIZE)):
        sess.run(train_op, feed_dict={X: trainX[start:end], Y: trainY[start:end]})
    mse = sess.run(tf.nn.l2_loss(yhat - validY),  feed_dict={X: validX})/(NUM_EXAMPLES-train_size)*2 # squared error
    #pc = sess.run(accuracy, feed_dict={X: validX})
    errors.append(mse)
    #percent_classified.append(pc)
    if i % 1 == 0:
        print "epoch %d, validation MSE %g" % (i, mse)
        #print "Percent Classified", pc
#plt.plot(errors)
#plt.xlabel('epochs')
#plt.ylabel('MSE')
#plt.show()

#print sess.run(yhat,  feed_dict={X: validX}) # squared error

def vehicledyn(x, y, theta, v, dt, utheta, uv):
    xn = x + v*np.cos(theta + utheta/2.0)*dt
    yn = y + v*np.sin(theta + utheta/2.0)*dt
    thetan = theta + utheta*dt
    vn = v + uv*dt

    if thetan >= 2*np.pi:
        thetan = 0
    elif thetan < 0:
        thetan = 2*np.pi
    return [xn, yn, thetan], vn

dt = 1
NT = 5000
v = .0025   #### need to always update

width = 2
height = width

deltatheta_max = np.pi/4
utheta_options = [-deltatheta_max, 0, deltatheta_max]


state_traj = np.zeros((NT, 3))
state = np.random.uniform(0, 1, (1, 3))
print np.shape(state)
state[0, 2] = 2*np.pi*state[0, 2]
state_traj[0, 0] = state[0, 0]
state_traj[0, 1] = state[0, 1]
state_traj[0, 2] = state[0, 2]

#num_actions = 50
for time in range(NT - 1):
    state[0,:] = state_traj[time, :]
    #state.reshape((1, 3))
    #print np.shape(state)

    action = sess.run(yhat, feed_dict={X: state})

    new_state = np.random.uniform(0, 1, (1,3))
    ns, new_v = vehicledyn(state[0,0], state[0,1], state[0,2], v, dt, action, 0)
    new_state[0,:] = ns
    #print new_state
    #print np.shape(new_state)
    state_traj[time+1, 0] = new_state[0, 0]
    state_traj[time + 1, 1] = new_state[0,1]
    state_traj[time+1, 2] = new_state[0,2]

plt.plot(state_traj[:, 0], state_traj[:,1])
plt.scatter(state_traj[0, 0], state_traj[0, 1])
plt.show()
'''