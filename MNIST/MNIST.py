# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 08:37:50 2016
Training MNIST with NNs with Keras and TensorFlow

@author: mike
"""

import tensorflow as tf
sess = tf.Session()
from keras import backend as K
print K.learning_phase()

K.set_session(sess)

img = tf.placeholder(tf.float32, shape = (None, 784))

from keras.layers import Dense
from keras.layers import Dropout

x = Dense(128, activation = 'relu')(img)  #originally 128
x = Dropout(0.5)(x)
x = Dense(128, activation = 'relu')(x)
x = Dropout(0.5)(x)
preds = Dense(10, activation = 'softmax')(x)

labels = tf.placeholder(tf.float32, shape = (None,10))

from keras.objectives import categorical_crossentropy
loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

from tensorflow.examples.tutorials.mnist import input_data
mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

train_step = tf.train.GradientDescentOptimizer(.5).minimize(loss)
with sess.as_default():
    for j in range(10):
        for i in range(100):
            batch = mnist_data.train.next_batch(50)
            train_step.run(feed_dict = {img: batch[0], labels: batch[1],
                                        K.learning_phase(): 1})

from keras.metrics import categorical_accuracy as accuracy

acc_value = accuracy(labels, preds)
with sess.as_default():
    print acc_value.eval(feed_dict={img: mnist_data.test.images,
                                    labels: mnist_data.test.labels,
                                    K.learning_phase(): 0})