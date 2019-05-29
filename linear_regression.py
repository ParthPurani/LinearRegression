'''
#2.7 folks
from __future__ import absolute_import, print_function
'''
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
rng = np.random

#Paramrters for learning
lr = 0.001
epochs = 100
display_step = 50

#Locations
train_csv = "E:\\kaggle_pubg\\train.csv"
test_csv = "E:\\kaggle_pubg\\test.csv"

df = pd.read_csv(train_csv)
dff = pd.read_csv(test_csv)
#Training Data
train_X = np.asarray(df['kills'])

train_Y = np.asarray(df['walkDistance'])

n_samples = train_X.shape[0]

#Input fror tensorflow graph
X= tf.placeholder("float")
Y= tf.placeholder("float")

#Presetting model weights and biases
w = tf.Variable(rng.randn(), name= "weight")
b = tf.Variable(rng.randn(), name = "bias")

#Model time
pred = tf.add(tf.multiply(X, w), b)

#Defining Cost function
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)

#Optimizer
optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cost)

#Initializers
init = tf.global_variables_initializer()

#Trainig

with tf.Session() as sess:

    #Run Initializers
    sess.run(init)

    #Fit x
    for epoch in range(epochs):
        for (x,y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X:x, Y:y})

        if (epochs+1) % display_step == 0:
            c= sess.run(cost, feed_dict = {X:train_X, Y:train_Y})
            print("Epoch:", '%04d' % (epochs+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(w), "b=", sess.run(b))
    print("optimization finished")
    training_cost = sess.run(cost, feed_dict = {X:train_X, Y:train_Y})
    print("Training cost=", training_cost, "W=", sess.run(w), "b=", sess.run(b), '\n')

    #It's Plotting time
    plt.plot(train_X, train_Y, 'ro', label='Example Data')
    plt.plot(train_X, sess.run(w) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()

    #Testing n_samples
    test_X = np.asarray(dff['kills'])
    test_Y = np.asarray(dff['walkDistance'])

    print("Testing...")
    testing_cost = sess.run(
    tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
    feed_dict={X: test_X, Y: test_Y})  # same function as cost above
    print("Testing cost=", testing_cost)
    print("Absolute mean square loss difference:", abs(
        training_cost - testing_cost))

    plt.plot(test_X, test_Y, 'bo', label='Testing data')
    plt.plot(train_X, sess.run(w) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
