import pickle
import numpy as np
#from numpy import vstack
from numpy import zeros
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tensorflow as tf
import cv2

import collections

import random
from random import shuffle

import sys

def compute_saliency_maps(X, y):
    """
    Compute a class saliency map using the model for images X and labels y.
    
    Input:
    - X: Input images, numpy array of shape (N, H, W, 3)
    - y: Labels for X, numpy of shape (N,)
    - model: A SqueezeNet model that will be used to compute the saliency map.
    
    Returns:
    - saliency: A numpy array of shape (N, H, W) giving the saliency maps for the
    input images.
    """
    saliency = None
    # Compute the score of the correct class for each example.
    # This gives a Tensor with shape [N], the number of examples.
    #
    # Note: this is equivalent to scores[np.arange(N), y] we used in NumPy
    # for computing vectorized losses.
    #correct_scores = tf.gather_nd(tf.stack((tf.range(X, shape[0], y), axis=1)))
    # correct_scores = tf.gather_nd(model.classifier,
    #                               tf.stack((tf.range(X.shape[0]), model.labels), axis=1))
    ###############################################################################
    # TODO: Implement this function. You should use the correct_scores to compute #
    # the loss, and tf.gradients to compute the gradient of the loss with respect #
    # to the input image stored in model.image.                                   #
    # Use the global sess variable to finally run the computation.                #
    # Note: model.image and model.labels are placeholders and must be fed values  #
    # when you call sess.run().                                                   #
    ###############################################################################
    N, H, W, _ = X.shape
    grad = tf.abs(tf.gradients(y, [X])[0])
    #saliency = tf.reduce_max(grad, [-1])
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return grad

class network():
    def __init__(self, data, labels, test_data, test_labels):
        self.data = data
        self.labels = labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.data_len = np.prod(data[0].shape)
        print(data[0].shape)
        print(self.data_len)
        self.label_len = np.prod(labels[0].shape)
        dat_shape = [None] + list(data[0].shape)
        
        lab_shape = [None] + list(labels[0].shape)

        self.x = tf.placeholder(tf.float32, shape=dat_shape, name="x")
        self.y_= tf.placeholder(tf.float32, shape=lab_shape, name="y_")
        x = tf.layers.conv2d(self.x, filters=10 ,kernel_size=[8, 8], strides=(4, 4))
        x = tf.layers.average_pooling2d(x, [10,10], [5, 5])
        flatten_shape = int(np.prod(x.shape[1:]))
        print(flatten_shape)
        rs = tf.reshape(x,  shape = [-1, flatten_shape])

        dense_layer = tf.layers.dense(rs, units=self.label_len)
        
        y = tf.nn.softmax(dense_layer, name='softmax')
        cross_entropy = -tf.reduce_sum(self.y_*tf.log(y))
        starter_learning_rate = 1e-5
        learning_rate = tf.train.exponential_decay(starter_learning_rate, 1, 1000, 0.9, staircase=True)
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
        self.correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
        self.init = tf.global_variables_initializer()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        for variable in variables:
            var_norm = tf.norm(variable)
            tf.summary.scalar("/grad_norm/" + variable.name, var_norm)

        loss_saliency = compute_saliency_maps(self.x, cross_entropy)
        loss_summary = tf.summary.scalar("loss", cross_entropy)
        accuracy_summary = tf.summary.scalar("accuracy", self.accuracy)
        loss_saliency_summary = tf.summary.image("saliency", loss_saliency)
        image = tf.summary.image("image", self.x)
        self.summary_op = tf.summary.merge_all()
        
    def train(self, savefile="0"):
        sess = tf.Session()
        sess.run(self.init)
        saver = tf.train.Saver()
        # create log writer object
        writer_train = tf.summary.FileWriter('temp/' + str(savefile) + '/train', sess.graph)
        writer_test = tf.summary.FileWriter('temp/' + str(savefile) + '/val', sess.graph)

        for epoch in range(1000):
            for i in range(100):
                val_indices = random.sample(range(0, len(self.test_labels)), 130)
                batch_xs, batch_ys = self.data[i*130:(i+1)*130], self.labels[i*130:(i+1)*130]
                
                batch_xs_test, batch_ys_test = self.test_data[val_indices], self.test_labels[val_indices]
                _, sumr = sess.run([self.train_step, self.summary_op], feed_dict={self.x:batch_xs, self.y_: batch_ys})
                writer_train.add_summary(sumr, (epoch * 100) + i)
                vsumr = sess.run(self.summary_op, feed_dict={self.x: batch_xs_test, self.y_: batch_ys_test})
                writer_test.add_summary(vsumr, (epoch * 100) + i)
                # write log

            print("Done epoch " + str(epoch))
            saver.save(sess, 'temp/' + str(savefile) + '/checkpoint', global_step=epoch)
            


print(sys.argv[1])
data1 = pickle.load(open('train_dataset.pkl', 'rb'))
data2 = pickle.load(open('train_dataset2.pkl', 'rb'))

separated = [list(t) for t in zip(*(data1+data2))]
print("The total size = " + str(len(separated[0])))
images, labels = separated[0], separated[1]

test1 = pickle.load(open('test_dataset.pkl', 'rb'))
test2 = pickle.load(open('test_dataset2.pkl', 'rb'))
testseparated = [list(t) for t in zip(*(test1 + test2))]
print("The total test size = " + str(len(testseparated[0])))
testimages, testlabels = testseparated[0], testseparated[1]

n = network(np.asarray(images), np.asarray(labels), np.asarray(testimages), np.asarray(testlabels))
n.train(sys.argv[1])

