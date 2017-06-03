import cv2
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#-------------- Weight Initialization ---------------#
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


def main(test_data):

    # Placeholders
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])


    # First Convolutional Layer
    W_conv1 = weight_variable([5,5,1,32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1,28,28,1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)


    # Second Convolutional Layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)


    # Densely Connected Layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


    # Dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


    # Readout Layer
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    y = tf.nn.softmax(y_conv)


    # Restore model
    saver = tf.train.Saver({"W_conv1": W_conv1, "b_conv1": b_conv1,
                            "W_conv2": W_conv2, "b_conv2": b_conv2,
                            "W_fc1": W_fc1, "b_fc1": b_fc1,
                            "W_fc2": W_fc2, "b_fc2": b_fc2})

    with tf.Session() as sess:
        saver.restore(sess, "/Users/AnYameng/Documents/OCR/mnist/model.ckpt")
        # prediction = sess.run(y, feed_dict={x: test_data, keep_prob: 1.0})

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

        # print(tf.argmax(prediction, 1).eval())


if __name__ == '__main__':
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    test_data = mnist.test.images[100:101]


    # test_image = Image.open("result_bw.png").convert('L')
    # test_data = 1 - np.asarray(test_image) / 255
    image = cv2.resize(test_data, (28, 28))
    test_data = (test_data.flatten().reshape(784, -1)).transpose()
    # test_data = test_data.transpose()
    print(test_data.shape)
    main(test_data)


    # path = "/Users/AnYameng/Downloads/Character_Segmentation-master/segmented_img/img1"
    # files = os.listdir(path)
    # for file in files:
    #     if file[0] != '.':
    #         print(file)
    #         img = cv2.imread(path + '/' + file)
    #         # print(img.shape)
    #         img = cv2.resize(img, (28, 28))
    #         test_data = (img.flatten().reshape(784, -1)).transpose()
    #         main(test_data)
        #
        # test_data = (img.flatten().reshape(784, -1)).transpose()
        # main(test_data)


    # image = cv2.imread(image_path) / 255.0
    # image = cv2.resize(image, (28,28), interpolation=cv2.INTER_CUBIC)
    # brightness = 0.299*image[:,:,2] + 0.587*image[:,:,1] + 0.114*image[:,:,0]
    # img = 1-brightness
    # test_data = (img.flatten().reshape(784, -1)).transpose()

    # main(test_data)
