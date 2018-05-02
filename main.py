""" 
Net2Net using Tensorflow

  @ Test in MNIST dataset
  1. Train a teacher network
  2. Train a student network (Net2Wider)
    - # of filters in 'conv1' layer [32->128]
  3. Random pad (Net2Wider baseline)
  4. Train a student network (Net2Deeper)
    - Insert a new layer after 'conv1' layer
  5. Random initialization (Net2Deeper baseline)

  NOTE: All parameters are fixed expect new weights from Net2Net.

Author: Kyunghyun Paeng

"""
import numpy as np
import scipy.signal
import tensorflow as tf
from slim import ops
from slim import scopes
from slim import variables
from net2net import Net2Net

from tensorflow.examples.tutorials.mnist import input_data

MODEL='./my-model-500.meta'
WEIGHT='./my-model-500'
BATCH_SIZE = 50
MAX_ITER = 1000
TEST_ITER = 500
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def train_a_student_network_deeper():
    new_w1, new_b1 = tf_net2deeper(MODEL, WEIGHT, 'conv1')
    with tf.Graph().as_default():
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            x = tf.placeholder(tf.float32, shape=[None, 784])
            y_ = tf.placeholder(tf.float32, shape=[None, 10])
            x_image = tf.reshape(x, [-1,28,28,1])
            net = ops.conv2d(x_image, 32, [5, 5], scope='conv1')
            net = ops.conv2d(net, 32, [5, 5], scope='conv1_new', initializer='constant', weights=new_w1, bias=new_b1, restore=False)
            net = ops.max_pool(net, [2, 2], scope='pool1')
            net = ops.conv2d(net, 64, [5, 5], scope='conv2')
            net = ops.max_pool(net, [2, 2], scope='pool2')
            net = ops.flatten(net, scope='pool2_flat')
            net = ops.fc(net, 1024, scope='fc1')
            net = ops.fc(net, 10, activation=None, scope='fc2')
            y_conv = tf.nn.softmax(net)
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), axis=[1]))
            model = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
            correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('loss', cross_entropy)
            tf.summary.scalar('acc', accuracy)
            merged = tf.summary.merge_all()
            saver = tf.train.Saver()
            writer = tf.summary.FileWriter('./logs-deeper', sess.graph)
            sess.run(tf.global_variables_initializer())
            variables_to_restore = tf.get_collection(variables.VARIABLES_TO_RESTORE)
            saver = tf.train.Saver(variables_to_restore)
            saver.restore(sess, WEIGHT)
            print('Net2Deeper...')
            for i in range(MAX_ITER):
                batch = mnist.train.next_batch(BATCH_SIZE)
                sess.run(model, feed_dict={x: batch[0], y_: batch[1]})
                if i % 100 == 0:
                    summary_str, acc = sess.run([merged, accuracy], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                    writer.add_summary(summary_str, i)
                    print('[Iter: {}] Validation Accuracy : {:.4f}'.format(i,acc))

def train_a_student_network_deeper_rand_init():
    with tf.Graph().as_default():
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            x = tf.placeholder(tf.float32, shape=[None, 784])
            y_ = tf.placeholder(tf.float32, shape=[None, 10])
            x_image = tf.reshape(x, [-1,28,28,1])
            net = ops.conv2d(x_image, 32, [5, 5], scope='conv1', stddev=0.1, bias=0.1)
            net = ops.conv2d(net, 32, [5, 5], scope='conv1_new', stddev=0.1, bias=0.1, restore=False)
            net = ops.max_pool(net, [2, 2], scope='pool1')
            net = ops.conv2d(net, 64, [5, 5], scope='conv2', stddev=0.1, bias=0.1)
            net = ops.max_pool(net, [2, 2], scope='pool2')
            net = ops.flatten(net, scope='pool2_flat')
            net = ops.fc(net, 1024, scope='fc1', stddev=0.1, bias=0.1)
            net = ops.fc(net, 10, activation=None, scope='fc2', stddev=0.1, bias=0.1)
            y_conv = tf.nn.softmax(net)
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), axis=[1]))
            model = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
            correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('loss', cross_entropy)
            tf.summary.scalar('acc', accuracy)
            merged = tf.summary.merge_all()
            saver = tf.train.Saver()
            writer = tf.summary.FileWriter('./logs-deeper-rand', sess.graph)
            sess.run(tf.global_variables_initializer())
            variables_to_restore = tf.get_collection(variables.VARIABLES_TO_RESTORE)
            saver = tf.train.Saver(variables_to_restore)
            saver.restore(sess, WEIGHT)
            print('Net2Deeper Baseline (Rand init)...')
            for i in range(MAX_ITER):
                batch = mnist.train.next_batch(BATCH_SIZE)
                sess.run(model, feed_dict={x: batch[0], y_: batch[1]})
                if i % 100 == 0:
                    summary_str, acc = sess.run([merged, accuracy], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                    writer.add_summary(summary_str, i)
                    print('[Iter: {}] Validation Accuracy : {:.4f}'.format(i,acc))

def train_a_student_network_wider():
    new_width_conv = 128
    new_w1, new_b1, new_w2, new_b2 = tf_net2wider(MODEL, WEIGHT, 'conv1', 'conv2', new_width_conv)
    with tf.Graph().as_default():
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            x = tf.placeholder(tf.float32, shape=[None, 784])
            y_ = tf.placeholder(tf.float32, shape=[None, 10])
            x_image = tf.reshape(x, [-1,28,28,1])
            net = ops.conv2d(x_image, new_width_conv, [5, 5], scope='conv1', initializer='constant', weights=new_w1, bias=new_b1, restore=False)
            net = ops.max_pool(net, [2, 2], scope='pool1')
            net = ops.conv2d(net, 64, [5, 5], scope='conv2', initializer='constant', weights=new_w2, bias=new_b2, restore=False)
            net = ops.max_pool(net, [2, 2], scope='pool2')
            net = ops.flatten(net, scope='pool2_flat')
            net = ops.fc(net, 1024, scope='fc1')
            net = ops.fc(net, 10, activation=None, scope='fc2')
            y_conv = tf.nn.softmax(net)
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), axis=[1]))
            model = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
            correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('loss', cross_entropy)
            tf.summary.scalar('acc', accuracy)
            merged = tf.summary.merge_all()
            saver = tf.train.Saver()
            writer = tf.summary.FileWriter('./logs-wider', sess.graph)
            sess.run(tf.global_variables_initializer())
            variables_to_restore = tf.get_collection(variables.VARIABLES_TO_RESTORE)
            saver = tf.train.Saver(variables_to_restore)
            saver.restore(sess, WEIGHT)
            print('Net2Wider...')
            for i in range(MAX_ITER):
                batch = mnist.train.next_batch(BATCH_SIZE)
                sess.run(model, feed_dict={x: batch[0], y_: batch[1]})
                if i % 100 == 0:
                    summary_str, acc = sess.run([merged, accuracy], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                    writer.add_summary(summary_str, i)
                    print('[Iter: {}] Validation Accuracy : {:.4f}'.format(i,acc))

def train_a_student_network_wider_rand_pad():
    new_width_conv = 128
    new_w1, new_b1, new_w2, new_b2 = tf_net2wider_rand(MODEL, WEIGHT, 'conv1', 'conv2', new_width_conv)
    with tf.Graph().as_default():
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            x = tf.placeholder(tf.float32, shape=[None, 784])
            y_ = tf.placeholder(tf.float32, shape=[None, 10])
            x_image = tf.reshape(x, [-1,28,28,1])
            net = ops.conv2d(x_image, new_width_conv, [5, 5], scope='conv1', initializer='constant', weights=new_w1, bias=new_b1, restore=False)
            net = ops.max_pool(net, [2, 2], scope='pool1')
            net = ops.conv2d(net, 64, [5, 5], scope='conv2', initializer='constant', weights=new_w2, bias=new_b2, restore=False)
            net = ops.max_pool(net, [2, 2], scope='pool2')
            net = ops.flatten(net, scope='pool2_flat')
            net = ops.fc(net, 1024, scope='fc1', stddev=0.1, bias=0.1)
            net = ops.fc(net, 10, activation=None, scope='fc2', stddev=0.1, bias=0.1)
            y_conv = tf.nn.softmax(net)
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), axis=[1]))
            model = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
            correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('loss', cross_entropy)
            tf.summary.scalar('acc', accuracy)
            merged = tf.summary.merge_all()
            saver = tf.train.Saver()
            writer = tf.summary.FileWriter('./logs-wider-rand', sess.graph)
            sess.run(tf.global_variables_initializer())
            variables_to_restore = tf.get_collection(variables.VARIABLES_TO_RESTORE)
            saver = tf.train.Saver(variables_to_restore)
            saver.restore(sess, WEIGHT)
            print('Net2Wider Baseline (Rand pad)...')
            for i in range(MAX_ITER):
                batch = mnist.train.next_batch(BATCH_SIZE)
                sess.run(model, feed_dict={x: batch[0], y_: batch[1]})
                if i % 100 == 0:
                    summary_str, acc = sess.run([merged, accuracy], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                    writer.add_summary(summary_str, i)
                    print('[Iter: {}] Validation Accuracy : {:.4f}'.format(i,acc))

def train_a_teacher_network():
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    x_image = tf.reshape(x, [-1,28,28,1])
    net = ops.conv2d(x_image, 32, [5, 5], scope='conv1', stddev=0.1, bias=0.1)
    net = ops.max_pool(net, [2, 2], scope='pool1')
    net = ops.conv2d(net, 64, [5, 5], scope='conv2', stddev=0.1, bias=0.1)
    net = ops.max_pool(net, [2, 2], scope='pool2')
    net = ops.flatten(net, scope='pool2_flat')
    net = ops.fc(net, 1024, scope='fc1', stddev=0.1, bias=0.1)
    net = ops.fc(net, 10, activation=None, scope='fc2', stddev=0.1, bias=0.1)
    y_conv = tf.nn.softmax(net)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), axis=[1]))
    model = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('loss', cross_entropy)
    tf.summary.scalar('acc', accuracy)
    merged = tf.summary.merge_all()
    saver = tf.train.Saver()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        writer = tf.summary.FileWriter('./logs', sess.graph)
        sess.run(tf.global_variables_initializer())
        print('Teacher Network...')
        for i in range(MAX_ITER):
            batch = mnist.train.next_batch(BATCH_SIZE)
            sess.run(model, feed_dict={x: batch[0], y_: batch[1]})
            # saver.save(sess, './my-model', global_step=TEST_ITER)
            if i % 100 == 0:
                summary_str, acc = sess.run([merged, accuracy], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                writer.add_summary(summary_str, i)
                print('[Iter: {}] Validation Accuracy : {:.4f}'.format(i,acc))
                saver.save(sess, './my-model', global_step=TEST_ITER)

def load_teacher_net(sess, model, weights):
    saver = tf.train.import_meta_graph(model)
    saver.restore(sess, weights)
    return sess.graph

def get_weight_bias_of_layer(net, layer_name, numpy=True):
    layer_name = [ op.name for op in net.get_operations()
                   if layer_name+'/weights'==op.name
                   or layer_name+'/biases'==op.name ]
    assert len(layer_name) == 2, 'Check layer name'
    weights = net.get_tensor_by_name(layer_name[0]+':0')
    biases = net.get_tensor_by_name(layer_name[1]+':0')
    if numpy:
        return weights.eval(), biases.eval()
    else:
        return weights, biases

def tf_net2wider(model, weight, target_layer, next_layer, new_width):
    n2n = Net2Net()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        net = load_teacher_net(sess, model, weight)
        w1, b1 = get_weight_bias_of_layer(net, target_layer)
        w2, b2 = get_weight_bias_of_layer(net, next_layer)
        nw1, nb1, nw2 = n2n.wider(w1, b1, w2, new_width, True)
    return nw1, nb1, nw2, b2

def tf_net2wider_rand(model, weight, target_layer, next_layer, new_width):
    n2n = Net2Net()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        net = load_teacher_net(sess, model, weight)
        w1, b1 = get_weight_bias_of_layer(net, target_layer)
        w2, b2 = get_weight_bias_of_layer(net, next_layer)
        nw1, nb1, nw2 = n2n.wider_rand(w1, b1, w2, new_width)
    return nw1, nb1, nw2, b2

def tf_net2deeper(model, weight, target_layer):
    n2n = Net2Net()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        net = load_teacher_net(sess, model, weight)
        w1, b1 = get_weight_bias_of_layer(net, target_layer)
        new_w, new_b = n2n.deeper(w1, True)
    return new_w, new_b

if __name__ == '__main__':
    # 1. Train a teacher network
    train_a_teacher_network()
    # 2. Train a student network (Net2Wider)
    train_a_student_network_wider()
    # 3. Random pad (Net2Wider baseline)
    train_a_student_network_wider_rand_pad()
    # 4. Train a student network (Net2Deeper)
    train_a_student_network_deeper()
    # 5. Random initialization (Net2Deeper baseline)
    train_a_student_network_deeper_rand_init()
