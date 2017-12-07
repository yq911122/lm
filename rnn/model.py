""" Dynamic Recurrent Neural Network.
TensorFlow implementation of a Recurrent Neural Network (LSTM) that performs
dynamic computation over sequences with variable length. This example is using
a toy dataset to classify linear sequences. The generated sequences have
variable length.
Links:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function

import tensorflow as tf
import random
from reader import load
from tensorflow.python import debug as tf_debug
import numpy as np
# ==========
#   MODEL
# ==========

# Parameters
learning_rate = 0.01
training_steps = 10000
batch_size = 2
display_step = 200

# Network Parameters
# seq_max_len = 20 # Sequence max length
n_hidden = 64 # hidden layer num of features
# n_classes = 2 # linear sequence or not

trainset, _, testset = load()
n_classes = trainset.num_ner_classes
seq_max_len = trainset.maxlen
x_dim = trainset.dim

# tf Graph input
x = tf.placeholder("float", [None, seq_max_len, x_dim], name="x")
y = tf.placeholder("float", [None, n_classes], name="y")
# A placeholder for indicating each sequence length
seqlen = tf.placeholder(tf.int32, [None], name="seqlen")
mask = tf.placeholder(tf.int32, [None], name="mask")

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([2 * n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

sess = tf.InteractiveSession()

def dynamicRNN(x, seqlen, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, seq_max_len, 1)

    # Forward direction cell
    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output, providing 'sequence_length' will perform dynamic
    # calculation.
    # outputs of shape [seq_max_len, batch_size, n_hidden]
    
    try:
        outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                              dtype=tf.float32)
    except Exception: # Old TensorFlow version only returns outputs not states
        outputs = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                        dtype=tf.float32)
    
    outputs = [tf.matmul(output, weights['out']) + biases['out'] for output in outputs]
    outputs = tf.stack(outputs)
    # outputs of shape [batch_size, seq_max_len, n_class]
    outputs = tf.transpose(outputs, [1, 0, 2])
    one = tf.constant(1, dtype=tf.int32)
    index = tf.where(tf.equal(mask, one))
    index = tf.reshape(index, [-1])
    
    # outputs of shape [n_utterance, n_class]
#     outputs = tf.gather(tf.reshape(outputs, [-1, n_classes]), index)
    return outputs, index

def dynamicRNNCost(outputs, y):
    # outputs of shape [n_utterance, classes]
    # y of shape [n_utterance, classes]

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=y))
    return cost

def dynamicRNNAccuracy(outputs, y):
    # Evaluate model
    correct_pred = tf.equal(tf.argmax(outputs,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy

def _print_debug_info(sess, batch_x, batch_y, batch_seqlen, batch_mask):
    one = tf.constant(1, dtype=tf.int32)
    print(sess.run([tf.shape(pred), tf.shape(outputs), tf.shape(reshaped_outputs), tf.shape(index),
                    tf.shape(tf.equal(mask, one))], feed_dict={x: batch_x, y: batch_y,
                                   seqlen: batch_seqlen,
                                   mask:batch_mask}))

outputs, index = dynamicRNN(x, seqlen, weights, biases)
reshaped_outputs = tf.reshape(outputs, [-1, n_classes])
pred = tf.gather(reshaped_outputs, index)
# Define loss and optimizer
cost = dynamicRNNCost(pred, y)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

accuracy = dynamicRNNAccuracy(pred, y)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Run the initializer
sess.run(init)

# sess = tf_debug.LocalCLIDebugWrapperSession(sess)

for step in range(1, training_steps + 1):
    batch_x, batch_y, batch_seqlen, batch_mask = trainset.next(batch_size)

#     Run optimization op (backprop)
    sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                   seqlen: batch_seqlen,
                                   mask:batch_mask})

    if step % display_step == 0 or step == 1:
        # Calculate batch accuracy & loss
        acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y,
                                            seqlen: batch_seqlen, mask:batch_mask})
        print("Step " + str(step*batch_size) + ", Training Accuracy= " + \
              "{:.5f}".format(acc))

print("Optimization Finished!")

# Calculate accuracy
test_data = testset.data
test_label = testset.labels
test_seqlen = testset.seqlen
print("Testing Accuracy:", \
    sess.run(accuracy, feed_dict={x: test_data, y: test_label,
                                  seqlen: test_seqlen}))
