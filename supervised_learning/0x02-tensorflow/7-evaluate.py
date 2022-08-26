#!/usr/bin/env python3
'''create layer'''

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def evaluate(X, Y, save_path):
    """
    evaluate the output of the neural network
    :X: is a numpy.ndarray containing the input data
    :Y: is a numpy.ndarray containing the one-hot labels for X
    :save_path: is the path to the TensorFlow model
    """
    with tf.Session() as sess:
        save_NN = tf.train.import_meta_graph("{}.meta".format(save_path))
        save_NN.restore(sess, save_path)

        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]

        y_pred = tf.get_collection("y_pred")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]

        prediction = sess.run(y_pred, feed_dict={x: X, y: Y})
        accuracy = sess.run(accuracy, feed_dict={x: X, y: Y})
        loss = sess.run(loss, feed_dict={x: X, y: Y})
        return prediction, accuracy, loss
