#!/usr/bin/env python3
'''create layer'''

import tensorflow.compat.v1 as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop
tf.disable_v2_behavior()


def train(X_train, Y_train, X_valid, Y_valid,
          layer_sizes, activations, alpha,
          iterations, save_path="/tmp/model.ckpt"):
    '''
    builds, trains, and saves a neural network classifier
    :X_train: is a numpy.ndarray of training data
    :Y_train: is a numpy.ndarray of training data labels
    :X_valid: is a numpy.ndarray of validation data
    :Y_valid: is a numpy.ndarray of validation data labels
    :layer_sizes: is a list containing the number of nodes
    in each layer of the network
    :activations: is a list containing the activation functions
    :alpha: is the learning rate
    :iterations: is the number of iterations to train for
    :save_path: is the path to save the model to
    '''
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    y_pred = forward_prop(x, layer_sizes, activations)
    tf.add_to_collection('y_pred', y_pred)
    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)
    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)
    train_op = create_train_op(loss, alpha)
    tf.add_to_collection('train_op', train_op)

    init = tf.global_variables_initializer()
    save_NN = tf.train.Saver()
    sess = tf.Session()
    sess.run(init)

    for step in range(iterations + 1):
        training_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
        training_acc = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
        valid_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
        valid_acc = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
        if step % 100 == 0 or step == iterations:
            print("After {} iterations:".format(step))
            print("\tTraining Cost: {}".format(training_cost))
            print("\tTraining Accuracy: {}".format(training_acc))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_acc))
        if step < iterations:
            sess.run(train_op, feed_dict={x: X_train, y: Y_train})
    return save_NN.save(sess, save_path)
