"""
Whitebox adversarial training code for the publication

 Fortified Networks: Improving the Robustness of Deep Networks
 by Modeling the Manifold of Hidden Representations.

 Alex Lamb, Jonathan Binas, Anirudh Goyal,
 Dmitriy Serdyuk, Sandeep Subramanian, Ioannis Mitliagkas, Yoshua Bengio

 https://arxiv.org/pdf/1804.02485

(Code partially adapted from Cleverhans tutorial implementation.)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import numpy as np
import os
import tensorflow as tf
import time
import warnings
import logging
from distutils.version import LooseVersion
from six.moves import range

from cleverhans.utils import batch_indices, _ArgsWrapper, create_logger
from cleverhans.utils import set_log_level, get_log_level

_logger = create_logger("cleverhans.utils.tf")


def _get_width(x):
    width = 1
    for factor in x.shape[1:]:
        width *= factor.value
    return width


def compute_rec_err(xpre, xpost, blocking_option=False):
    if isinstance(xpre, list) and isinstance(xpost, list):
        if len(xpre) == 0 or len(xpost) == 0:
            return None
        return sum([compute_rec_err(pre_, post_, blocking_option) for (pre_, post_) in zip(xpre, xpost)])
    # flatten layers of rank > 1
    width_pre, width_post = _get_width(xpre), _get_width(xpost)
    assert width_pre == width_post
    xpre = tf.reshape(xpre, (-1, width_pre))
    xpost = tf.reshape(xpost, (-1, width_post))
    if blocking_option:
        return tf.reduce_mean(tf.square(tf.stop_gradient(xpre) - xpost), axis=1)
    else:
        return tf.reduce_mean(tf.square(xpre - xpost), axis=1)


def model_loss(y, model, mean=True, aux_loss=None):
    """
    Define loss of TF graph
    :param y: correct labels
    :param model: output of the model
    :param mean: boolean indicating whether should return mean of loss
                 or vector of losses for each input of the batch
    :return: return mean of loss if True, otherwise return vector with per
             sample loss
    """

    op = model.op
    if "softmax" in str(op).lower():
        logits, = op.inputs
    else:
        logits = model

    out = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y)
    if aux_loss is not None:
        out += aux_loss
    if mean:
        out = tf.reduce_mean(out)
    return out


def initialize_uninitialized_global_variables(sess):
    """
    Only initializes the variables of a TensorFlow session that were not
    already initialized.
    :param sess: the TensorFlow session
    :return:
    """
    # List all global variables
    global_vars = tf.global_variables()

    # Find initialized status for all variables
    is_var_init = [tf.is_variable_initialized(var) for var in global_vars]
    is_initialized = sess.run(is_var_init)

    # List all variables that were not initialized previously
    not_initialized_vars = [var for (var, init) in
                            zip(global_vars, is_initialized) if not init]

    # Initialize all uninitialized variables found, if any
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))


def model_train(sess, x, y, predictions, X_train, Y_train, save=False,
                predictions_adv=None, init_all=True, evaluate=None,
                verbose=True, feed=None, args=None, rng=None, aux_loss=None,
                opt_type=None, summary=None):
    """
    Train a TF graph
    :param sess: TF session to use when training the graph
    :param x: input placeholder
    :param y: output placeholder (for labels)
    :param predictions: model output predictions
    :param X_train: numpy array with training inputs
    :param Y_train: numpy array with training outputs
    :param save: boolean controlling the save operation
    :param predictions_adv: if set with the adversarial example tensor,
                            will run adversarial training
    :param init_all: (boolean) If set to true, all TF variables in the session
                     are (re)initialized, otherwise only previously
                     uninitialized variables are initialized before training.
    :param evaluate: function that is run after each training iteration
                     (typically to display the test/validation accuracy).
    :param verbose: (boolean) all print statements disabled when set to False.
    :param feed: An optional dictionary that is appended to the feeding
                 dictionary before the session runs. Can be used to feed
                 the learning phase of a Keras model for instance.
    :param args: dict or argparse `Namespace` object.
                 Should contain `nb_epochs`, `learning_rate`,
                 `batch_size`
                 If save is True, should also contain 'train_dir'
                 and 'filename'
    :param rng: Instance of numpy.random.RandomState
    :return: True if model trained
    """
    args = _ArgsWrapper(args or {})

    train_writer = tf.summary.FileWriter('./logs/train', sess.graph)

    # Check that necessary arguments were given (see doc above)
    assert args.nb_epochs, "Number of epochs was not given in args dict"
    assert args.learning_rate, "Learning rate was not given in args dict"
    assert args.batch_size, "Batch size was not given in args dict"

    if save:
        assert args.train_dir, "Directory for save was not given in args dict"
        assert args.filename, "Filename for save was not given in args dict"

    if not verbose:
        old_log_level = get_log_level(name=_logger.name)
        set_log_level(logging.WARNING, name=_logger.name)
        warnings.warn("verbose argument is deprecated and will be removed"
                      " on 2018-02-11. Instead, use utils.set_log_level()."
                      " For backward compatibility, log_level was set to"
                      " logging.WARNING (30).")

    if rng is None:
        rng = np.random.RandomState()

    # Define loss
    loss = model_loss(y, predictions, aux_loss=aux_loss)
    if predictions_adv is not None:
        loss = (loss + model_loss(y, predictions_adv, aux_loss=aux_loss)) / 2

    #XXX this is new
    if opt_type == "momentum":
        initial_learning_rate = 0.1 * args.batch_size / 128 * 0.01
        batches_per_epoch = X_train.shape[0] / args.batch_size
        global_step = tf.train.get_or_create_global_step()
        _MOMENTUM = 0.9

        # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
        boundaries = [int(batches_per_epoch * epoch) for epoch in [100, 150, 200]]
        values = [initial_learning_rate * decay for decay in [1, 0.1, 0.01, 0.001]]
        learning_rate = tf.train.piecewise_constant(
            tf.cast(global_step, tf.int32), boundaries, values)

        # Create a tensor named learning_rate for logging purposes
        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)

        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=_MOMENTUM)
    elif opt_type == "adam":
        optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        global_step = tf.train.get_or_create_global_step()
    else:
        raise ValueError

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = optimizer.minimize(loss, global_step)
    #XXX original version:
    #train_step = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
    #train_step = train_step.minimize(loss)

    with sess.as_default():
        if init_all:
            tf.global_variables_initializer().run()
        else:
            initialize_uninitialized_global_variables(sess)

        for epoch in range(args.nb_epochs):
            # Compute number of batches
            nb_batches = int(math.ceil(float(len(X_train)) / args.batch_size))
            assert nb_batches * args.batch_size >= len(X_train)

            # Indices to shuffle training set
            index_shuf = list(range(len(X_train)))
            rng.shuffle(index_shuf)

            prev = time.time()
            for batch in range(nb_batches):

                # Compute batch start and end indices
                start, end = batch_indices(
                    batch, len(X_train), args.batch_size)

                # Perform one training step
                feed_dict = {x: X_train[index_shuf[start:end]],
                             y: Y_train[index_shuf[start:end]]}
                if feed is not None:
                    feed_dict.update(feed)
                if summary is None:
                    train_step.run(feed_dict=feed_dict)
                else:
                    summary_val, _ = sess.run([summary, train_step], feed_dict=feed_dict)
                    train_writer.add_summary(summary_val, batch + epoch * nb_batches)
            assert end >= len(X_train)  # Check that all examples were used
            cur = time.time()
            if verbose:
                _logger.info("Epoch " + str(epoch) + " took " +
                             str(cur - prev) + " seconds")
            if evaluate is not None:
                evaluate()

        if save:
            save_path = os.path.join(args.train_dir, args.filename)
            saver = tf.train.Saver()
            saver.save(sess, save_path)
            _logger.info("Completed model training and saved at: " +
                         str(save_path))
        else:
            _logger.info("Completed model training.")

    if not verbose:
        set_log_level(old_log_level, name=_logger.name)

    return True


def model_eval(sess, x, y, predictions, X_test=None, Y_test=None,
               feed=None, args=None, aux_loss_lst=[None], summary=None):
    """
    Compute the accuracy of a TF model on some data
    :param sess: TF session to use when training the graph
    :param x: input placeholder
    :param y: output placeholder (for labels)
    :param predictions: model output predictions
    :param X_test: numpy array with training inputs
    :param Y_test: numpy array with training outputs
    :param feed: An optional dictionary that is appended to the feeding
             dictionary before the session runs. Can be used to feed
             the learning phase of a Keras model for instance.
    :param args: dict or argparse `Namespace` object.
                 Should contain `batch_size`
    :return: a float with the accuracy value
    """
    args = _ArgsWrapper(args or {})

    test_writer = tf.summary.FileWriter('./logs/test')

    assert args.batch_size, "Batch size was not given in args dict"
    if X_test is None or Y_test is None:
        raise ValueError("X_test argument and Y_test argument "
                         "must be supplied.")

    # Define accuracy symbolically
    if LooseVersion(tf.__version__) >= LooseVersion('1.0.0'):
        correct_preds = tf.equal(tf.argmax(y, axis=-1),
                                 tf.argmax(predictions, axis=-1))
    else:
        correct_preds = tf.equal(tf.argmax(y, axis=tf.rank(y) - 1),
                                 tf.argmax(predictions,
                                     axis=tf.rank(predictions) - 1))

    # Init result var
    accuracy = 0.0
    total_aux_loss = [0.] * len(aux_loss_lst)

    with sess.as_default():
        # Compute number of batches
        nb_batches = int(math.ceil(float(len(X_test)) / args.batch_size))
        assert nb_batches * args.batch_size >= len(X_test)

        X_cur = np.zeros((args.batch_size,) + X_test.shape[1:],
                         dtype=X_test.dtype)
        Y_cur = np.zeros((args.batch_size,) + Y_test.shape[1:],
                         dtype=Y_test.dtype)
        for batch in range(nb_batches):
            if batch % 100 == 0 and batch > 0:
                _logger.debug("Batch " + str(batch))

            # Must not use the `batch_indices` function here, because it
            # repeats some examples.
            # It's acceptable to repeat during training, but not eval.
            start = batch * args.batch_size
            end = min(len(X_test), start + args.batch_size)

            # The last batch may be smaller than all others. This should not
            # affect the accuarcy disproportionately.
            cur_batch_size = end - start
            X_cur[:cur_batch_size] = X_test[start:end]
            Y_cur[:cur_batch_size] = Y_test[start:end]
            feed_dict = {x: X_cur, y: Y_cur}
            if feed is not None:
                feed_dict.update(feed)
            if summary is None:
                cur_corr_preds = correct_preds.eval(feed_dict=feed_dict)
            else:
                cur_corr_preds, summary_val = sess.run([correct_preds, summary], feed_dict=feed_dict)
                test_writer.add_summary(summary_val)

            accuracy += cur_corr_preds[:cur_batch_size].sum()

            for i, aux_loss in enumerate(aux_loss_lst):
                if aux_loss is None:
                    continue
                cur_aux_loss = aux_loss.eval(feed_dict=feed_dict)
                total_aux_loss[i] += cur_aux_loss[:cur_batch_size].sum()

        assert end >= len(X_test)

        # Divide by number of examples to get final value
        accuracy /= len(X_test)
        for i in range(len(aux_loss_lst)):
            total_aux_loss[i] /= len(X_test)

    return [accuracy] + total_aux_loss


def tf_model_load(sess, file_path=None):
    """
    :param sess: the session object to restore
    :param file_path: path to the restored session, if None is
                      taken from FLAGS.train_dir and FLAGS.filename
    :return:
    """
    FLAGS = tf.app.flags.FLAGS
    with sess.as_default():
        saver = tf.train.Saver()
        if file_path is None:
            warnings.warn("Please provide file_path argument, "
                          "support for FLAGS.train_dir and FLAGS.filename "
                          "will be removed on 2018-04-23.")
            file_path = os.path.join(FLAGS.train_dir, FLAGS.filename)
        saver.restore(sess, file_path)

    return True


def batch_eval(sess, tf_inputs, tf_outputs, numpy_inputs, feed=None,
               args=None):
    """
    A helper function that computes a tensor on numpy inputs by batches.

    :param sess:
    :param tf_inputs:
    :param tf_outputs:
    :param numpy_inputs:
    :param feed: An optional dictionary that is appended to the feeding
             dictionary before the session runs. Can be used to feed
             the learning phase of a Keras model for instance.
    :param args: dict or argparse `Namespace` object.
                 Should contain `batch_size`
    """
    args = _ArgsWrapper(args or {})

    assert args.batch_size, "Batch size was not given in args dict"

    n = len(numpy_inputs)
    assert n > 0
    assert n == len(tf_inputs)
    m = numpy_inputs[0].shape[0]
    for i in range(1, n):
        assert numpy_inputs[i].shape[0] == m
    out = []
    for _ in tf_outputs:
        out.append([])
    with sess.as_default():
        for start in range(0, m, args.batch_size):
            batch = start // args.batch_size
            if batch % 100 == 0 and batch > 0:
                _logger.debug("Batch " + str(batch))

            # Compute batch start and end indices
            start = batch * args.batch_size
            end = start + args.batch_size
            numpy_input_batches = [numpy_input[start:end]
                                   for numpy_input in numpy_inputs]
            cur_batch_size = numpy_input_batches[0].shape[0]
            assert cur_batch_size <= args.batch_size
            for e in numpy_input_batches:
                assert e.shape[0] == cur_batch_size

            feed_dict = dict(zip(tf_inputs, numpy_input_batches))
            if feed is not None:
                feed_dict.update(feed)
            numpy_output_batches = sess.run(tf_outputs, feed_dict=feed_dict)
            for e in numpy_output_batches:
                assert e.shape[0] == cur_batch_size, e.shape
            for out_elem, numpy_output_batch in zip(out, numpy_output_batches):
                out_elem.append(numpy_output_batch)

    out = [np.concatenate(x, axis=0) for x in out]
    for e in out:
        assert e.shape[0] == m, e.shape
    return out


def model_argmax(sess, x, predictions, samples, feed=None):
    """
    Helper function that computes the current class prediction
    :param sess: TF session
    :param x: the input placeholder
    :param predictions: the model's symbolic output
    :param samples: numpy array with input samples (dims must match x)
    :param feed: An optional dictionary that is appended to the feeding
             dictionary before the session runs. Can be used to feed
             the learning phase of a Keras model for instance.
    :return: the argmax output of predictions, i.e. the current predicted class
    """
    feed_dict = {x: samples}
    if feed is not None:
        feed_dict.update(feed)
    probabilities = sess.run(predictions, feed_dict)

    if samples.shape[0] == 1:
        return np.argmax(probabilities)
    else:
        return np.argmax(probabilities, axis=1)


def l2_batch_normalize(x, epsilon=1e-12, scope=None):
    """
    Helper function to normalize a batch of vectors.
    :param x: the input placeholder
    :param epsilon: stabilizes division
    :return: the batch of l2 normalized vector
    """
    with tf.name_scope(scope, "l2_batch_normalize") as scope:
        x_shape = tf.shape(x)
        x = tf.contrib.layers.flatten(x)
        x /= (epsilon + tf.reduce_max(tf.abs(x), 1, keepdims=True))
        square_sum = tf.reduce_sum(tf.square(x), 1, keepdims=True)
        x_inv_norm = tf.rsqrt(np.sqrt(epsilon) + square_sum)
        x_norm = tf.multiply(x, x_inv_norm)
        return tf.reshape(x_norm, x_shape, scope)


def kl_with_logits(p_logits, q_logits, scope=None,
                   loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES):
    """Helper function to compute kl-divergence KL(p || q)
    """
    with tf.name_scope(scope, "kl_divergence") as name:
        p = tf.nn.softmax(p_logits)
        p_log = tf.nn.log_softmax(p_logits)
        q_log = tf.nn.log_softmax(q_logits)
        loss = tf.reduce_mean(tf.reduce_sum(p * (p_log - q_log), axis=1),
                              name=name)
        tf.losses.add_loss(loss, loss_collection)
        return loss


def clip_eta(eta, ord, eps):
    """
    Helper function to clip the perturbation to epsilon norm ball.
    :param eta: A tensor with the current perturbation.
    :param ord: Order of the norm (mimics Numpy).
                Possible values: np.inf, 1 or 2.
    :param eps: Epilson, bound of the perturbation.
    """

    # Clipping perturbation eta to self.ord norm ball
    if ord not in [np.inf, 1, 2]:
        raise ValueError('ord must be np.inf, 1, or 2.')
    reduc_ind = list(range(1, len(eta.get_shape())))
    avoid_zero_div = 1e-12
    if ord == np.inf:
        eta = tf.clip_by_value(eta, -eps, eps)
    else:
        if ord == 1:
            norm = tf.maximum(avoid_zero_div,
                              tf.reduce_sum(tf.abs(eta),
                                            reduc_ind, keepdims=True))
        elif ord == 2:
            # avoid_zero_div must go inside sqrt to avoid a divide by zero
            # in the gradient through this operation
            norm = tf.sqrt(tf.maximum(avoid_zero_div,
                                      tf.reduce_sum(tf.square(eta),
                                                    reduc_ind,
                                                    keepdims=True)))
        # We must *clip* to within the norm ball, not *normalize* onto the
        # surface of the ball
        factor = tf.minimum(1., eps / norm)
        eta = eta * factor
    return eta
