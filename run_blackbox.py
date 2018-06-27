"""
Whitebox adversarial training code for the publication

 Fortified Networks: Improving the Robustness of Deep Networks
 by Modeling the Manifold of Hidden Representations.

 Alex Lamb, Jonathan Binas, Anirudh Goyal,
 Dmitriy Serdyuk, Sandeep Subramanian, Ioannis Mitliagkas, Yoshua Bengio

 https://arxiv.org/pdf/1804.02485

"""
#Code partially adapted from Cleverhans tutorial implementation.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import numpy as np
import tensorflow as tf
from cleverhans.attacks_tf import jacobian_graph, jacobian_augmentation
from cleverhans.utils import set_log_level
from cleverhans.utils import to_categorical
from cleverhans.utils_mnist import data_mnist
from six.moves import xrange
from tensorflow.python.platform import flags

from fortnet_dae.attacks import FastGradientMethod, MadryEtAl
from fortnet_dae.models_tf import make_basic_model, MLP
from fortnet_dae.utils_tf import model_train, model_eval, batch_eval, compute_rec_err


FLAGS = flags.FLAGS
attacks = {'fgsm': FastGradientMethod, 'pgd': MadryEtAl}

attack_par = {
        'mnist': {
            'fgsm': {'eps': 0.3, 'ord': np.inf, 'clip_min': 0., 'clip_max': 1.},
            'pgd': {'eps': 0.3, 'eps_iter': 0.01, 'nb_iter': 40},
            'pgd_restart': {'eps': 0.3, 'eps_iter': 0.01, 'nb_iter': 40, 'nb_restarts': 1},
            },
        'cifar': {
            'fgsm': {'eps': 0.1 + 0.0*8.0/255, 'ord': np.inf, 'clip_min': 0., 'clip_max': 1.},
            'pgd': {'eps': 0.1 + 0.0*8.0/255, 'eps_iter': 0.01, 'nb_iter': 40},
            'pgd_restart': {'eps': 8.0/255, 'eps_iter': 0.01, 'nb_iter': 40, 'nb_restarts': 1},
            },
        'fashion_mnist': {
            'fgsm': {'eps': 0.1, 'ord': np.inf, 'clip_min': 0., 'clip_max': 1.},
            'pgd': {'eps': 0.3, 'eps_iter': 0.01, 'nb_iter': 40},
            'pgd_restart': {'eps': 0.3, 'eps_iter': 0.01, 'nb_iter': 40, 'nb_restarts': 1},
            },
        }


def setup_tutorial():
    tf.set_random_seed(1234)
    return True

def rec_err_fct(use_rec_err, blocking_option):
    def get_rec_err(pre_, post_):
        if not use_rec_err:
            return None
        return compute_rec_err(pre_, post_, blocking_option)
    return get_rec_err

def weight_rec_err(err, wgt):
    if err is None:
        return err
    return err * wgt

def prep_bbox(sess, x, y, X_train, Y_train, X_test, Y_test,
              nb_filters, nb_epochs, batch_size, learning_rate,
              rng, use_rec_err, model_arch, attack_name, use_cross_err,
              blocking_option, dataset_name, opt_type, merged):
    """
    Define and train a model that simulates the "remote"
    black-box oracle described in the original paper.
    :param sess: the TF session
    :param x: the input placeholder
    :param y: the ouput placeholder
    :param X_train: the training data for the oracle
    :param Y_train: the training labels for the oracle
    :param X_test: the testing data for the oracle
    :param Y_test: the testing labels for the oracle
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param rng: numpy.random.RandomState
    :return:
    """

    # Define input TF placeholder
    input_shape = {
            'mnist': (None, 28, 28, 1),
            'cifar': (None, 32, 32, 3),
            'fashion_mnist': (None,28,28,1),
            }[dataset_name]
    model_params = {
            'nb_filters': nb_filters, 'model_arch': model_arch,
            'blocking_option': blocking_option, 'input_shape': input_shape
            }

    get_rec_err = rec_err_fct(use_rec_err, blocking_option)

    # Define TF model graph (for the black-box model)
    model = make_basic_model(**model_params)
    predictions = model(x)
    pre_ae_states, post_ae_states = model.get_ae_states()
    rec_err = get_rec_err(pre_ae_states, post_ae_states)
    print("Defined TensorFlow model graph.")
    
    # generate adversarial training data
    attack = attacks[attack_name](model, sess=sess)
    x_adv = attack.generate(x, **attack_par[dataset_name][attack_name])
    predictions_adv = model(x_adv)
    pre_ae_states_adv, post_ae_states_adv = model.get_ae_states()
    rec_err_cross = get_rec_err(pre_ae_states, post_ae_states_adv)
    if rec_err_cross is None or rec_err is None:
        use_cross_err = False #XXX hack
    train_rec_err = rec_err_cross + rec_err if use_cross_err else rec_err

    # Train a model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }

    # adversarial training
    model_train(sess, x, y, predictions, X_train, Y_train,
                args=train_params, rng=rng, aux_loss=0.01*train_rec_err,
                predictions_adv=predictions_adv, opt_type=opt_type)

    # Print the accuracy on legitimate data
    eval_params = {'batch_size': batch_size}
    accuracy, rec_loss_eval = model_eval(
            sess, x, y, predictions, X_test, Y_test,
            args=eval_params, aux_loss_lst=[rec_err])
    print('Test accuracy of black-box on legitimate test examples: ' + str(accuracy))
    print('AE reconstruction error on legitimate test examples: ' + str(rec_loss_eval))

    return model, predictions, accuracy #XXX return adv preds?


def substitute_model(img_rows=28, img_cols=28, nb_classes=10, model_arch_sub=None, blocking_option=None):
    input_shape = (None, img_rows, img_cols, 1)
    model_params = {
            'model_arch': model_arch_sub, 'model_class': MLP,
            'blocking_option': blocking_option, 'input_shape': input_shape
            }
    # Define a model (it's different from the black-box)
    return make_basic_model(**model_params)


def train_sub(sess, x, y, bbox_preds, X_sub, Y_sub, nb_classes,
              nb_epochs_s, batch_size, learning_rate, data_aug, lmbda,
              rng, model_arch_sub, merged, opt_type, blocking_option):
    """
    This function creates the substitute by alternatively
    augmenting the training data and training the substitute.
    """
    # Define TF model graph (for the black-box model)
    model_sub = substitute_model(model_arch_sub=model_arch_sub,
            blocking_option=blocking_option)
    preds_sub = model_sub(x)
    #return model_sub, preds_sub

    print("Defined TensorFlow model graph for the substitute.")

    # Define the Jacobian symbolically using TensorFlow
    grads = jacobian_graph(preds_sub, x, nb_classes)

    # Train the substitute and augment dataset alternatively
    for rho in xrange(data_aug):
        print("Substitute training epoch #" + str(rho))
        train_params = {
            'nb_epochs': nb_epochs_s,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }
        model_train(sess, x, y, preds_sub, X_sub, to_categorical(Y_sub),
                    init_all=False, args=train_params,
                    rng=rng, opt_type=opt_type,
                    #summary=merged
                    )

        # If we are not at last substitute training iteration, augment dataset
        if rho < data_aug - 1:
            print("Augmenting substitute training data.")
            # Perform the Jacobian augmentation
            lmbda_coef = 2 * int(int(rho / 3) != 0) - 1
            X_sub = jacobian_augmentation(sess, x, X_sub, Y_sub, grads,
                                          lmbda_coef * lmbda)

            print("Labeling substitute training data.")
            # Label the newly generated synthetic points using the black-box
            Y_sub = np.hstack([Y_sub, Y_sub])
            X_sub_prev = X_sub[int(len(X_sub)/2):]
            eval_params = {'batch_size': batch_size}
            bbox_val = batch_eval(sess, [x], [bbox_preds], [X_sub_prev],
                                  args=eval_params)[0]
            # Note here that we take the argmax because the adversary
            # only has access to the label (not the probabilities) output
            # by the black-box model
            Y_sub[int(len(X_sub)/2):] = np.argmax(bbox_val, axis=1)

    return model_sub, preds_sub


def mnist_blackbox(train_start=0, train_end=60000, test_start=0,
                   test_end=10000, nb_classes=10, batch_size=128,
                   learning_rate=0.001, nb_epochs=10, holdout=150,
                   nb_filters=64, data_aug=6, nb_epochs_s=10,
                   lmbda=0.1, use_rec_err=True,
                   model_arch=None, model_arch_sub=None, attack_name=None,
                   use_cross_err=None, dataset_name=None,
                   blocking_option=None, opt_type='adam'):
    """
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :return: a dictionary with:
             * black-box model accuracy on test set
             * substitute model accuracy on test set
             * black-box model accuracy on adversarial examples transferred
               from the substitute model
    """
    get_rec_err = rec_err_fct(use_rec_err, blocking_option)

    merged = None #XXX switched this off for the moment

    # Set logging level to see debug information
    set_log_level(logging.DEBUG)

    # Dictionary used to keep track and return key accuracies
    accuracies = {}

    # Perform tutorial setup
    assert setup_tutorial()

    # Create TF session
    sess = tf.Session()

    # Get data
    X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
                                                  train_end=train_end,
                                                  test_start=test_start,
                                                  test_end=test_end)

    # Initialize substitute training set reserved for adversary
    X_sub = X_test[:holdout]
    Y_sub = np.argmax(Y_test[:holdout], axis=1)

    # Redefine test set as remaining samples unavailable to adversaries
    X_test = X_test[holdout:]
    Y_test = Y_test[holdout:]

    # Define input and output TF placeholders
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    # Seed random number generator so tutorial is reproducible
    rng = np.random.RandomState([2017, 8, 30])

    # Simulate the black-box model locally
    # You could replace this by a remote labeling API for instance
    print("Preparing the black-box model.")
    prep_bbox_out = prep_bbox(sess, x, y, X_train, Y_train, X_test, Y_test,
            nb_filters, nb_epochs, batch_size, learning_rate,
            rng=rng, use_rec_err=use_rec_err, model_arch=model_arch,
            attack_name=attack_name, use_cross_err=use_cross_err,
            dataset_name=dataset_name, blocking_option=blocking_option,
            opt_type=opt_type, merged=merged)
    #model, bbox_preds, accuracies['bbox'] = prep_bbox_out
    model, _, _ = prep_bbox_out
    bbox_preds = model(x)

    # Train substitute using method from https://arxiv.org/abs/1602.02697
    print("Training the substitute model.")
    model_sub, preds_sub = train_sub(sess, x, y, bbox_preds, X_sub, Y_sub,
                              nb_classes, nb_epochs_s, batch_size,
                              learning_rate, data_aug, lmbda, rng=rng,
                              model_arch_sub=model_arch_sub,
                              merged=merged, opt_type=opt_type,
                              blocking_option=blocking_option,
                              )

    # Evaluate the substitute model on clean test examples
    eval_params = {'batch_size': batch_size}
    acc_sub = model_eval(
            sess, x, y, preds_sub, X_test, Y_test, args=eval_params)
    accuracies['sub'] = acc_sub



    #XXX evaluating on clean samples after training sub
    preds = model(x)
    pre_ae_states, post_ae_states = model.get_ae_states()
    rec_err = get_rec_err(pre_ae_states, post_ae_states)
    accuracy, rec_loss_eval = model_eval(
            sess, x, y, preds, X_test, Y_test,
            args=eval_params, aux_loss_lst=[rec_err])
    print('Test accuracy of oracle on clean examples: ' + str(accuracy))
    print('reconstr. err. of oracle on clean examples: ' + str(rec_loss_eval))
    #XXX --> the result should be as before


    # Initialize the attack
    attack = attacks[attack_name](model_sub, sess=sess)

    x_adv_sub = attack.generate(x, **attack_par[dataset_name][attack_name])
    preds_adv = model(x_adv_sub)
    pre_ae_states_adv, post_ae_states_adv = model.get_ae_states()
    rec_err_adv = get_rec_err(pre_ae_states_adv, post_ae_states_adv)

    # Evaluate the accuracy of the "black-box" model on adversarial examples
    accuracy, rec_loss_eval = model_eval(
            sess, x, y, preds_adv, X_test, Y_test,
            args=eval_params, aux_loss_lst=[rec_err_adv])
    print('Test accuracy of oracle on adversarial examples: ' + str(accuracy))
    print('reconstr. err. of oracle on adversarial examples: ' + str(rec_loss_eval))
    accuracies['bbox_on_sub_adv_ex'] = accuracy

    print('Accuracies', accuracies)

    return accuracies


def main(argv=None):
    print('using configuration:')
    for k_, v_ in tf.flags.FLAGS.__flags.items():
        print(k_,'=', v_.value)

    mnist_blackbox(nb_classes=FLAGS.nb_classes, batch_size=FLAGS.batch_size,
                   learning_rate=FLAGS.learning_rate,
                   nb_epochs=FLAGS.nb_epochs, holdout=FLAGS.holdout,
                   data_aug=FLAGS.data_aug, nb_epochs_s=FLAGS.nb_epochs_s,
                   lmbda=FLAGS.lmbda, use_rec_err=FLAGS.rec_err,
                   model_arch=FLAGS.arch, model_arch_sub=FLAGS.arch_sub,
                   attack_name=FLAGS.attack, use_cross_err=FLAGS.cross_err,
                   dataset_name=FLAGS.dataset)


if __name__ == '__main__':
    # General flags
    flags.DEFINE_integer('nb_classes', 10, 'Number of classes in problem')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')

    # Flags related to oracle
    flags.DEFINE_integer('nb_epochs', 10, 'Number of epochs to train model')
    flags.DEFINE_bool('rec_err', True, 'Train DAE using aux loss')

    # Flags related to substitute
    flags.DEFINE_integer('holdout', 150, 'Test set holdout for adversary')
    flags.DEFINE_integer('data_aug', 6, 'Nb of substitute data augmentations')
    flags.DEFINE_integer('nb_epochs_s', 10, 'Training epochs for substitute')
    flags.DEFINE_float('lmbda', 0.1, 'Lambda from arxiv.org/abs/1602.02697')
    flags.DEFINE_string('arch', 'cnn', 'model architecture used for main model')
    flags.DEFINE_string('arch_sub', 'fcn_sub', 'model architecture used for substitute model')
    flags.DEFINE_string('attack', 'fgsm', 'attack carried out')
    flags.DEFINE_string('dataset', 'mnist', 'attack carried out')
    flags.DEFINE_bool('cross_err', True, 'Whether to use adv->clean or clean->clean reconstruction during adversarial training')
    flags.DEFINE_bool('blocking_option', True, 'do some blocking')

    tf.app.run()

