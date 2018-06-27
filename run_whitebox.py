"""
Whitebox adversarial training code for the publication

 Fortified Networks: Improving the Robustness of Deep Networks
 by Modeling the Manifold of Hidden Representations.

 Alex Lamb, Jonathan Binas, Anirudh Goyal,
 Dmitriy Serdyuk, Sandeep Subramanian, Ioannis Mitliagkas, Yoshua Bengio

 https://arxiv.org/pdf/1804.02485

"""
# Code partially adapted from Cleverhans tutorial implementation.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging, time

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags

from cleverhans.utils import AccuracyReport, set_log_level
from cleverhans.utils_mnist import data_mnist

from datasets.utils_cifar10 import data_cifar10, preprocess_image
from datasets.utils_fashion_mnist import fashion_mnist
from fortnet_dae.models_tf import make_basic_model
from fortnet_dae.utils_tf import model_train, model_eval, compute_rec_err
from fortnet_dae.attacks import FastGradientMethod, MadryEtAl, CarliniWagnerL2, MadryEtAl_WithRestarts


FLAGS = flags.FLAGS


def train(train_start=0, train_end=60000, test_start=0,
          test_end=10000, nb_epochs=6, batch_size=128,
          learning_rate=0.001,
          clean_train=True,
          testing=False,
          use_rec_err=True,
          backprop_through_attack=False,
          nb_filters=64, num_threads=None,
          model_arch=None,
          use_cross_error=True,
          attack_name=None,
          dataset_name='mnist',
          blocking_option=None,
          opt_type='adam',
          rec_error_weight=None):
    """
    Train model
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param clean_train: perform normal training on clean examples only
                        before performing adversarial training.
    :param testing: if true, complete an AccuracyReport for unit tests
                    to verify that performance is adequate
    :param backprop_through_attack: If True, backprop through adversarial
                                    example construction process during
                                    adversarial training.
    :param clean_train: if true, train on clean examples
    :return: an AccuracyReport object
    """

    attacks = {
            'fgsm': FastGradientMethod,
            'pgd': MadryEtAl,
            'pgd_restart': MadryEtAl_WithRestarts,
            'cw': CarliniWagnerL2
            }

    attack_params = {
            'mnist': {
                'fgsm': {'eps': 0.3, 'ord': np.inf, 'clip_min': 0., 'clip_max': 1.},
                'pgd': {'eps': 0.3, 'eps_iter': 0.01, 'nb_iter': 40},
                'pgd_restart': {'eps': 0.3, 'eps_iter': 0.01, 'nb_iter': 40, 'nb_restarts': 1},
                'cw' : {'batch_size': batch_size},
                },
            'cifar': {
                'fgsm': {'eps': 0.1 + 0.0*8.0/255, 'ord': np.inf, 'clip_min': 0., 'clip_max': 1.},
                'pgd': {'eps': 0.1 + 0.0*8.0/255, 'eps_iter': 0.01, 'nb_iter': 40},
                'pgd_restart': {'eps': 8.0/255, 'eps_iter': 0.01, 'nb_iter': 40, 'nb_restarts': 1},
                'cw' : {'batch_size': batch_size},
                },
            'fashion_mnist': {
                'fgsm': {'eps': 0.1, 'ord': np.inf, 'clip_min': 0., 'clip_max': 1.},
                'pgd': {'eps': 0.3, 'eps_iter': 0.01, 'nb_iter': 40},
                'pgd_restart': {'eps': 0.3, 'eps_iter': 0.01, 'nb_iter': 40, 'nb_restarts': 1},
                'cw' : {'batch_size': batch_size},
                },
            }[dataset_name]


    print("attack parameters:", attack_params[attack_name])

    def get_rec_err(pre_, post_):
        if not use_rec_err:
            return None
        return compute_rec_err(pre_, post_, blocking_option)

    def weight_rec_err(err, wgt):
        if err is None:
            return err
        return err * wgt

    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Set logging level to see debug information
    set_log_level(logging.DEBUG)

    # Create TF session
    if num_threads:
        config_args = dict(intra_op_parallelism_threads=1)
    else:
        config_args = {}
    sess = tf.Session(config=tf.ConfigProto(**config_args))

    # Get datasets
    datasets = {
            'mnist': data_mnist,
            'cifar': data_cifar10,
            'fashion_mnist': fashion_mnist,
            }
    X_train, Y_train, X_test, Y_test = datasets[dataset_name](
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end)

    # Use label smoothing
    assert Y_train.shape[1] == 10
    label_smooth = .1
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

    # Define input TF placeholder
    input_shape = {
            'mnist': (None, 28, 28, 1),
            'cifar': (None, 32, 32, 3),
            'fashion_mnist': (None,28,28,1)}[dataset_name]
    x = tf.placeholder(tf.float32, shape=input_shape)
    #tf.summary.image('input', x)
    y = tf.placeholder(tf.float32, shape=(None, 10))
    if dataset_name == 'cifar':
        x = tf.map_fn(lambda frame: preprocess_image(frame, True), x)

    # Train a model
    train_params = {
            'nb_epochs': nb_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate
            }
    model_params = {
            'nb_filters': nb_filters, 'model_arch': model_arch,
            'blocking_option': blocking_option, 'input_shape': input_shape
            }
    rng = np.random.RandomState([2017, 8, 30])

    merged = tf.summary.merge_all()

    if clean_train:
        print('.. using clean training')
        model = make_basic_model(**model_params)
        preds = model.get_probs(x)
        pre_ae_states, post_ae_states = model.get_ae_states()
        rec_err = get_rec_err(pre_ae_states, post_ae_states)

        class Evaluate(object):
            def __init__(self):
                self.best_accuracy = 0.

            def __call__(self):
                # Evaluate the accuracy of the model on legitimate test
                # examples
                eval_params = {'batch_size': batch_size}
                acc, rec_loss = model_eval(
                    sess, x, y, preds, X_test, Y_test,
                    args=eval_params, aux_loss_lst =[rec_err])
                self.best_accuracy = max(self.best_accuracy, acc)
                report.clean_train_clean_eval = acc
                assert X_test.shape[0] == test_end - test_start, X_test.shape
                print('Test accuracy on legitimate examples:   %0.4f' % acc)
                print('Best accuracy so far:                   {:0.4f}'.format(self.best_accuracy))
                print('reconstruction error on legit examples: %0.4f' % rec_loss)
        evaluate = Evaluate()

        model_train(sess, x, y, preds, X_train, Y_train,
                evaluate=evaluate, args=train_params, rng=rng,
                aux_loss=weight_rec_err(rec_err, rec_error_weight),
                opt_type=opt_type, summary=merged)

        # Calculate training error
        if testing:
            eval_params = {'batch_size': batch_size}
            acc, rec_loss = model_eval(
                sess, x, y, preds, X_train, Y_train,
                args=eval_params, aux_loss_lst=[rec_err])
            report.train_clean_train_clean_eval = acc

        # Initialize the Fast Gradient Sign Method (FGSM) attack object and
        # graph
        attack = attacks[attack_name](model, sess=sess)
        adv_x = attack.generate(x, **attack_params[attack_name])
        preds_adv = model.get_probs(adv_x)
        pre_ae_states_adv, post_ae_states_adv = model.get_ae_states()
        rec_err_adv = get_rec_err(pre_ae_states_adv, post_ae_states_adv)

        # Evaluate the accuracy of the model on adversarial examples
        eval_par = {'batch_size': batch_size}
        acc, rec_loss = model_eval(sess, x, y, preds_adv, X_test,
                Y_test, args=eval_par, aux_loss_lst=[rec_err_adv])
        print('Test accuracy on adversarial examples: %0.4f\n' % acc)
        print('reconstruction error on adv examples: %0.4f' % rec_loss)
        report.clean_train_adv_eval = acc

        # Calculate training error
        if testing:
            eval_par = {'batch_size': batch_size}
            acc, rec_loss = model_eval(sess, x, y, preds_adv, X_train,
                    Y_train, args=eval_par, aux_loss_lst=[rec_err_adv])
            report.train_clean_train_adv_eval = acc
    print('.. using adversarial training')
    # Redefine TF model graph
    model_2 = make_basic_model(**model_params)
    tf.summary.image('input', x)
    preds_2 = model_2(x)
    pre_ae_states_2, post_ae_states_2 = model_2.get_ae_states()
    rec_err_2 = get_rec_err(pre_ae_states_2, post_ae_states_2)
    attack2 = attacks[attack_name](model_2, sess=sess)
    adv_x_2 = attack2.generate(x, **attack_params[attack_name])
    tf.summary.image('adversarial', adv_x_2)
    tf.summary.image('diff', x - adv_x_2)
    if not backprop_through_attack:
        adv_x_2 = tf.stop_gradient(adv_x_2)
    preds_2_adv = model_2(adv_x_2)
    pre_ae_states_adv_2, post_ae_states_adv_2 = model_2.get_ae_states()
    rec_err_adv_2 = get_rec_err(pre_ae_states_adv_2, post_ae_states_adv_2)

    # adv -> clean reconstruction
    rec_err_cross_2 = get_rec_err(pre_ae_states_2, post_ae_states_adv_2)

    class Evaluate2(object):
        def __init__(self):
            self.best_accuracy = 0.

        def __call__(self):
            # Accuracy of adversarially trained model on legitimate test inputs
            eval_params = {'batch_size': batch_size}
            accuracy, rec_loss = model_eval(
                sess, x, y, preds_2, X_test, Y_test,
                args=eval_params, aux_loss_lst=[rec_err_2], summary=merged)
            self.best_accuracy = max(self.best_accuracy, accuracy)
            print('Test accuracy on legitimate examples:   %0.4f' % accuracy)
            print('Best test accuracy so far:              {:0.4f}'.format(self.best_accuracy))
            print('reconstruction error on legit examples: %0.4f' % rec_loss)
            report.adv_train_clean_eval = accuracy

            # Accuracy of the adversarially trained model on adversarial examples
            accuracy_adv, rec_loss_adv, rec_loss_cross = model_eval(
                sess, x, y, preds_2_adv, X_test, Y_test,
                args=eval_params, aux_loss_lst=[rec_err_adv_2, rec_err_cross_2],
                summary=merged)
            print('Test accuracy on adversarial examples:  %0.4f' % accuracy_adv)
            print('reconstruction error on adv->adv:       %0.4f' % rec_loss_adv)
            print('reconstruction error on adv->clean:     %0.4f' % rec_loss_cross)
            report.adv_train_adv_eval = accuracy_adv

            with open('report.dat', 'a') as f:
                f.write(' '.join(['%0.4f' % v for v in [
                    accuracy,
                    accuracy_adv,
                    rec_loss,
                    rec_loss_adv,
                    rec_loss_cross,
                    ]]) + '\n')
    evaluate_2 = Evaluate2()

    # combine both errors
    if rec_err_cross_2 is None or rec_err_2 is None:
        use_cross_error = False #XXX hack
    rec_use_train = rec_err_cross_2 + rec_err_2 if use_cross_error else rec_err_2

    # Perform and evaluate adversarial training
    model_train(sess, x, y, preds_2, X_train, Y_train,
                predictions_adv=preds_2_adv, evaluate=evaluate_2,
                args=train_params, rng=rng,
                aux_loss=weight_rec_err(rec_use_train, rec_error_weight),
                opt_type=opt_type, summary=merged)


    # Calculate training errors
    if testing:
        eval_params = {'batch_size': batch_size}
        accuracy, rec_loss = model_eval(sess, x, y, preds_2, X_train, Y_train,
                                        args=eval_params, aux_loss_lst=[rec_err_2])
        report.train_adv_train_clean_eval = accuracy
        accuracy, rec_loss = model_eval(sess, x, y, preds_2_adv, X_train,
                                        Y_train, args=eval_params,
                                        aux_loss_lst=[rec_err_adv_2])
        report.train_adv_train_adv_eval = accuracy

    return report


def main(argv=None):
    with open('report.dat', 'w') as f:
        f.write(time.strftime("%% whitebox - %Y-%m-%d %H:%M:%S\n"))
    print('using configuration:')
    with open('report.dat', 'a') as f:
        for k_, v_ in tf.flags.FLAGS.__flags.items():
            f.write('%% %s: %s\n' % (k_, v_.value))
            print(k_, '=', v_.value)

        f.write('acc acc_adv rec rec_adv rec_cross\n')

    train(nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size,
          learning_rate=FLAGS.learning_rate,
          clean_train=FLAGS.clean_train,
          backprop_through_attack=FLAGS.backprop_through_attack,
          nb_filters=FLAGS.nb_filters, use_rec_err=FLAGS.rec_err,
          model_arch=FLAGS.arch, use_cross_error=FLAGS.cross_err,
          attack_name=FLAGS.attack, dataset_name=FLAGS.dataset,
          blocking_option=FLAGS.blocking_option,
          opt_type=FLAGS.opt_type,
          rec_error_weight=FLAGS.rec_error_weight)

if __name__ == '__main__':
    flags.DEFINE_integer('nb_filters', 64, 'Model size multiplier')
    flags.DEFINE_integer('nb_epochs', 10, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
    flags.DEFINE_bool('clean_train', False, 'Train on clean examples')
    flags.DEFINE_bool('rec_err', False, 'Train DAE using aux loss')
    flags.DEFINE_bool('backprop_through_attack', False,
            'If True, backprop through adversarial example construction process during adversarial training')
    flags.DEFINE_bool('cross_err', False, 'Whether to use adv->clean or clean->clean reconstruction during adversarial training')
    flags.DEFINE_string('arch', 'cnn', 'model architecture used for main model')
    flags.DEFINE_string('attack', 'fgsm', 'attack carried out')
    flags.DEFINE_bool('blocking_option', False, 'Whether to block reconstruction loss gradient from effecting classifier params')
    flags.DEFINE_string('dataset', 'mnist', 'Dataset name')
    flags.DEFINE_string('opt_type', 'adam', 'The type of optimizer to use')
    flags.DEFINE_float('rec_error_weight', 1, 'Reweight all reconstruction errors by this scalar during training')

    tf.app.run()


