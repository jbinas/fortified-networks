"""
Whitebox adversarial training code for the publication

 Fortified Networks: Improving the Robustness of Deep Networks
 by Modeling the Manifold of Hidden Representations.

 Alex Lamb, Jonathan Binas, Anirudh Goyal,
 Dmitriy Serdyuk, Sandeep Subramanian, Ioannis Mitliagkas, Yoshua Bengio

 https://arxiv.org/pdf/1804.02485

(Code partially adapted from Cleverhans tutorial implementation.)
"""


import numpy as np

from cleverhans.model import Model, CallableModelWrapper
from cleverhans.attacks import Attack
from cleverhans.attacks import FastGradientMethod, MadryEtAl, CarliniWagnerL2


class MadryEtAl_WithRestarts(Attack):

    """
    The Projected Gradient Descent Attack (Madry et al. 2017).
    Paper link: https://arxiv.org/pdf/1706.06083.pdf
    """

    def __init__(self, model, back='tf', sess=None):
        """
        Create a MadryEtAl instance.
        """
        super(MadryEtAl_WithRestarts, self).__init__(model, back, sess)
        self.feedable_kwargs = {'eps': np.float32,
                                'eps_iter': np.float32,
                                'y': np.float32,
                                'y_target': np.float32,
                                'clip_min': np.float32,
                                'clip_max': np.float32,
                                'nb_restarts': np.float32}
        self.structural_kwargs = ['ord', 'nb_iter', 'rand_init']

        if not isinstance(self.model, Model):
            self.model = CallableModelWrapper(self.model, 'probs')

    def generate(self, x, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.
        :param x: The model's symbolic inputs.
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        :param rand_init: (optional bool) If True, an initial random
                    perturbation is added.
        """

        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        labels, nb_classes = self.get_or_guess_labels(x, kwargs)
        self.targeted = self.y_target is not None

        print("targeted?", self.targeted)

        # Initialize loop variables
        adv_x = self.attack(x, labels)

        return adv_x

    def parse_params(self, eps=0.3, eps_iter=0.01, nb_iter=40, y=None,
                     ord=np.inf, clip_min=None, clip_max=None,
                     y_target=None, rand_init=True, nb_restarts=1, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.

        Attack-specific parameters:
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        :param rand_init: (optional bool) If True, an initial random
                    perturbation is added.
        """

        # Save attack-specific parameters
        self.eps = eps
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.y = y
        self.y_target = y_target
        self.ord = ord
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.rand_init = rand_init
        self.nb_restarts = nb_restarts

        if self.y is not None and self.y_target is not None:
            raise ValueError("Must not set both y and y_target")
        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, 1, 2]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")

        return True

    def attack_single_step(self, x, eta, y):
        """
        Given the original image and the perturbation computed so far, computes
        a new perturbation.

        :param x: A tensor with the original input.
        :param eta: A tensor the same shape as x that holds the perturbation.
        :param y: A tensor with the target labels or ground-truth labels.
        """
        import tensorflow as tf
        from cleverhans.utils_tf import model_loss, clip_eta

        adv_x = x + eta
        preds = self.model.get_probs(adv_x)
        loss = model_loss(y, preds)
        loss_vector = model_loss(y, preds, mean=False)
        if self.targeted:
            loss = -loss
        grad, = tf.gradients(loss, adv_x)
        scaled_signed_grad = self.eps_iter * tf.sign(grad)
        adv_x = adv_x + scaled_signed_grad
        if self.clip_min is not None and self.clip_max is not None:
            adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)
        eta = adv_x - x
        eta = clip_eta(eta, self.ord, self.eps)
        return eta, loss, loss_vector

    def attack(self, x, y):
        """
        This method creates a symbolic graph that given an input image,
        first randomly perturbs the image. The
        perturbation is bounded to an epsilon ball. Then multiple steps of
        gradient descent is performed to increase the probability of a target
        label or decrease the probability of the ground-truth label.

        :param x: A tensor with the input image.
        """
        import tensorflow as tf
        from cleverhans.utils_tf import clip_eta

        best_loss = None
        best_eta = None

        print("Number of steps running", self.nb_restarts+1)

        for restart_step in range(0,self.nb_restarts+1):
            if self.rand_init:
                eta = tf.random_uniform(tf.shape(x), -self.eps, self.eps)
                eta = clip_eta(eta, self.ord, self.eps)
            else:
                eta = tf.zeros_like(x)
            #eta = tf.Print(eta, [eta[0:2,0:3],restart_step], "Clipped Eta drawn on this step")

            for i in range(self.nb_iter):
                eta,loss,loss_vec = self.attack_single_step(x, eta, y)

            if best_loss == None:
                #print("first time in loop")
                best_loss = loss_vec
                best_eta = eta
            else:
                #print("second time in loop")
                switch_cond = tf.less(best_loss,loss_vec)
                new_best_loss = tf.where(switch_cond, loss_vec*1.0, best_loss*1.0)
                new_best_eta = tf.where(switch_cond, eta*1.0, best_eta*1.0)
                #best_loss = tf.Print(best_loss, [best_loss[0:10], restart_step], "This is the best loss")
                #best_eta = tf.Print(best_eta, [best_loss[0:5],loss_vec[0:5],new_best_loss[0:5],best_eta[0:3,0,0,0],eta[0:3,0,0,0],new_best_eta[0:3,0,0,0],tf.shape(eta),restart_step], "Best_Loss, Loss_vec, New_Best_Loss, Best_eta,Eta_Curr, New_Best_Eta, Eta_Shape")
                best_loss = new_best_loss*1.0
                best_eta = new_best_eta*1.0

        adv_x = x + best_eta
        if self.clip_min is not None and self.clip_max is not None:
            adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

        return adv_x

