

# Fortified Networks

Experiment code corresponding to the publication

**Fortified Networks: Improving the Robustness of Deep Networks by Modeling the Manifold of Hidden Representations**

Alex Lamb, Jonathan Binas, Anirudh Goyal, Dmitriy Serdyuk, Sandeep Subramanian, Ioannis Mitliagkas, Yoshua Bengio

<https://arxiv.org/pdf/1804.02485>

Abstract: Deep networks have achieved impressive results across a variety of important tasks. However a known weakness is a failure to perform well when evaluated on data which differ from the training distribution, even if these differences are very small, as is the case with adversarial examples. We propose Fortified Networks, a simple transformation of existing networks, which fortifies the hidden layers in a deep network by identifying when the hidden states are off of the data manifold, and maps these hidden states back to parts of the data manifold where the network performs well. Our principal contribution is to show that fortifying these hidden states improves the robustness of deep networks and our experiments (i) demonstrate improved robustness to standard adversarial attacks in both black-box and white-box threat models; (ii) suggest that our improvements are not primarily due to the gradient masking problem and (iii) show the advantage of doing this fortification in the hidden layers instead of the input space.


# Running the experiments

## Whitebox attack

To perform a whitebox attack on a CNN trained on MNIST without fortification (no reconstruction loss on the DAE,) run
```
$ python mnist_whitebox.py
```

After 10 epochs, the output should be something like
```
Test accuracy on legitimate examples:   0.9909
Best test accuracy so far:              0.9922
reconstruction error on legit examples: 0.0000
Test accuracy on adversarial examples:  0.9730
reconstruction error on adv->adv:       0.0000
reconstruction error on adv->clean:     0.0000
```

Now, train the same model with reconstruction loss on the DAE:
```
$ python mnist_whitebox.py --rec_err
```

After around 10 epochs, this should yield something like
```
Test accuracy on legitimate examples:   0.9904
Best test accuracy so far:              0.9908
reconstruction error on legit examples: 0.0043
Test accuracy on adversarial examples:  0.9894
reconstruction error on adv->adv:       0.5213
reconstruction error on adv->clean:     2.2694
```

The network architecture used is selected with the `--arch` flag. Provided models are `fcn`, `cnn`, `deep_cnn`, `resnet`. Models are specified in `models_tf.py` and can easily be changed there.


Possible arguments are
```
run_whitebox.py:
  --arch: model architecture used for main model
    (default: 'cnn')
  --attack: attack to run (fgsm, pgd, pgd_restart, or cw.)
    (default: 'fgsm')
  --[no]backprop_through_attack: If True, backprop through adversarial example
    construction process during adversarial training
    (default: 'false')
  --batch_size: Size of training batches
    (default: '128')
    (an integer)
  --[no]blocking_option: Whether to block reconstruction loss gradient from
    affecting classifier params
    (default: 'false')
  --[no]clean_train: Train on clean examples
    (default: 'false')
  --[no]cross_err: Whether to use adv->clean or clean->clean reconstruction
    during adversarial training
    (default: 'false')
  --dataset: Dataset name (mnist, cifar, fashion_mnist.)
    (default: 'mnist')
  --learning_rate: Learning rate for training
    (default: '0.001')
    (a number)
  --nb_epochs: Number of epochs to train model
    (default: '10')
    (an integer)
  --nb_filters: Model size multiplier
    (default: '64')
    (an integer)
  --opt_type: The type of optimizer to use
    (default: 'adam')
  --[no]rec_err: Train DAE using auxiliary loss
    (default: 'false')
  --rec_error_weight: Reweight all reconstruction errors by this scalar during
    training
    (default: '1.0')
    (a number)
```


## Blackbox attack

To run and defend a blackbox attack on a CNN trained on MNIST, run
```
$ python run_blackbox.py --arch=cnn_orig --arch_sub=fcn_sub --noblocking_option --nocross_err --nb_epochs 30
```

The result should roughly be
```
Test accuracy of oracle on clean examples:        0.9902538071065989
reconstr. err. of oracle on clean examples:       0.6672933348060259
Test accuracy of oracle on adversarial examples:  0.9721827411167513
reconstr. err. of oracle on adversarial examples: 2.1407209986962643
```

The network architecture used is selected with the `--arch` flag. Models are specified in `models_tf.py` and can easily be changed there. Provided substitute model architectures, specified using the `--arch_sub` flag are  `fcn_sub`, `cnn_sub`, `cnn_sub_small` (see `models_tf.py` for details.)


Possible arguments are
```
run_blackbox.py:
  --arch: model architecture used for main model
    (default: 'cnn')
  --arch_sub: model architecture used for substitute model
    (default: 'fcn_sub')
  --attack: attack carried out
    (default: 'fgsm')
  --batch_size: Size of training batches
    (default: '128')
    (an integer)
  --[no]blocking_option: do some blocking
    (default: 'true')
  --[no]cross_err: Whether to use adv->clean or clean->clean reconstruction
    during adversarial training
    (default: 'true')
  --data_aug: Nb of substitute data augmentations
    (default: '6')
    (an integer)
  --dataset: attack carried out
    (default: 'mnist')
  --holdout: Test set holdout for adversary
    (default: '150')
    (an integer)
  --learning_rate: Learning rate for training
    (default: '0.001')
    (a number)
  --lmbda: Lambda from arxiv.org/abs/1602.02697
    (default: '0.1')
    (a number)
  --nb_classes: Number of classes in problem
    (default: '10')
    (an integer)
  --nb_epochs: Number of epochs to train model
    (default: '10')
    (an integer)
  --nb_epochs_s: Training epochs for substitute
    (default: '10')
    (an integer)
  --[no]rec_err: Train DAE using auxiliary loss
    (default: 'true')
```

