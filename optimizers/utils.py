import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import random

from tensorflow_core.python.keras.callbacks import Callback
from tensorflow_core.python.util.tf_export import keras_export
from termcolor import colored
'''Helper methods for optimizers
'''


def warn_str():
    return colored('WARNING: ', 'red')


def get_weight_decays(model, verbose=1):
    wd_dict = {}
    for layer in model.layers:
        layer_l2regs = _get_layer_l2regs(layer)
        if layer_l2regs:
            for layer_l2 in layer_l2regs:
                weight_name, weight_l2 = layer_l2
                wd_dict.update({weight_name: weight_l2})
                if weight_l2 != 0 and verbose:
                    print((warn_str() + "{} l2-regularization = {} - should be "
                          "set 0 before compiling model").format(
                                  weight_name, weight_l2))
    return wd_dict


def fill_dict_in_order(_dict, _list_of_vals):
    for idx, key in enumerate(_dict.keys()):
        _dict[key] = _list_of_vals[idx]
    return _dict


def _get_layer_l2regs(layer):
    if hasattr(layer, 'cell') or \
      (hasattr(layer, 'layer') and hasattr(layer.layer, 'cell')):
        return _rnn_l2regs(layer)
    elif hasattr(layer, 'layer') and not hasattr(layer.layer, 'cell'):
        layer = layer.layer
    l2_lambda_kb = []
    for weight_name in ['kernel', 'bias']:
        _lambda = getattr(layer, weight_name + '_regularizer', None)
        if _lambda is not None:
            l2_lambda_kb.append([getattr(layer, weight_name).name,
                                 float(_lambda.l2)])
    return l2_lambda_kb


def _rnn_l2regs(layer):
    l2_lambda_krb = []
    if hasattr(layer, 'backward_layer'):
        for layer in [layer.forward_layer, layer.backward_layer]:
            l2_lambda_krb += _cell_l2regs(layer.cell)
        return l2_lambda_krb
    else:
        return _cell_l2regs(layer.cell)


def _cell_l2regs(rnn_cell):
    cell = rnn_cell
    l2_lambda_krb = []  # kernel-recurrent-bias

    for weight_idx, weight_type in enumerate(['kernel', 'recurrent', 'bias']):
        _lambda = getattr(cell, weight_type + '_regularizer', None)
        if _lambda is not None:
            weight_name = cell.weights[weight_idx].name
            l2_lambda_krb.append([weight_name, float(_lambda.l2)])
    return l2_lambda_krb


def _apply_weight_decays(cls, var, var_t):
    wd = cls.weight_decays[var.name]
    wd_normalized = wd * K.cast(
            K.sqrt(cls.batch_size / cls.total_iterations_wd), 'float32')
    var_t = var_t - cls.eta_t * wd_normalized * var

    if cls.init_verbose and not cls._init_notified:
        print('{} weight decay set for {}'.format(
                K_eval(wd_normalized), var.name))
    return var_t


def _compute_eta_t(cls):
    PI = 3.141592653589793
    t_frac = K.cast(K.cast(cls.iter_updates,'float32') / cls.total_iterations, 'float32')
    eta_t = cls.eta_min + 0.5 * (cls.eta_max - cls.eta_min) * \
        (1 + K.cos(PI * t_frac))
    return eta_t


def _apply_lr_multiplier(cls, lr_t, var):
    multiplier_name = [mult_name for mult_name in cls.lr_multipliers
                       if mult_name in var.name]
    if multiplier_name != []:
        lr_mult = cls.lr_multipliers[multiplier_name[0]]
    else:
        lr_mult = 1
    lr_t = lr_t * lr_mult

    if cls.init_verbose and not cls._init_notified:
        if lr_mult != 1:
            print('{} init learning rate set for {} -- {}'.format(
               '%.e' % K_eval(lr_t), var.name, lr_t))
        else:
            print('No change in learning rate {} -- {}'.format(
                                              var.name, K_eval(lr_t)))
    return lr_t


def _check_args(total_iterations, use_cosine_annealing, weight_decays):
    if use_cosine_annealing and total_iterations != 0:
        print('Using cosine annealing learning rates')
    elif (use_cosine_annealing or weight_decays != {}) and total_iterations == 0:
        print(warn_str() + "'total_iterations'==0, must be !=0 to use "
              + "cosine annealing and/or weight decays; "
              + "proceeding without either")


def reset_seeds(reset_graph_with_backend=None, verbose=1):
    if reset_graph_with_backend is not None:
        K = reset_graph_with_backend
        K.clear_session()
        tf.compat.v1.reset_default_graph()
        if verbose:
            print("KERAS AND TENSORFLOW GRAPHS RESET")

    np.random.seed(1)
    random.seed(2)
    if tf.__version__[0] == '2':
        tf.random.set_seed(3)
    else:
        tf.set_random_seed(3)
    if verbose:
        print("RANDOM SEEDS RESET")


def K_eval(x, backend=K):
    K = backend
    try:
        return K.get_value(K.to_dense(x))
    except Exception as e:
        try:
            eval_fn = K.function([], [x])
            return eval_fn([])[0]
        except Exception as e:
            return K.eager(K.eval)(x)


def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 20, 30 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3

    if epoch >= 30:
        lr *= 1e-2
    elif epoch >= 20:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def wd_schedule(epoch):
    """Weight Decay Schedule
    Weight decay is scheduled to be reduced after 20, 30 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        wd (float32): weight decay
    """
    wd = 1e-4

    if epoch >= 30:
        wd *= 1e-2
    elif epoch >= 20:
        wd *= 1e-1
    print('Weight decay: ', wd)
    return wd


# just copy the implement of LearningRateScheduler, and then change the lr with weight_decay
@keras_export('keras.callbacks.WeightDecayScheduler')
class WeightDecayScheduler(Callback):
    """Weight Decay Scheduler.

    Arguments:
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and returns a new
            weight decay as output (float).
        verbose: int. 0: quiet, 1: update messages.

    ```python
    # This function keeps the weight decay at 0.001 for the first ten epochs
    # and decreases it exponentially after that.
    def scheduler(epoch):
      if epoch < 10:
        return 0.001
      else:
        return 0.001 * tf.math.exp(0.1 * (10 - epoch))

    callback = WeightDecayScheduler(scheduler)
    model.fit(data, labels, epochs=100, callbacks=[callback],
              validation_data=(val_data, val_labels))
    ```
    """

    def __init__(self, schedule, verbose=0):
        super(WeightDecayScheduler, self).__init__()
        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'weight_decay'):
            raise ValueError('Optimizer must have a "weight_decay" attribute.')
        try:  # new API
            weight_decay = float(K.get_value(self.model.optimizer.weight_decay))
            weight_decay = self.schedule(epoch, weight_decay)
        except TypeError:  # Support for old API for backward compatibility
            weight_decay = self.schedule(epoch)
        if not isinstance(weight_decay, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        K.set_value(self.model.optimizer.weight_decay, weight_decay)
        if self.verbose > 0:
            print('\nEpoch %05d: WeightDecayScheduler reducing weight '
                  'decay to %s.' % (epoch + 1, weight_decay))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['weight_decay'] = K.get_value(self.model.optimizer.weight_decay)


class DecayHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.lr = []
        self.wd = []

    def on_batch_end(self, batch, logs=None):
        self.lr.append(self.model.optimizer.lr(self.model.optimizer.iterations))
        self.wd.append(self.model.optimizer.weight_decay)


class WarmRestart(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        K.set_value(self.model.optimizer.t_cur,0)