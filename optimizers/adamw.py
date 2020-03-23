from tensorflow.python.framework import ops
from tensorflow.python.keras import backend_config
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
from tensorflow.python.keras.optimizer_v2 import learning_rate_schedule
from tensorflow.python.ops import array_ops, control_flow_ops, math_ops, state_ops
from tensorflow.python.util.tf_export import keras_export
import tensorflow.keras.backend as K
from .utils import _apply_weight_decays, _compute_eta_t
from .utils import _apply_lr_multiplier, _check_args, K_eval
import tensorflow as tf
from tensorflow.python.keras import backend


class AdamW(OptimizerV2):
    """AdamW optimizer.
    Default parameters follow those provided in the original paper.
    For extended documentation, see optimizer_v2.Adam.__doc__.
    # Arguments
        learning_rate: A Tensor or a floating point value.  The learning rate.
        beta_1: A float value or a constant float tensor. The exponential decay
            rate for the 1st moment estimates.
        beta_2: A float value or a constant float tensor. The exponential decay
            rate for the 2nd moment estimates.
        epsilon: A small constant for numerical stability. This epsilon is
            "epsilon hat" in the Kingma and Ba paper (in the formula just before
            Section 2.1), not the epsilon in Algorithm 1 of the paper.
        amsgrad: boolean. Whether to apply AMSGrad variant of this algorithm from
            the paper "On the Convergence of Adam and beyond".
        name: Optional name for the operations created when applying gradients.
            Defaults to "Adam".  @compatibility(eager) When eager execution is
            enabled, `learning_rate`, `beta_1`, `beta_2`, and `epsilon` can each be
            a callable that takes no arguments and returns the actual value to use.
            This can be useful for changing these values across different
            invocations of optimizer functions. @end_compatibility
        **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
            `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip
            gradients by value, `decay` is included for backward compatibility to
            allow time inverse decay of learning rate. `lr` is included for backward
            compatibility, recommended to use `learning_rate` instead.
        batch_size:       int >= 1. Train input batch size; used for normalization
        total_iterations: int >= 0. Total expected iterations / weight updates
                          throughout training, used for normalization; <1>
        weight_decays:    dict / None. Name-value pairs specifying weight decays,
                          as {<weight matrix name>:<weight decay value>}; <2>
        lr_multipliers:   dict / None. Name-value pairs specifying per-layer lr
                          multipliers, as {<layer name>:<multiplier value>}; <2>
        use_cosine_annealing: bool. If True, multiplies lr each train iteration
                              as a function of eta_min, eta_max, total_iterations,
                              and t_cur (current); [2]-Appendix, 2
        eta_min, eta_max: int, int. Min & max values of cosine annealing
                          lr multiplier; [2]-Appendix, 2
        t_cur: int. Value to initialize t_cur to - used for 'warm restarts'.
               To be used together with use_cosine_annealing==True
        total_iterations_wd: int / None. If not None, weight_decays will be
                     applied according to total_iterations_wd instead of
                     total_iterations, contrary to authors' scheme. Set to
                     sum(total_iterations) over all restarts to normalize over
                     all epochs. May yield improvement over `None`.
        init_verbose: bool. If True, print weight-name--weight-decay, and
                      lr-multiplier--layer-name value pairs set during
                      optimizer initialization (recommended)
    # <1> - if using 'warm restarts', then refers to total expected iterations
            for a given restart; can be an estimate, and training won't stop
            at iterations == total_iterations. [2]-Appendix, pg 1
    # <2> - [AdamW Keras Implementation - Github repository]
            (https://github.com/OverLordGoldDragon/keras_adamw)
    # References
        - [1][Adam - A Method for Stochastic Optimization]
             (http://arxiv.org/abs/1412.6980v8)
        - [2][Fixing Weight Decay Regularization in Adam]
             (https://arxiv.org/abs/1711.05101)
    """

    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, amsgrad=False,
                 total_iterations=0,
                 total_iterations_wd=None, use_cosine_annealing=False,
                 weight_decays=None, lr_multipliers=None, init_verbose=True,
                 eta_min=0, eta_max=1, name="AdamW", **kwargs):

        super(AdamW, self).__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('decay', self._initial_decay)
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)

        self.eta_min = K.constant(eta_min, name='eta_min')
        self.eta_max = K.constant(eta_max, name='eta_max')
        if use_cosine_annealing:
            self._eta_t = None
            self._iter_updates = None
        self.total_iterations = total_iterations
        self.total_iterations_wd = total_iterations_wd or total_iterations
        self.lr_multipliers = lr_multipliers
        self.weight_decays = weight_decays or {}
        self.init_verbose = init_verbose
        self.use_cosine_annealing = use_cosine_annealing
        self.epsilon = epsilon or backend_config.epsilon()
        self.amsgrad = amsgrad

        _check_args(total_iterations, use_cosine_annealing, self.weight_decays)
        self._updates_processed = 0  # to track num calls to '_resource_apply_...'
        self._init_notified = False

    @property
    def eta_t(self):
        if self._eta_t is None:
            self._eta_t = self.add_weight(
                "eta_t",
                shape=[],
                dtype=tf.float32,
                trainable=False,
                aggregation=tf.VariableAggregation.NONE,
                initializer='ones'
            )
            self._weights.append(self._eta_t)
        return self._eta_t

    @property
    def iter_updates(self):
        if self._iter_updates is None:
            self._iter_updates = self.add_weight(
                "iter_updates",
                shape=[],
                dtype=tf.int64,
                trainable=False,
                aggregation=tf.VariableAggregation.NONE,
            )
            self._weights.append(self._iter_updates)
        return self._iter_updates

    def minimize(self, *args,**kwargs):
        """Minimize `loss` by updating `var_list`.
        This method simply computes gradient using `tf.GradientTape` and calls
        `apply_gradients()`. If you want to process the gradient before applying
        then call `tf.GradientTape` and `apply_gradients()` explicitly instead
        of using this function.
        Args:
          loss: A callable taking no arguments which returns the value to minimize.
          var_list: list or tuple of `Variable` objects to update to minimize
            `loss`, or a callable returning the list or tuple of `Variable` objects.
            Use callable when the variable list would otherwise be incomplete before
            `minimize` since the variables are created at the first time `loss` is
            called.
          grad_loss: Optional. A `Tensor` holding the gradient computed for `loss`.
          name: Optional name for the returned operation.
        Returns:
          An `Operation` that updates the variables in `var_list`. The `iterations`
          will be automatically increased by 1.
        Raises:
          ValueError: If some of the variables are not `Variable` objects.
        """
        if self.use_cosine_annealing:
            with backend.name_scope(self._name):
                with ops.init_scope:
                    _ = self.iter_updates
                    _ = self.eta
        return super(AdamW, self).minimize(*args,**kwargs)

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        # Separate for-loops to respect the ordering of slot variables from v1.
        for var in var_list:
            self.add_slot(var, 'm')
        for var in var_list:
            self.add_slot(var, 'v')
        if self.amsgrad:
            for var in var_list:
                self.add_slot(var, 'vhat')
        self._updates_per_iter = len(var_list)

    def _resource_apply_dense(self, grad, var, **kwargs):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        beta_1_t = array_ops.identity(self._get_hyper('beta_1', var_dtype))
        beta_2_t = array_ops.identity(self._get_hyper('beta_2', var_dtype))
        beta_1_power = math_ops.pow(beta_1_t, local_step)
        beta_2_power = math_ops.pow(beta_2_t, local_step)
        epsilon_t = ops.convert_to_tensor(self.epsilon, var_dtype)
        total_iterations = self.total_iterations

        lr_t = lr_t * math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)

        # Learning rate multipliers
        if self.lr_multipliers is not None:
            lr_t = _apply_lr_multiplier(self, lr_t, var)
        # Cosine annealing
        if self.use_cosine_annealing and total_iterations != 0:
            self._eta_t = _compute_eta_t(self)

        m_t = state_ops.assign(m,
                               beta_1_t * m + (1.0 - beta_1_t) * grad,
                               use_locking=self._use_locking)
        v_t = state_ops.assign(v,
                               beta_2_t * v + (1.0 - beta_2_t
                                               ) * math_ops.square(grad),
                               use_locking=self._use_locking)
        if self.amsgrad:
            vhat = self.get_slot(var, 'vhat')
            vhat_t = state_ops.assign(vhat,
                                      math_ops.maximum(vhat, v_t),
                                      use_locking=self._use_locking)
            var_delta = m_t / (math_ops.sqrt(vhat_t) + epsilon_t)
        else:
            var_delta = m_t / (math_ops.sqrt(v_t) + epsilon_t)

        if self.use_cosine_annealing:
            var_t = math_ops.sub(var, self._eta_t * lr_t * var_delta)
        else:
            var_t = math_ops.sub(var, lr_t * var_delta)
        # Weight decays
        if var.name in self.weight_decays.keys() and total_iterations != 0:
            var_t = _apply_weight_decays(self, var, var_t)

        iteration_done = self._updates_processed == (self._updates_per_iter - 1)
        _up = self._updates_processed
        self._updates_processed = (_up + 1) if not iteration_done else 0
        if iteration_done and not self._init_notified:
            self._init_notified = True

        var_update = state_ops.assign(var, var_t, use_locking=self._use_locking)
        updates = [var_update, m_t, v_t]
        if self.use_cosine_annealing:
            t_cur = state_ops.assign_add(self._iter_updates, int(iteration_done),
                                         use_locking=self._use_locking)
            updates.append(t_cur)
        if self.amsgrad:
            updates.append(vhat_t)
        return control_flow_ops.group(*updates)

    def _resource_apply_sparse(self, grad, var, indices, **kwargs):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        beta_1_t = array_ops.identity(self._get_hyper('beta_1', var_dtype))
        beta_2_t = array_ops.identity(self._get_hyper('beta_2', var_dtype))
        beta_1_power = math_ops.pow(beta_1_t, local_step)
        beta_2_power = math_ops.pow(beta_2_t, local_step)
        epsilon_t = ops.convert_to_tensor(self.epsilon, var_dtype)
        total_iterations = self.total_iterations

        lr_t = lr_t * math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)

        # Learning rate multipliers
        if self.lr_multipliers is not None:
            lr_t = _apply_lr_multiplier(self, lr_t, var)
        # Cosine annealing
        if self.use_cosine_annealing and total_iterations != 0:
            self._eta_t = _compute_eta_t(self)

        m_scaled_g_values = grad * (1 - beta_1_t)
        m_t = state_ops.assign(m, m * beta_1_t, use_locking=self._use_locking)
        with ops.control_dependencies([m_t]):
            m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)

        v_scaled_g_values = (grad * grad) * (1 - beta_2_t)
        v_t = state_ops.assign(v, v * beta_2_t, use_locking=self._use_locking)
        with ops.control_dependencies([v_t]):
            v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)

        if self.amsgrad:
            vhat = self.get_slot(var, 'vhat')
            vhat_t = state_ops.assign(vhat,
                                      math_ops.maximum(vhat, v_t),
                                      use_locking=self._use_locking)
            var_delta = m_t / (math_ops.sqrt(vhat_t) + epsilon_t)
        else:
            var_delta = m_t / (math_ops.sqrt(v_t) + epsilon_t)

        if self.use_cosine_annealing:
            var_t = math_ops.sub(var, self._eta_t * lr_t * var_delta)
        else:
            var_t = math_ops.sub(var, lr_t * var_delta)

        # Weight decays
        if var.name in self.weight_decays.keys() and total_iterations != 0:
            var_t = _apply_weight_decays(self, var, var_t)

        iteration_done = self._updates_processed == (self._updates_per_iter - 1)
        _up = self._updates_processed
        self._updates_processed = (_up + 1) if not iteration_done else 0
        if iteration_done and not self._init_notified:
            self._init_notified = True

        var_update = state_ops.assign(var, var_t, use_locking=self._use_locking)
        updates = [var_update, m_t, v_t]
        if self.use_cosine_annealing:
            t_cur = state_ops.assign_add(self._iter_updates, int(iteration_done),
                                         use_locking=self._use_locking)
            updates.append(t_cur)
        if self.amsgrad:
            updates.append(vhat_t)
        return control_flow_ops.group(*updates)

    def set_weights(self, weights):
        params = self.weights
        # If the weights are generated by Keras V1 optimizer, it includes vhats
        # even without amsgrad, i.e, V1 optimizer has 3x + 1 variables, while V2
        # optimizer has 2x + 1 variables. Filter vhats out for compatibility.
        num_vars = int((len(params) - 1) / 2)
        if len(weights) == 3 * num_vars + 1:
            weights = weights[:len(params)]
        super(AdamW, self).set_weights(weights)

    def get_config(self):
        config = super(AdamW, self).get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'decay': self._serialize_hyperparameter('decay'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad,
            'total_iterations': int(self.total_iterations),
            'weight_decays': self.weight_decays,
            'use_cosine_annealing': self.use_cosine_annealing,
            'eta_min': float(K_eval(self.eta_min)),
            'eta_max': float(K_eval(self.eta_max)),
            'init_verbose': self.init_verbose
        })
        return config
