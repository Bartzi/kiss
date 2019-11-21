from __future__ import division
import math
import warnings

import numpy

from chainer import backend
from chainer.backends import cuda
from chainer.backends import intel64
from chainer import optimizer
from chainer import types


if types.TYPE_CHECKING:
    import typing_extensions as tpe

    class AdamHyperparameter(tpe.Protocol):
        """Protocol class for hyperparameter of RAdam.

        This is only for PEP 544 compliant static type checkers.
        """
        alpha = None  # type: float
        beta1 = None  # type: float
        beta2 = None  # type: float
        eps = None  # type: float
        weight_decay_rate = None  # type: float


_default_hyperparam = optimizer.Hyperparameter()  # type: AdamHyperparameter # NOQA
_default_hyperparam.alpha = 0.001
_default_hyperparam.beta1 = 0.9
_default_hyperparam.beta2 = 0.999
_default_hyperparam.eps = 1e-8
_default_hyperparam.weight_decay_rate = 0


def _learning_rate(hp, t):
    if t == 0:
        raise RuntimeError(
            'Can\'t determine the learning rate of Adam optimizer '
            'because the update steps have not been started.')
    fix1 = 1. - math.pow(hp.beta1, t)
    fix2 = 1. - math.pow(hp.beta2, t)
    return hp.alpha * math.sqrt(fix2) / fix1


def _get_intermediate_dtype(dtype):
    # Returns the dtype for intermediate calculation.
    # For float16 input, float32 is used.
    # Otherwise the same dtype as the parameter is used.
    if dtype == numpy.float16:
        return numpy.float32
    return dtype


def _inplace_axpby(x, a, b, y):
    # in-place axpby: x = a * x + b * y
    if isinstance(x, intel64.mdarray):
        x.inplace_axpby(a, b, y)
    else:
        if a == 1:
            x += b * y
        else:
            x[:] = a * x + b * y


class RAdamRule(optimizer.UpdateRule):

    """Update rule of RAdam optimization algorithm.

    See: `On the Variance of the Adaptive Learning Rate and Beyond \
          <https://arxiv.org/abs/1908.03265v1>`_

    See :class:`optimizers.RAdam` for the default values
    of the hyperparameters.

    Args:
        parent_hyperparam (~chainer.optimizer.Hyperparameter): Hyperparameter
            that provides the default values.
        alpha (float): Coefficient of learning rate.
        beta1 (float): Exponential decay rate of the first order moment.
        beta2 (float): Exponential decay rate of the second order moment.
        eps (float): Small value for the numerical stability.
        weight_decay_rate (float): Weight decay rate.

    """
    _kernel = None

    # Only used in `update_core_gpu`.
    # A dummy ndarray to help ElementwiseKernel deduce generic type T as
    # `dtype`.
    # It cannot be deduced only by scalar arguments.
    _dummy = None

    def __init__(self, parent_hyperparam=None,
                 alpha=None, beta1=None, beta2=None, eps=None,
                 weight_decay_rate=None):
        super(RAdamRule, self).__init__(
            parent_hyperparam or _default_hyperparam)
        if alpha is not None:
            self.hyperparam.alpha = alpha
        if beta1 is not None:
            self.hyperparam.beta1 = beta1
        if beta2 is not None:
            self.hyperparam.beta2 = beta2
        if eps is not None:
            self.hyperparam.eps = eps
        if weight_decay_rate is not None:
            self.hyperparam.weight_decay_rate = weight_decay_rate

        self.rho_max = 2 / (1 - self.hyperparam.beta2) - 1

    def init_state(self, param):
        xp = backend.get_array_module(param.data)
        with cuda.get_device_from_array(param.data):
            self.state['m'] = xp.zeros_like(param.data)
            self.state['v'] = xp.zeros_like(param.data)

        # For iDeep
        if isinstance(param.data, intel64.mdarray):
            self.state['m'] = intel64.ideep.array(
                self.state['m'], itype=intel64.ideep.wgt_array)
            self.state['v'] = intel64.ideep.array(
                self.state['v'], itype=intel64.ideep.wgt_array)

    def _check_eps(self, interm_dtype):
        # Checks that the eps does not underflow.
        hp = self.hyperparam
        eps = interm_dtype(hp.eps)
        if hp.eps != 0 and eps == 0:
            raise ValueError(
                'eps of Adam optimizer is too small for {} ({})'.format(
                    interm_dtype.name, hp.eps))
        # Note that the converted `eps` (numpy scalar) is discarded here and
        # the original `hp.eps` is used in calculation, because Python
        # scalars are faster in cupy elementwise kernels.

    def update_core_cpu(self, param):
        grad = param.grad
        if grad is None:
            return
        hp = self.hyperparam
        dtype = _get_intermediate_dtype(param.dtype.type)
        self._check_eps(dtype)
        grad = grad.astype(dtype, copy=False)

        m, v = self.state['m'], self.state['v']

        # m += (1 - beta1) * (grad - m)
        _inplace_axpby(m, 1.0, 1.0 - hp.beta1, grad - m)
        # v += (1 - beta2) * (grad * grad - v)
        _inplace_axpby(v, 1.0, 1.0 - hp.beta2, grad*grad - v)

        beta2_t = hp.beta2 ** self.t
        rho_t = self.rho_max - 2 * self.t * beta2_t / (1 - beta2_t)

        m_hat = m / (1 - hp.beta1 ** self.t)
        if rho_t >= 5:
            v_hat = numpy.sqrt(v / (1 - beta2_t))
            r_t = numpy.sqrt(
                ((rho_t - 4) * (rho_t - 2) * self.rho_max) /
                ((self.rho_max - 4) * (self.rho_max - 2) * rho_t)
            )
            step_size = self.alpha_t * r_t * m_hat / (v_hat + hp.eps)
        else:
            step_size = self.alpha_t * m_hat

        _inplace_axpby(param.data, 1.0 - hp.weight_decay_rate, 1.0, -step_size)

    def update_core_gpu(self, param):
        grad = param.grad
        if grad is None:
            return
        hp = self.hyperparam
        dtype = _get_intermediate_dtype(param.dtype.type)
        self._check_eps(dtype)

        if self._dummy is None:
            self._dummy = cuda.cupy.empty((0,), dtype=dtype)

        if RAdamRule._kernel is None:
            RAdamRule._kernel = cuda.elementwise(
                'P grad, T alpha_t, T beta1, T beta2, T rho_max,'
                'T eps, T weight_decay_rate, T step, raw T dummy',
                'P param, P m, P v',
                '''T grad_ = static_cast<T>(grad);
                   m += (1 - beta1) * (grad_ - m);
                   v += (1 - beta2) * (grad_ * grad_ - v);
                   T beta2_t = pow(beta2, step); 
                   T rho_t = rho_max - 2 * step * beta2_t / (1 - beta2_t);
                   T m_hat = m / (1 - pow(beta1, step));
                   T step_size = alpha_t * m_hat;
                   
                   if (rho_t >= 5) {
                      T v_hat = sqrt(v / (1 - beta2_t));
                      T r_t = sqrt(((rho_t - 4) * (rho_t - 2) * rho_max) / ((rho_max - 4) * (rho_max - 2) * rho_t));
                      step_size *= r_t / (v_hat + eps);   
                   }
                   
                   param -= step_size + weight_decay_rate * param;
                ''',
                'adam')

        RAdamRule._kernel(
            grad, self.alpha_t, hp.beta1,
            hp.beta2, self.rho_max, hp.eps,
            hp.weight_decay_rate, self.t, self._dummy,
            param.data, self.state['m'], self.state['v'])

    @property
    def alpha_t(self):
        return _learning_rate(self.hyperparam, self.t)


class RAdam(optimizer.GradientMethod):

    """Rectified Adam optimizer.

    See: `On the Variance of the Adaptive Learning Rate and Beyond \
          <https://arxiv.org/abs/1908.03265>`_

    Args:
        alpha (float): Coefficient of learning rate.
        beta1 (float): Exponential decay rate of the first order moment.
        beta2 (float): Exponential decay rate of the second order moment.
        eps (float): Small value for the numerical stability.
        eta (float): Schedule multiplier, can be used for warm restarts.
        weight_decay_rate (float): Weight decay rate.

    """

    def __init__(self,
                 alpha=_default_hyperparam.alpha,
                 beta1=_default_hyperparam.beta1,
                 beta2=_default_hyperparam.beta2,
                 eps=_default_hyperparam.eps,
                 weight_decay_rate=_default_hyperparam.weight_decay_rate):
        super(RAdam, self).__init__()
        self.hyperparam.alpha = alpha
        self.hyperparam.beta1 = beta1
        self.hyperparam.beta2 = beta2
        self.hyperparam.eps = eps
        self.hyperparam.weight_decay_rate = weight_decay_rate

    alpha = optimizer.HyperparameterProxy('alpha')
    beta1 = optimizer.HyperparameterProxy('beta1')
    beta2 = optimizer.HyperparameterProxy('beta2')
    eps = optimizer.HyperparameterProxy('eps')
    weight_decay_rate = optimizer.HyperparameterProxy('weight_decay_rate')

    def create_update_rule(self):
        return RAdamRule(self.hyperparam)

    @property
    def alpha_t(self):
        return _learning_rate(self.hyperparam, self.t)

    @property
    def lr(self):
        warnings.warn(
            'RAdam.lr has been renamed to RAdamRule.alpha_t. '
            'Use of Adam.lr is deprecated in Chainer v6.',
            DeprecationWarning)
        return self.alpha_t
