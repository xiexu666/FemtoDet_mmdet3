import math
from typing import Optional,  Union

from torch.optim import Optimizer
from mmengine.optim.scheduler.lr_scheduler import LRSchedulerMixin
from mmengine.optim import OptimWrapper

from mmengine.optim.scheduler.param_scheduler import INF,_ParamScheduler
from mmengine.registry import PARAM_SCHEDULERS


@PARAM_SCHEDULERS.register_module()
class FemtoCosineAnnealingParamScheduler(_ParamScheduler):
    def __init__(self,
                 optimizer: Union[Optimizer, OptimWrapper],
                 param_name: str,
                 T_max: Optional[int] = None,
                 eta_min: Optional[float] = None,
                 begin: int = 0,
                 end: int = INF,
                 last_step: int = -1,
                 by_epoch: bool = True,
                 verbose: bool = False,
                 eta_min_ratio: Optional[float] = None):
        # To preserve backwards compatibility
        if eta_min is None and eta_min_ratio is None:
            eta_min = 0.
        assert (eta_min is None) ^ (eta_min_ratio is None), \
            'Either `eta_min` or `eta_min_ratio should be specified'
        self.T_max = T_max or (end - begin)
        self.eta_min = eta_min
        self.eta_min_ratio = eta_min_ratio
        super().__init__(
            optimizer,
            param_name=param_name,
            begin=begin,
            end=end,
            last_step=last_step,
            by_epoch=by_epoch,
            verbose=verbose)

    @classmethod
    def build_iter_from_epoch(cls,
                              *args,
                              T_max=None,
                              begin=0,
                              end=INF,
                              by_epoch=True,
                              epoch_length=None,
                              **kwargs):
        """Build an iter-based instance of this scheduler from an epoch-based
        config."""
        assert by_epoch, 'Only epoch-based kwargs whose `by_epoch=True` can ' \
                         'be converted to iter-based.'
        assert epoch_length is not None and epoch_length > 0, \
            f'`epoch_length` must be a positive integer, ' \
            f'but got {epoch_length}.'
        by_epoch = False
        if T_max is not None:
            T_max = T_max * epoch_length
        begin = int(begin * epoch_length)
        if end != INF:
            end = int(end * epoch_length)
        return cls(
            *args,
            T_max=T_max,
            begin=begin,
            end=end,
            by_epoch=by_epoch,
            **kwargs)

    def annealing_cos(self,start, end, factor, weight=1):
        """Calculate annealing cos learning rate.

        Cosine anneal from `weight * start + (1 - weight) * end` to `end` as
        percentage goes from 0.0 to 1.0.

        Args:
            start (float): The starting learning rate of the cosine annealing.
            end (float): The ending learing rate of the cosine annealing.
            factor (float): The coefficient of `pi` when calculating the current
                percentage. Range from 0.0 to 1.0.
            weight (float, optional): The combination factor of `start` and `end`
                when calculating the actual starting learning rate. Default to 1.
        """
        cos_out = math.cos(math.pi * factor) + 1
        return end + 0.5 * weight * (start - end) * cos_out
    
    
    def _get_value(self) -> list:
        """Compute value using chainable form of the scheduler."""

        def _get_eta_min(base_value):
            if self.eta_min_ratio is None:
                return self.eta_min
            return base_value * self.eta_min_ratio

        if self.last_step == 0:
            return [
                group[self.param_name] for group in self.optimizer.param_groups
            ]
        return [self.annealing_cos(base_value,_get_eta_min(base_value),(self._global_step - self.begin)/(self.end - self.begin)) for base_value, group in zip(
                    self.base_values, self.optimizer.param_groups)]

@PARAM_SCHEDULERS.register_module()
class FemtoCosineAnnealingLR(LRSchedulerMixin, FemtoCosineAnnealingParamScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial value and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::
        \begin{aligned}
            \eta_t & = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1
            + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right),
            & T_{cur} \neq (2k+1)T_{max}; \\
            \eta_{t+1} & = \eta_{t} + \frac{1}{2}(\eta_{max} - \eta_{min})
            \left(1 - \cos\left(\frac{1}{T_{max}}\pi\right)\right),
            & T_{cur} = (2k+1)T_{max}.
        \end{aligned}

    Notice that because the schedule
    is defined recursively, the learning rate can be simultaneously modified
    outside this scheduler by other operators. If the learning rate is set
    solely by this scheduler, the learning rate at each step becomes:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this
    only implements the cosine annealing part of SGDR, and not the restarts.

    Args:
        optimizer (Optimizer or OptimWrapper): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Defaults to None.
        begin (int): Step at which to start updating the learning rate.
            Defaults to 0.
        end (int): Step at which to stop updating the learning rate.
            Defaults to INF.
        last_step (int): The index of last step. Used for resume without
            state dict. Defaults to -1.
        by_epoch (bool): Whether the scheduled learning rate is updated by
            epochs. Defaults to True.
        verbose (bool): Whether to print the learning rate for each update.
            Defaults to False.
        eta_min_ratio (float, optional): The ratio of the minimum parameter
            value to the base parameter value. Either `eta_min` or
            `eta_min_ratio` should be specified. Defaults to None.
            New in version 0.3.2.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """
