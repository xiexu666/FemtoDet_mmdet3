# Copyright (c) OpenMMLab. All rights reserved.
from .quadratic_warmup import (QuadraticWarmupLR, QuadraticWarmupMomentum,
                               QuadraticWarmupParamScheduler)
from .femto_warmup import FemtoWarmupLR
from .femto_cosin import FemtoCosineAnnealingLR
__all__ = [
    'QuadraticWarmupParamScheduler', 'QuadraticWarmupMomentum',
    'QuadraticWarmupLR','FemtoWarmupLR','FemtoCosineAnnealingLR'
]
