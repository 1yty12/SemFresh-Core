"""Explicit caller-supplied physical parameters; no paper experiment defaults."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class CoreConfig:
    N: int
    M: int
    f_max: object
    P_max: object
    d_t: float
    kappa: float
    sigma2: float
    W_max: float
    gamma: float
    d_ub: float
    lambda_adoi: float
    F_edge_max: float
    edge_load_alpha: float
    task_prediction_feature_dim: int = 0

    def __post_init__(self):
        for name in ('N', 'M'):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(name + ' must be a positive integer')
        for name in ('f_max', 'P_max'):
            value = np.asarray(getattr(self, name), dtype=np.float32)
            if value.shape != (self.N,) or not np.isfinite(value).all() or (value <= 0).any():
                raise ValueError(name + ' must contain N positive finite values')
        for name in ('d_t', 'kappa', 'sigma2', 'W_max', 'd_ub', 'lambda_adoi', 'F_edge_max'):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(name + ' must be positive and finite')
        if not np.isfinite(self.gamma) or self.gamma < 0:
            raise ValueError('gamma must be finite and nonnegative')
        if not np.isfinite(self.edge_load_alpha) or not 0 <= self.edge_load_alpha <= 1:
            raise ValueError('edge_load_alpha must be in [0, 1]')
        if not isinstance(self.task_prediction_feature_dim, int) or self.task_prediction_feature_dim < 0:
            raise ValueError('task_prediction_feature_dim must be a nonnegative integer')
