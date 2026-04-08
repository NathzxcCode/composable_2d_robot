import torch
from typing import List, Callable, Optional, Union

class Gaussian:
    def __init__(self, dim: int, eta: Optional[torch.Tensor]=None, lam: Optional[torch.Tensor]=None, type: torch.dtype = torch.float):
        self.dim = dim

        if eta is not None and eta.shape == torch.Size([dim]):
            self.eta = eta.type(type)
        else:
            self.eta = torch.zeros(dim, dtype=type)

        if lam is not None and lam.shape == torch.Size([dim, dim]):
            self.lam = lam.type(type)
        else:
            self.lam = torch.zeros([dim, dim], dtype=type)

    def mean(self) -> torch.Tensor:
        return torch.matmul(torch.inverse(self.lam), self.eta)

    def cov(self) -> torch.Tensor:
        return torch.inverse(self.lam)

    def mean_and_cov(self) -> List[torch.Tensor]:
        cov = self.cov()
        mean = torch.matmul(cov, self.eta)
        return [mean, cov]

    def set_with_cov_form(self, mean: torch.Tensor, cov: torch.Tensor) -> None:
        self.lam = torch.inverse(cov)
        self.eta = self.lam @ mean

"""
    Defines squared loss functions that correspond to Gaussians.
    Robust losses are implemented by scaling the Gaussian covariance.
"""

class SquaredLoss():
    def __init__(self, dofs: int, diag_cov: Union[float, torch.Tensor]) -> None:
        """
            dofs: dofs of the measurement
            cov: diagonal elements of covariance matrix
        """
        assert diag_cov.shape == torch.Size([dofs])
        mat = torch.zeros(dofs, dofs, dtype=diag_cov.dtype)
        mat[range(dofs), range(dofs)] = diag_cov
        self.cov = mat
        self.effective_cov = mat.clone()

    def get_effective_cov(self, residual: torch.Tensor) -> None:
        """ Returns the covariance of the Gaussian (squared loss) that matches the loss at the error value. """
        self.effective_cov = self.cov.clone()

    def robust(self) -> bool:
        return not torch.equal(self.cov, self.effective_cov)


class HuberLoss(SquaredLoss):
    def __init__(self, dofs: int, diag_cov: Union[float, torch.Tensor], stds_transition: float) -> None:
        """
            stds_transition: num standard deviations from minimum at which quadratic loss transitions to linear
        """
        SquaredLoss.__init__(self, dofs, diag_cov)
        self.stds_transition = stds_transition

    def get_effective_cov(self, residual: torch.Tensor) -> None:
        mahalanobis_dist = torch.sqrt(residual @ torch.inverse(self.cov) @ residual)
        if mahalanobis_dist > self.stds_transition:
            self.effective_cov = self.cov * mahalanobis_dist**2 / (2 * self.stds_transition * mahalanobis_dist - self.stds_transition**2)
        else:
            self.effective_cov = self.cov.clone()


class TukeyLoss(SquaredLoss):
    def __init__(self, dofs: int, diag_cov: Union[float, torch.Tensor], stds_transition: float) -> None:
        """
            stds_transition: num standard deviations from minimum at which quadratic loss transitions to constant
        """
        SquaredLoss.__init__(self, dofs, diag_cov)
        self.stds_transition = stds_transition

    def get_effective_cov(self, residual: torch.Tensor) -> None:
        mahalanobis_dist = torch.sqrt(residual @ torch.inverse(self.cov) @ residual)
        if mahalanobis_dist > self.stds_transition:
            self.effective_cov = self.cov * mahalanobis_dist**2 / self.stds_transition**2
        else:
            self.effective_cov = self.cov.clone()

class MeasModel:
    def __init__(self, meas_fn: Callable, jac_fn: Callable, loss: SquaredLoss, *args) -> None:
        self._meas_fn = meas_fn
        self._jac_fn = jac_fn
        self.loss = loss
        self.args = args
        self.linear = True

    def jac_fn(self, x: torch.Tensor) -> torch.Tensor:
        return self._jac_fn(x, *self.args)

    def meas_fn(self, x: torch.Tensor) -> torch.Tensor:
        return self._meas_fn(x, *self.args)