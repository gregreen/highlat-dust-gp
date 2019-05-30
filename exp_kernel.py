import math
import torch
from gpytorch.kernels import Kernel
from gpytorch.constraints import Positive


class ExponentialKernel(Kernel):
    r"""
    The exponential kernel.
    
    The kernel falls off exponentially with distance.

    The kernel is given by
    $K \left( r \right)
    = \exp \left(
        -\frac{r}{\ell}
    \right)$

    The kernel has one parameter:
    1. "lengthscale": $\ell$ in the above equation, the
       decay radius.
    """

    def __init__(self, **kwargs):
        super(ExponentialKernel, self).__init__(
            has_lengthscale=True,
            **kwargs
        )

    def forward(self, x1, x2, **params):
        dist = self.covar_dist(x1, x2, **params)
        scaled_dist = dist.div(self.lengthscale)
        res = scaled_dist.mul(-1).exp()

        if (dist.ndimension() == 2) or params.get('diag', False):
            res = res.squeeze(0)
        
        return res

