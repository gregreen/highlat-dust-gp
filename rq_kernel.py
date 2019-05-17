import math
import torch
from gpytorch.kernels import Kernel
from gpytorch.constraints import Positive


class RationalQuadraticKernel(Kernel):
    r"""
    The "Rational Quadratic" kernel.
    
    Shorter than some length scale, the kernel is 1. At large
    radii, the kernel becomes a power law in distance.
    
    The kernel is given by
    $K \left( r \right)
    = \left(
        1 + \frac{r^2}{2 \alpha \ell^2}
    \right)^{-\alpha}$,
    which asymptotes to
    $K \left( r \right)
    \simeq
    \left(2 \alpha \right)^{\alpha}
    \left( \frac{r}{\ell} \right)^{-2 \alpha}$
    for $r \gg \ell$.

    The kernel has two parameters:
    1. "lengthscale": $\ell$ in the above equations, the
       typical radius within which the kernel asymptotes, related
       to the short cutoff scale in the power spectrum.
    2. "power_law": $\alpha$ in the above equations, equal to
       half the power-law slope at large radii.
    """

    def __init__(self, power_law_prior=None,
                       power_law_constraint=None,
                       **kwargs):
        super(RationalQuadraticKernel, self).__init__(
            has_lengthscale=True,
            **kwargs
        )

        self.register_parameter(
            name="raw_power_law",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        )

        if power_law_constraint is None:
            power_law_constraint = Positive()

        if power_law_prior is not None:
            self.register_prior(
                "power_law_prior",
                power_law_prior,
                lambda: self.power_law,
                lambda v: self._set_power_law(v),
            )

        self.register_constraint("raw_power_law", power_law_constraint)

    @property
    def power_law(self):
        return self.raw_power_law_constraint.transform(self.raw_power_law)

    @power_law.setter
    def power_law(self, value):
        return self._set_power_law(value)

    def _set_power_law(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_power_law)

        self.initialize(
            raw_power_law=self.raw_power_law_constraint.inverse_transform(value)
        )

    def forward(self, x1, x2, **params):
        dist2 = self.covar_dist(x1, x2, square_dist=True, **params)
        scaled_dist2 = dist2.div(
                self.lengthscale.pow(2).mul(self.power_law.mul(2))
        )
        res = scaled_dist2.add(1).pow(self.power_law.mul(-1))

        #ell = self.lengthscale.item()
        #alpha = self.power_law.item()
        #res_est = rational_quadratic(dist2.numpy(), ell, alpha)
        #print(torch.sqrt(dist2))
        #print(res)
        #print(res_est)

        if dist2.ndimension() == 2:
            res = res.squeeze(0)
        return res


def rational_quadratic(r2, l, a):
    return (1. + r2/(2.*a*l**2.))**(-a)

