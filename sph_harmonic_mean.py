import math
import torch
from gpytorch.means import Mean
from gpytorch.utils.broadcasting import _mul_broadcast_shape
from gpytorch.utils.deprecation import _deprecate_kwarg_with_transform


class SphericalHarmonicMean(Mean):
    def __init__(self, prior=None, batch_shape=torch.Size(), **kwargs):
        batch_shape = _deprecate_kwarg_with_transform(
            kwargs,
            "batch_size",
            "batch_shape",
            batch_shape,
            lambda n: torch.Size([n])
        )
        super(SphericalHarmonicMean, self).__init__()
        self.batch_shape = batch_shape
        self.register_parameter(
            name="raw_a_lm",
            parameter=torch.nn.Parameter(torch.zeros(*batch_shape, 16))
        )
        if prior is not None:
            self.register_prior("a_lm_prior", prior, "raw_a_lm")

    
    @property
    def a_lm(self):
        return self.raw_a_lm

    @a_lm.setter
    def a_lm(self, value):
        return self._set_a_lm(value)

    def _set_a_lm(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_a_lm)

        self.initialize(raw_a_lm=value)

    def forward(self, input):
        x = input[...,0]
        y = input[...,1]
        z = input[...,2]
        mu = (
            # l = 0
            self.a_lm[0]
            # l = 1
            + self.a_lm[1] * y
            + self.a_lm[2] * z
            + self.a_lm[3] * x
            # l = 2
            + self.a_lm[4] * x*y
            + self.a_lm[5] * y*z
            + self.a_lm[6] * (2*z.pow(2) - x.pow(2) - y.pow(2))
            + self.a_lm[7] * z*x
            + self.a_lm[8] * (x.pow(2) - y.pow(2))
            # l = 3
            + self.a_lm[9] * (3*x.pow(2) - y.pow(2)) * y
            + self.a_lm[10] * x*y*z
            + self.a_lm[11] * y * (4*z.pow(2) - x.pow(2) - y.pow(2))
            + self.a_lm[12] * z * (2*z.pow(2) - 3*x.pow(2) - 3*y.pow(2))
            + self.a_lm[13] * x * (4*z.pow(2) - x.pow(2) - y.pow(2))
            + self.a_lm[14] * (x.pow(2) - y.pow(2)) * z
            + self.a_lm[15] * (x.pow(2) - 3*y.pow(2)) * x
        )
        if input.shape[:-2] == self.batch_shape:
            return mu
        else:
            # TODO: test this
            return mu
            #self.constant.expand(
            #    _mul_broadcast_shape(input.shape[:-1], self.constant.shape)
            #)
