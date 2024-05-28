from ..tensor import Tensor
from .loss import MSELoss, BCELoss
import numpy as np


class Linear:
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mean: float = 0.0,
        std: float = 1.0,
        int_type: str = "he",
    ):
        if int_type == "xavier":
            self.weight = Tensor.xavier(in_features, out_features, requires_grad=True)
        elif int_type == "he":
            self.weight = Tensor.he(in_features, out_features, requires_grad=True)
        else:
            self.weight = Tensor.normal(
                mean, std, (out_features, in_features), requires_grad=True
            )
        self.bias = (
            Tensor(np.zeros((1, out_features)), requires_grad=True) if bias else None
        )  #  # set bias 'b' to zeros
        # self.bias = (
        #    Tensor.normal(mean, 0, (1, out_features), requires_grad=True)
        #    if bias
        #    else None
        # )

    def __call__(self, x: "Tensor") -> "Tensor":
        return x.linear(self.weight.T, self.bias)
