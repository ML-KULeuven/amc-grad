from typing import Optional

import numpy as np
import torch

from .utils import log1mexp
from .kompyle import RCircuit, Circuit


def to_numpy(tensor):
    assert tensor.ndim == 2
    return tensor.detach().cpu().numpy()


def to_torch(arr, device=None, dtype=None):
    return torch.as_tensor(np.array(arr), device=device, dtype=dtype)


class CircuitFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pos_weights, neg_weights, circuit):
        # weights: (nb_vars, batch_size)
        device, dtype = pos_weights.device, pos_weights.dtype
        pos_weights, neg_weights = to_numpy(pos_weights).astype(np.float64), to_numpy(neg_weights).astype(np.float64)
        result, pos_grads, neg_grads = circuit._wmc_np_grad(pos_weights, neg_weights)
        ctx.save_for_backward(
            to_torch(pos_grads, device, dtype),
            to_torch(neg_grads, device, dtype)
        )
        return to_torch(result, device, dtype)

    @staticmethod
    def backward(ctx, grad_output):
        pos_grads, neg_grads = ctx.saved_tensors
        return pos_grads * grad_output, neg_grads * grad_output, None


class LogCircuitFunction(CircuitFunction):
    @staticmethod
    def forward(ctx, pos_weights, neg_weights, circuit):
        device, dtype = pos_weights.device, pos_weights.dtype
        log_wmc = circuit._log_wmc_np_grad(to_numpy(pos_weights), to_numpy(neg_weights))
        result, pos_grads, neg_grads = (to_torch(t, device, dtype) for t in log_wmc)

        pos_grads = (pos_grads - result + pos_weights).exp()
        neg_grads = (neg_grads - result + neg_weights).exp()
        ctx.save_for_backward(pos_grads, neg_grads)
        return result


class CircuitModule(torch.nn.Module):
    def __init__(self, circuit, semiring="real"):
        super(CircuitModule, self).__init__()
        if isinstance(circuit, RCircuit):
            circuit = circuit.compile(circuit.nb_vars())
        assert isinstance(circuit, Circuit)
        self.circuit = circuit
        self.forward = {
            "real": self.real_forward,
            "log": self.log_forward,
        }[semiring]

    def real_forward(self, weights: torch.Tensor, neg_weights: Optional[torch.Tensor] = None):
        if weights.requires_grad:
            if neg_weights is None:
                neg_weights = 1 - weights
            return CircuitFunction.apply(weights, neg_weights, self.circuit)
        else:
            assert neg_weights is None, "not implemented"
            if weights.ndim == 1:
                wmc = self.circuit.wmc(weights.tolist())
            else:
                assert weights.ndim == 2
                wmc = self.circuit.wmc_np(weights.numpy())
            return torch.tensor(wmc, dtype=torch.float64)

    def log_forward(self, weights: torch.Tensor, neg_weights: Optional[torch.Tensor] = None):
        if weights.requires_grad:
            if neg_weights is None:
                neg_weights = log1mexp(weights)
            return LogCircuitFunction.apply(weights, neg_weights, self.circuit)
        else:
            assert neg_weights is None, "not implemented"
            log_wmc = self.circuit.log_wmc(weights.tolist())
            return torch.tensor(log_wmc, dtype=torch.float64)
