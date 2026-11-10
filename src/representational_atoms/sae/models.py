from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class SAEOutput:
    reconstruction: torch.Tensor
    pre_activations: torch.Tensor
    activations: torch.Tensor


def _rectangle(x: torch.Tensor) -> torch.Tensor:
    return ((x > -0.5) & (x < 0.5)).to(x)


class _JumpStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = float(bandwidth)
        return (x > threshold).to(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        x_grad = torch.zeros_like(grad_output)
        threshold_grad = torch.sum(
            -(1.0 / bandwidth)
            * _rectangle((x - threshold) / bandwidth)
            * grad_output,
            dim=0,
        )
        return x_grad, threshold_grad, None


class _JumpReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = float(bandwidth)
        return x * (x > threshold).to(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        x_grad = (x > threshold) * grad_output
        threshold_grad = torch.sum(
            -(threshold / bandwidth)
            * _rectangle((x - threshold) / bandwidth)
            * grad_output,
            dim=0,
        )
        return x_grad, threshold_grad, None


class _BiJumpStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, pos_threshold, neg_threshold, bandwidth):
        ctx.save_for_backward(x, pos_threshold, neg_threshold)
        ctx.bandwidth = float(bandwidth)
        return ((x > pos_threshold) | (x < -neg_threshold)).to(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, pos_threshold, neg_threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        x_grad = torch.zeros_like(grad_output)
        pos_grad = torch.sum(
            -(1.0 / bandwidth)
            * _rectangle((x - pos_threshold) / bandwidth)
            * grad_output,
            dim=0,
        )
        # Preserve the estimator used in the original experiment scripts.
        neg_grad = torch.sum(
            -(1.0 / bandwidth)
            * _rectangle((x + neg_threshold) / bandwidth)
            * grad_output,
            dim=0,
        )
        return x_grad, pos_grad, neg_grad, None


class _BiJumpReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, pos_threshold, neg_threshold, bandwidth):
        ctx.save_for_backward(x, pos_threshold, neg_threshold)
        ctx.bandwidth = float(bandwidth)
        return x * ((x > pos_threshold) | (x < -neg_threshold)).to(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, pos_threshold, neg_threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        mask = (x > pos_threshold) | (x < -neg_threshold)
        x_grad = mask * grad_output
        pos_grad = torch.sum(
            -(pos_threshold / bandwidth)
            * _rectangle((x - pos_threshold) / bandwidth)
            * grad_output,
            dim=0,
        )
        neg_grad = torch.sum(
            +(neg_threshold / bandwidth)
            * _rectangle((x + neg_threshold) / bandwidth)
            * grad_output,
            dim=0,
        )
        return x_grad, pos_grad, neg_grad, None


class BaseSAE(nn.Module):
    architecture = "base"

    def __init__(
        self,
        input_dim: int,
        dictionary_width: int,
        s_inv_half: torch.Tensor,
        *,
        use_pre_encoder_bias: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.dictionary_width = int(dictionary_width)
        self.use_pre_encoder_bias = bool(use_pre_encoder_bias)

        w_enc = torch.empty((self.input_dim, self.dictionary_width))
        w_dec = torch.empty((self.dictionary_width, self.input_dim))
        nn.init.xavier_uniform_(w_enc)
        nn.init.xavier_uniform_(w_dec)

        self.W_enc = nn.Parameter(s_inv_half @ w_enc)
        self.W_dec = nn.Parameter(w_dec)
        self.b_enc = nn.Parameter(torch.zeros((self.dictionary_width,)))
        self.b_dec = nn.Parameter(torch.zeros((self.input_dim,)))

    def pre_activations(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_pre_encoder_bias:
            x = x - self.b_dec
        return x @ self.W_enc + self.b_enc

    def activation(self, pre_acts: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.pre_activations(x))

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return acts @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> SAEOutput:
        pre_acts = self.pre_activations(x)
        acts = self.activation(pre_acts)
        return SAEOutput(self.decode(acts), pre_acts, acts)

    def sparsity_measure(self, output: SAEOutput) -> torch.Tensor:
        return torch.sum(output.activations != 0, dim=-1).float()

    def threshold_statistics(self) -> dict:
        return {}


class ReLUSAE(BaseSAE):
    architecture = "relu"

    def activation(self, pre_acts: torch.Tensor) -> torch.Tensor:
        return F.relu(pre_acts)


class TopKSAE(BaseSAE):
    architecture = "topk"

    def __init__(self, *args, k: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = int(k)
        if not 1 <= self.k <= self.dictionary_width:
            raise ValueError(
                f"k must satisfy 1 <= k <= dictionary_width; "
                f"got k={self.k}, dictionary_width={self.dictionary_width}."
            )

    def activation(self, pre_acts: torch.Tensor) -> torch.Tensor:
        topk = torch.topk(pre_acts, k=self.k, dim=-1)
        values = F.relu(topk.values)
        acts = torch.zeros_like(pre_acts)
        acts.scatter_(dim=-1, index=topk.indices, src=values)
        return acts


class AbsTopKSAE(BaseSAE):
    architecture = "abstopk"

    def __init__(self, *args, k: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = int(k)
        if not 1 <= self.k <= self.dictionary_width:
            raise ValueError(
                f"k must satisfy 1 <= k <= dictionary_width; "
                f"got k={self.k}, dictionary_width={self.dictionary_width}."
            )

    def activation(self, pre_acts: torch.Tensor) -> torch.Tensor:
        indices = torch.topk(pre_acts.abs(), k=self.k, dim=-1).indices
        values = torch.gather(pre_acts, dim=-1, index=indices)
        acts = torch.zeros_like(pre_acts)
        acts.scatter_(dim=-1, index=indices, src=values)
        return acts


class JumpReLUSAE(BaseSAE):
    architecture = "jumprelu"

    def __init__(self, *args, threshold_init: float, bandwidth: float, **kwargs):
        super().__init__(*args, **kwargs)
        if threshold_init <= 0:
            raise ValueError("threshold_init must be positive.")
        self.bandwidth = float(bandwidth)
        self.log_threshold = nn.Parameter(
            torch.log(torch.full((self.dictionary_width,), float(threshold_init)))
        )

    @property
    def threshold(self) -> torch.Tensor:
        return torch.exp(self.log_threshold)

    def activation(self, pre_acts: torch.Tensor) -> torch.Tensor:
        return _JumpReLU.apply(pre_acts, self.threshold, self.bandwidth)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre_acts = self.pre_activations(x)
        return (pre_acts > self.threshold) * F.relu(pre_acts)

    def sparsity_measure(self, output: SAEOutput) -> torch.Tensor:
        return torch.sum(_JumpStep.apply(output.pre_activations, self.threshold, self.bandwidth), dim=-1)

    def threshold_statistics(self) -> dict:
        t = self.threshold.detach().cpu()
        return {
            "threshold": {
                "mean": float(t.mean()),
                "min": float(t.min()),
                "max": float(t.max()),
            }
        }


class BiJumpReLUSAE(BaseSAE):
    architecture = "bijumprelu"

    def __init__(self, *args, threshold_init: float, bandwidth: float, **kwargs):
        super().__init__(*args, **kwargs)
        if threshold_init <= 0:
            raise ValueError("threshold_init must be positive.")
        self.bandwidth = float(bandwidth)
        init = torch.full((self.dictionary_width,), float(threshold_init))
        self.log_pos_threshold = nn.Parameter(torch.log(init.clone()))
        self.log_neg_threshold = nn.Parameter(torch.log(init.clone()))

    @property
    def pos_threshold(self) -> torch.Tensor:
        return torch.exp(self.log_pos_threshold)

    @property
    def neg_threshold(self) -> torch.Tensor:
        return torch.exp(self.log_neg_threshold)

    def activation(self, pre_acts: torch.Tensor) -> torch.Tensor:
        return _BiJumpReLU.apply(
            pre_acts,
            self.pos_threshold,
            self.neg_threshold,
            self.bandwidth,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre_acts = self.pre_activations(x)
        mask = (pre_acts > self.pos_threshold) | (pre_acts < -self.neg_threshold)
        return mask * pre_acts

    def sparsity_measure(self, output: SAEOutput) -> torch.Tensor:
        return torch.sum(
            _BiJumpStep.apply(
                output.pre_activations,
                self.pos_threshold,
                self.neg_threshold,
                self.bandwidth,
            ),
            dim=-1,
        )

    def threshold_statistics(self) -> dict:
        pos = self.pos_threshold.detach().cpu()
        neg = self.neg_threshold.detach().cpu()
        return {
            "pos_threshold": {
                "mean": float(pos.mean()),
                "min": float(pos.min()),
                "max": float(pos.max()),
            },
            "neg_threshold": {
                "mean": float(-neg.mean()),
                "min": float(-neg.max()),
                "max": float(-neg.min()),
            },
        }


def build_sae(
    config: Mapping,
    *,
    input_dim: int,
    dictionary_width: int,
    s_inv_half: torch.Tensor,
) -> BaseSAE:
    architecture = str(config["architecture"]).lower()
    common = dict(
        input_dim=input_dim,
        dictionary_width=dictionary_width,
        s_inv_half=s_inv_half,
        use_pre_encoder_bias=bool(config.get("use_pre_encoder_bias", True)),
    )

    if architecture == "relu":
        return ReLUSAE(**common)
    if architecture == "topk":
        return TopKSAE(**common, k=int(config.get("k", 50)))
    if architecture == "abstopk":
        return AbsTopKSAE(**common, k=int(config.get("k", 50)))
    if architecture == "jumprelu":
        return JumpReLUSAE(
            **common,
            threshold_init=float(config.get("threshold_init", 0.5)),
            bandwidth=float(config.get("bandwidth", 2.0)),
        )
    if architecture == "bijumprelu":
        return BiJumpReLUSAE(
            **common,
            threshold_init=float(config.get("threshold_init", 0.5)),
            bandwidth=float(config.get("bandwidth", 2.0)),
        )
    raise ValueError(f"Unsupported SAE architecture: {architecture}")
