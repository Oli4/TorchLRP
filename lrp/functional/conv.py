import torch
import torch.nn.functional as F
from torch.autograd import Function

from .utils import identity_fn, gamma_fn, add_epsilon_fn, normalize
from .. import trace


def _forward_rho(
    rho, incr, ctx, input, weight, bias, stride, padding, dilation, groups
):
    ctx.save_for_backward(input, weight, bias)
    ctx.rho = rho
    ctx.incr = incr
    # Ensure stride and padding are tuples
    ctx.stride = stride if isinstance(stride, tuple) else (stride, stride)
    ctx.padding = padding if isinstance(padding, tuple) else (padding, padding)
    ctx.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
    ctx.groups = groups
    ctx.input_size = input.size()

    Z = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
    return Z


def _calculate_output_padding(
    input_size, output_size, kernel_size, stride, padding, dilation
):
    """Helper function to calculate output padding safely"""

    # Ensure all inputs are pairs of integers
    def to_pair(x):
        return x if isinstance(x, tuple) else (x, x)

    stride = to_pair(stride)
    padding = to_pair(padding)
    dilation = to_pair(dilation)

    h_out_pad = max(
        0,
        input_size[2]
        - (
            (output_size[2] - 1) * stride[0]
            - 2 * padding[0]
            + dilation[0] * (kernel_size[2] - 1)
            + 1
        ),
    )
    w_out_pad = max(
        0,
        input_size[3]
        - (
            (output_size[3] - 1) * stride[1]
            - 2 * padding[1]
            + dilation[1] * (kernel_size[3] - 1)
            + 1
        ),
    )
    return (h_out_pad, w_out_pad)


def _backward_rho(ctx, relevance_output):
    input, weight, bias = ctx.saved_tensors

    weight, bias = ctx.rho(weight, bias)
    Z = ctx.incr(
        F.conv2d(input, weight, bias, ctx.stride, ctx.padding, ctx.dilation, ctx.groups)
    )

    relevance_output = relevance_output / Z

    output_padding = _calculate_output_padding(
        ctx.input_size, Z.size(), weight.size(), ctx.stride, ctx.padding, ctx.dilation
    )

    relevance_input = F.conv_transpose2d(
        relevance_output,
        weight,
        None,
        stride=ctx.stride,
        padding=ctx.padding,
        output_padding=output_padding,
        dilation=ctx.dilation,
        groups=ctx.groups,
    )
    relevance_input = relevance_input * input

    trace.do_trace(relevance_input)
    return (
        relevance_input,
        None,
        None,
        None,
        None,
        None,
        None,
    )


class Conv2DEpsilon(Function):
    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        **kwargs
    ):
        return _forward_rho(
            identity_fn,
            add_epsilon_fn(1e-1),
            ctx,
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
        )

    @staticmethod
    def backward(ctx, relevance_output):
        return _backward_rho(ctx, relevance_output)


class Conv2DGamma(Function):
    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        **kwargs
    ):
        return _forward_rho(
            gamma_fn(0.1),
            add_epsilon_fn(1e-10),
            ctx,
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
        )

    @staticmethod
    def backward(ctx, relevance_output):
        return _backward_rho(ctx, relevance_output)


class Conv2DGammaEpsilon(Function):
    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        **kwargs
    ):
        return _forward_rho(
            gamma_fn(0.1),
            add_epsilon_fn(1e-1),
            ctx,
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
        )

    @staticmethod
    def backward(ctx, relevance_output):
        return _backward_rho(ctx, relevance_output)


def _conv_alpha_beta_forward(
    ctx, input, weight, bias, stride, padding, dilation, groups, **kwargs
):
    Z = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
    ctx.save_for_backward(input, weight, Z, bias)
    # Save additional context and ensure values are tuples
    ctx.stride = stride if isinstance(stride, tuple) else (stride, stride)
    ctx.padding = padding if isinstance(padding, tuple) else (padding, padding)
    ctx.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
    ctx.groups = groups
    ctx.input_size = input.size()
    return Z


def _conv_alpha_beta_backward(alpha, beta, ctx, relevance_output):
    input, weights, Z, bias = ctx.saved_tensors
    sel = weights > 0
    zeros = torch.zeros_like(weights)

    weights_pos = torch.where(sel, weights, zeros)
    weights_neg = torch.where(~sel, weights, zeros)

    input_pos = torch.where(input > 0, input, torch.zeros_like(input))
    input_neg = torch.where(input <= 0, input, torch.zeros_like(input))

    def f(X1, X2, W1, W2):
        Z1 = F.conv2d(
            X1,
            W1,
            bias=None,
            stride=ctx.stride,
            padding=ctx.padding,
            dilation=ctx.dilation,
            groups=ctx.groups,
        )
        Z2 = F.conv2d(
            X2,
            W2,
            bias=None,
            stride=ctx.stride,
            padding=ctx.padding,
            dilation=ctx.dilation,
            groups=ctx.groups,
        )
        Z = Z1 + Z2

        rel_out = relevance_output / (Z + (Z == 0).float() * 1e-6)

        output_padding = _calculate_output_padding(
            ctx.input_size, Z.size(), W1.size(), ctx.stride, ctx.padding, ctx.dilation
        )

        t1 = F.conv_transpose2d(
            rel_out,
            W1,
            bias=None,
            stride=ctx.stride,
            padding=ctx.padding,
            output_padding=output_padding,
            dilation=ctx.dilation,
            groups=ctx.groups,
        )
        t2 = F.conv_transpose2d(
            rel_out,
            W2,
            bias=None,
            stride=ctx.stride,
            padding=ctx.padding,
            output_padding=output_padding,
            dilation=ctx.dilation,
            groups=ctx.groups,
        )

        r1 = t1 * X1
        r2 = t2 * X2

        return r1 + r2

    pos_rel = f(input_pos, input_neg, weights_pos, weights_neg)
    neg_rel = f(input_neg, input_pos, weights_pos, weights_neg)
    relevance_input = pos_rel * alpha - neg_rel * beta

    trace.do_trace(relevance_input)
    return relevance_input, None, None, None, None, None, None


class Conv2DAlpha1Beta0(Function):
    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        **kwargs
    ):
        return _conv_alpha_beta_forward(
            ctx, input, weight, bias, stride, padding, dilation, groups, **kwargs
        )

    @staticmethod
    def backward(ctx, relevance_output):
        return _conv_alpha_beta_backward(1.0, 0.0, ctx, relevance_output)


class Conv2DAlpha2Beta1(Function):
    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        **kwargs
    ):
        return _conv_alpha_beta_forward(
            ctx, input, weight, bias, stride, padding, dilation, groups, **kwargs
        )

    @staticmethod
    def backward(ctx, relevance_output):
        return _conv_alpha_beta_backward(2.0, 1.0, ctx, relevance_output)


def _pattern_forward(
    attribution, ctx, input, weight, bias, stride, padding, dilation, groups, pattern
):
    Z = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
    ctx.save_for_backward(input, weight, pattern)

    ctx.stride = stride if isinstance(stride, tuple) else (stride, stride)
    ctx.padding = padding if isinstance(padding, tuple) else (padding, padding)
    ctx.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
    ctx.groups = groups
    ctx.input_size = input.size()
    ctx.attribution = attribution
    return Z


def _pattern_backward(ctx, relevance_output):
    input, weight, P = ctx.saved_tensors

    if ctx.attribution:
        P = P * weight  # PatternAttribution

    output_padding = _calculate_output_padding(
        ctx.input_size,
        relevance_output.size(),
        P.size(),
        ctx.stride,
        ctx.padding,
        ctx.dilation,
    )

    relevance_input = F.conv_transpose2d(
        relevance_output,
        P,
        stride=ctx.stride,
        padding=ctx.padding,
        output_padding=output_padding,
        dilation=ctx.dilation,
        groups=ctx.groups,
    )

    trace.do_trace(relevance_input)
    return relevance_input, None, None, None, None, None, None, None


class Conv2DPatternAttribution(Function):
    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        pattern=None,
    ):
        return _pattern_forward(
            True, ctx, input, weight, bias, stride, padding, dilation, groups, pattern
        )

    @staticmethod
    def backward(ctx, relevance_output):
        return _pattern_backward(ctx, relevance_output)


class Conv2DPatternNet(Function):
    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        pattern=None,
    ):
        return _pattern_forward(
            False, ctx, input, weight, bias, stride, padding, dilation, groups, pattern
        )

    @staticmethod
    def backward(ctx, relevance_output):
        return _pattern_backward(ctx, relevance_output)


conv2d = {
    "gradient": F.conv2d,
    "epsilon": Conv2DEpsilon.apply,
    "gamma": Conv2DGamma.apply,
    "gamma+epsilon": Conv2DGammaEpsilon.apply,
    "alpha1beta0": Conv2DAlpha1Beta0.apply,
    "alpha2beta1": Conv2DAlpha2Beta1.apply,
    "patternattribution": Conv2DPatternAttribution.apply,
    "patternnet": Conv2DPatternNet.apply,
}
