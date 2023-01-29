import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import typing
import warnings

class MultiHeadAttention(nn.Module):
    r"""MultiHead Attention layer.
    Defines the MultiHead Attention operation as described in
    [Attention Is All You Need](https://arxiv.org/abs/1706.03762) which takes
    in the tensors `query`, `key`, and `value`, and returns the dot-product attention
    between them:
    >>> mha = MultiHeadAttention(head_size=128, num_heads=12)
    >>> query = np.random.rand(3, 5, 4) # (batch_size, query_elements, query_depth)
    >>> key = np.random.rand(3, 6, 5) # (batch_size, key_elements, key_depth)
    >>> value = np.random.rand(3, 6, 6) # (batch_size, key_elements, value_depth)
    >>> attention = mha([query, key, value]) # (batch_size, query_elements, value_depth)
    >>> attention.shape
    TensorShape([3, 5, 6])
    If `value` is not given then internally `value = key` will be used:
    >>> mha = MultiHeadAttention(head_size=128, num_heads=12)
    >>> query = np.random.rand(3, 5, 5) # (batch_size, query_elements, query_depth)
    >>> key = np.random.rand(3, 6, 10) # (batch_size, key_elements, key_depth)
    >>> attention = mha([query, key]) # (batch_size, query_elements, key_depth)
    >>> attention.shape
    TensorShape([3, 5, 10])
    Args:
        head_size: int, dimensionality of the `query`, `key` and `value` tensors
            after the linear transformation.
        num_heads: int, number of attention heads.
        output_size: int, dimensionality of the output space, if `None` then the
            input dimension of `value` or `key` will be used,
            default `None`.
        dropout: float, `rate` parameter for the dropout layer that is
            applied to attention after softmax,
        default `0`.
        use_projection_bias: bool, whether to use a bias term after the linear
            output projection.
        return_attn_coef: bool, if `True`, return the attention coefficients as
            an additional output argument.
        kernel_initializer: initializer, initializer for the kernel weights.
        kernel_regularizer: regularizer, regularizer for the kernel weights.
        kernel_constraint: constraint, constraint for the kernel weights.
        bias_initializer: initializer, initializer for the bias weights.
        bias_regularizer: regularizer, regularizer for the bias weights.
        bias_constraint: constraint, constraint for the bias weights.
    Call Args:
        inputs:  List of `[query, key, value]` where
            * `query`: Tensor of shape `(..., query_elements, query_depth)`
            * `key`: `Tensor of shape '(..., key_elements, key_depth)`
            * `value`: Tensor of shape `(..., key_elements, value_depth)`, optional, if not given `key` will be used.
        mask: a binary Tensor of shape `[batch_size?, num_heads?, query_elements, key_elements]`
        which specifies which query elements can attendo to which key elements,
        `1` indicates attention and `0` indicates no attention.
    Output shape:
        * `(..., query_elements, output_size)` if `output_size` is given, else
        * `(..., query_elements, value_depth)` if `value` is given, else
        * `(..., query_elements, key_depth)`
    """

    def __init__(
        self,
        input_channels,
        head_size: int,
        num_heads: int,
        output_size: int = None,
        dropout: float = 0.0,
        use_projection_bias: bool = True,
        return_attn_coef: bool = False,
    ):
        super(MultiHeadAttention, self).__init__()

        if output_size is not None and output_size < 1:
            raise ValueError("output_size must be a positive number")

        self.head_size = head_size
        self.num_heads = num_heads
        self.output_size = output_size
        self.use_projection_bias = use_projection_bias
        self.return_attn_coef = return_attn_coef

        self.dropout = nn.Dropout(dropout)
        self._droput_rate = dropout

        num_query_features = input_channels[0]
        num_key_features = input_channels[1]
        num_value_features = (
            input_channels[2] if len(input_channels) > 2 else num_key_features
        )
        output_size = (
            self.output_size if self.output_size is not None else num_value_features
        )

        self.query_kernel = nn.Parameter(nn.init.xavier_uniform_(torch.zeros([self.num_heads, num_query_features, self.head_size])))
        self.key_kernel = nn.Parameter(nn.init.xavier_uniform_(torch.zeros([self.num_heads, num_key_features, self.head_size])))
        self.value_kernel = nn.Parameter(nn.init.xavier_uniform_(torch.zeros([self.num_heads, num_key_features, self.head_size])))
        self.projection_kernel = nn.Parameter(nn.init.xavier_uniform_(torch.zeros([self.num_heads, self.head_size, output_size])))


        if self.use_projection_bias:
            self.projection_bias = nn.Parameter(torch.zeros([output_size]))
        else:
            self.projection_bias = None

    def forward(self, inputs, mask=None):

        # einsum nomenclature
        # ------------------------
        # N = query elements
        # M = key/value elements
        # H = heads
        # I = input features
        # O = output features

        query = inputs[0]
        key = inputs[1]
        value = inputs[2] if len(inputs) > 2 else key

        # verify shapes
        if key.shape[-2] != value.shape[-2]:
            raise ValueError(
                "the number of elements in 'key' must be equal to the same as the number of elements in 'value'"
            )

        if mask is not None:
            if len(mask.shape) < 2:
                raise ValueError("'mask' must have atleast 2 dimensions")
            if query.shape[-2] != mask.shape[-2]:
                raise ValueError(
                    "mask's second to last dimension must be equal to the number of elements in 'query'"
                )
            if key.shape[-2] != mask.shape[-1]:
                raise ValueError(
                    "mask's last dimension must be equal to the number of elements in 'key'"
                )

        # Linear transformations
        query = torch.einsum("...NI , HIO -> ...NHO", query, self.query_kernel)
        key = torch.einsum("...MI , HIO -> ...MHO", key, self.key_kernel)
        value = torch.einsum("...MI , HIO -> ...MHO", value, self.value_kernel)

        # Scale dot-product, doing the division to either query or key
        # instead of their product saves some computation
        depth = torch.tensor(self.head_size, dtype=query.dtype).detach()
        query /= torch.sqrt(depth)

        # Calculate dot product attention
        logits = torch.einsum("...NHO,...MHO->...HNM", query, key)

        # apply mask
        if mask is not None:
            mask = mask.to(torch.float32)

            # possibly expand on the head dimension so broadcasting works
            if len(mask.shape) != len(logits.shape):
                mask = torch.unsqueeze(mask, -3)

            logits += -10e9 * (1.0 - mask)

        attn_coef = F.softmax(logits, dim=-1)

        # attention dropout
        attn_coef_dropout = self.dropout(attn_coef)

        # attention * value
        multihead_output = torch.einsum("...HNM,...MHI->...NHI", attn_coef_dropout, value)

        # Run the outputs through another linear projection layer. Recombining heads
        # is automatically done.
        output = torch.einsum(
            "...NHI,HIO->...NO", multihead_output, self.projection_kernel
        )

        if self.projection_bias is not None:
            output += self.projection_bias

        if self.return_attn_coef:
            return output, attn_coef
        else:
            return output


