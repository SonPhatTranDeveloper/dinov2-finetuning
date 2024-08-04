import torch
from torch import nn as nn
from einops import rearrange
from dinov2.layers.attention import MemEffAttention, Attention
import numpy as np


class SinkhornDistanceFast(nn.Module):
    def __init__(self, eps=1, max_iter=1):
        """
        Initialize the Sinkhorn Fast and Stable AutoDiff algorithm
        Adapted from the paper: https://arxiv.org/pdf/1607.05816.pdf
        :param eps: the epsilon in the Sink
        :param max_iter: the number of Sinkhorn iteration
        """
        # Initialize
        super(SinkhornDistanceFast, self).__init__()

        # Cache the parameters
        self.eps = eps
        self.max_iter = max_iter

    def forward(self, attention_score):
        """
        Perform the Sinkhorn algorithm on the attention score matrix
        :param attention_score: attention_score matrix of shape (batch_size, number_of_query, number_of_value)
        :return: the processed attention_score matrix after the Sinkhorn algorithm
        """
        # Create the cost matrix, which is the negative of the attention score
        cost = - attention_score

        # Get the dimensions
        x_points = cost.shape[-2]
        y_points = cost.shape[-1]
        batch_size = cost.shape[0]

        # Create two marginals mu and nu with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False, device=cost.device).fill_(1.0 / x_points).squeeze()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False, device=cost.device).fill_(1.0 / y_points).squeeze()

        # Create two vectors u and v with same dimension as mu and nu
        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)

        # Threshold to stop Sinkhorn
        threshold = 1e-12
        err = None

        # Perform Sinkhorn iterations
        for i in range(self.max_iter):
            if i % 2 == 0:
                u1 = u  # useful to check the update
                u = self.eps * (torch.log(mu) - torch.logsumexp(self.modified_cost(cost, u, v), dim=-1)) + u
                err = (u - u1).abs().sum(-1).mean()
            else:
                v = self.eps * (torch.log(nu) - torch.logsumexp(self.modified_cost(cost, u, v).transpose(-2, -1),
                                                                dim=-1)) + v
                v = v.detach().requires_grad_(False)
                v[v > 9 * 1e8] = 0.0
                v = v.detach().requires_grad_(True)

            if err.item() < threshold:
                break

        # Calculate the result pi
        pi = torch.exp(self.modified_cost(cost, u, v))
        return pi

    def modified_cost(self, cost_matrix, u, v):
        """
        Calculate the modified cost for logarithmic updates
        :param cost_matrix:
        :param u:
        :param v:
        :return:
        """
        "Modified cost for logarithmic updates"
        return (-cost_matrix + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps


class ScaledProductAttentionSinkhorn(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias: bool = False,
                 proj_bias: bool = True,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 max_iter=1,
                 eps=1):
        """
        Initialize the scaled product attention sinkhorn normalization block
        :param dim: the number of features of the input
        :param num_heads: the number of heads
        :param attn_drop: dropout rate of attention head
        :param proj_drop: the dropout rate
        :param max_iter: Sinkhorn iteration
        :param eps: parameter of Sinkhorn distance
        """
        super(ScaledProductAttentionSinkhorn, self).__init__()

        # Cache the variables
        self.dim = dim
        self.num_heads = num_heads
        self.d_head = dim // num_heads
        self.p_dropout = proj_drop
        self.attn_dropout = attn_drop
        self.max_iter = max_iter
        self.eps = eps

        # Calculate the inner dimension of multi-headed attention
        self.inner_dim = self.num_heads * self.d_head

        # Create the Sinkhorn Distance block
        self.sinkhorn = SinkhornDistanceFast(max_iter=self.max_iter, eps=self.eps)

        # Mapping the input to (query, key, value)
        # no bias
        self.qkv = nn.Linear(in_features=self.dim, out_features=self.dim * 3, bias=qkv_bias)

        # Create attention dropout layer
        self.attn_drop = nn.Dropout(attn_drop)

        # Output projection if any
        # else it is just an identity layer
        self.proj = nn.Linear(self.dim, self.dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(p=self.p_dropout)

    def forward(self, inputs):
        """
        Perform the forward operation of scaled product sinkhorn normalization
        :param inputs: tensor of shape (batch_size, number_of_batches, d_model)
        :return: outputs: tensor of shape (batch_size, number_of_batches, d_model)
        """
        # Get the shape of input (batch_size, number_of_patches, d_model)
        B, N, C = inputs.shape

        # Calculate q, k, and v values
        qkv = self.qkv(inputs).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Calculate the attention scores and weights
        # attention_scores has shape (batch_size, n_heads, number_of_patches, number_of_patches)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_head)

        # Reshape attention score
        # attention_score has shape (batch_size x number_of_heads, number_of_patches, number_of_patches)
        attention_score_shape = attention_scores.shape
        attention_scores = attention_scores.view(-1, attention_score_shape[2], attention_score_shape[3])

        # Perform Sinkhorn iteration to calculate attention weights
        attention_weights = self.sinkhorn(attention_scores)
        attention_weights = attention_weights * attention_weights.shape[-1]
        attention_weights = attention_weights.view(attention_score_shape)

        # Go through attention dropout
        attention_weights = self.attn_drop(attention_weights)

        # Calculate the output
        # output has size (batch_size, number_of_heads, number_of_patches, dim_head)
        outputs = torch.matmul(attention_weights, v)

        # Reshape outputs
        # outputs has size (batch_size, number_of_batches, (number_of_head x dim_head))
        outputs = rearrange(outputs, 'b h n d -> b n (h d)')

        # Map to output
        # outputs has size (batch_size, number_of_batches, d_model)
        outputs = self.proj(outputs)
        outputs = self.proj_drop(outputs)

        # Return outputs and attention weights
        return outputs


class WeightedCombinationAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias: bool = False,
                 proj_bias: bool = True,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 max_iter=3,
                 eps=1,
                 sinkhorn_weight=0.0):
        """
        Initialize the scaled product attention sinkhorn normalization block
        :param dim: the number of features of the input
        :param num_heads: the number of heads
        :param d_head: the dimension of each heads
        :param attn_drop: dropout rate of attention head
        :param proj_drop: the dropout rate
        :param max_iter: Sinkhorn iteration
        :param eps: parameter of Sinkhorn distance
        """

        # Call the superclass constructor
        super(WeightedCombinationAttention, self).__init__()

        # Create softmax attention and sinkhorn attention
        self.softmax_attn = Attention(dim,
                                      num_heads,
                                      qkv_bias,
                                      proj_bias,
                                      attn_drop,
                                      proj_drop
                                      )

        self.sinkhorn_attn = ScaledProductAttentionSinkhorn(dim,
                                                            num_heads,
                                                            qkv_bias,
                                                            proj_bias,
                                                            attn_drop,
                                                            proj_drop,
                                                            max_iter,
                                                            eps
                                                            )

        # Save the weight
        self.sinkhorn_weight = sinkhorn_weight

    def forward(self, inputs):
        """
        Perform the forward operation of scaled product sinkhorn normalization
        :param inputs: tensor of shape (batch_size, number_of_batches, d_model)
        :return: outputs: tensor of shape (batch_size, number_of_batches, d_model)
        """
        # Calculate the softmax result
        outputs_softmax = self.softmax_attn(inputs)
        outputs_sinkhorn = self.sinkhorn_attn(inputs)

        # Calculate the weight combination of them
        outputs = (1 - self.sinkhorn_weight) * outputs_softmax + self.sinkhorn_weight * outputs_sinkhorn
        return outputs


class WeightedLearnableAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias: bool = False,
                 proj_bias: bool = True,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 max_iter=3,
                 eps=1):
        """
        Initialize the scaled product attention sinkhorn normalization block
        :param dim: the number of features of the input
        :param num_heads: the number of heads
        :param d_head: the dimension of each heads
        :param attn_drop: dropout rate of attention head
        :param proj_drop: the dropout rate
        :param max_iter: Sinkhorn iteration
        :param eps: parameter of Sinkhorn distance
        """

        # Call the superclass constructor
        super(WeightedLearnableAttention, self).__init__()

        # Create weight
        self.weight_logit = nn.Parameter(torch.logit(torch.tensor([0.5])), requires_grad=True)

        # Create softmax attention and sinkhorn attention
        self.softmax_attn = MemEffAttention(dim,
                                            num_heads,
                                            qkv_bias,
                                            proj_bias,
                                            attn_drop,
                                            proj_drop
                                            )

        self.sinkhorn_attn = ScaledProductAttentionSinkhorn(dim,
                                                            num_heads,
                                                            qkv_bias,
                                                            proj_bias,
                                                            attn_drop,
                                                            proj_drop,
                                                            max_iter,
                                                            eps
                                                            )

    def forward(self, inputs):
        """
        Perform the forward operation of scaled product sinkhorn normalization
        :param inputs: tensor of shape (batch_size, number_of_batches, d_model)
        :return: outputs: tensor of shape (batch_size, number_of_batches, d_model)
        """
        # Calculate the softmax result
        outputs_softmax = self.softmax_attn(inputs)
        outputs_sinkhorn = self.sinkhorn_attn(inputs)

        # Calculate the weight
        weight = self.weight_logit.sigmoid()

        # Calculate the weight combination of them
        outputs = weight * outputs_softmax + (1 - weight) * outputs_sinkhorn
        return outputs
