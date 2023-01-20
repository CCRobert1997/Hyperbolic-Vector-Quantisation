import numpy as np
import torch
from torch import nn
from torch.autograd import Function, Variable
import torch.nn.functional as F
import hyptorch.nn as hypnn
from hyptorch import pmath
import math

import matplotlib.pyplot as plt

class NearestEmbedFunc(Function):
    """
    Input:
    ------
    x - (batch_size, emb_dim, *)
        Last dimensions may be arbitrary
    emb - (emb_dim, num_emb)
    """
    @staticmethod
    def forward(ctx, input, emb, curvature):
        if input.size(1) != emb.size(0):
            raise RuntimeError('invalid argument: input.size(1) ({}) must be equal to emb.size(0) ({})'.
                               format(input.size(1), emb.size(0)))

        # save sizes for backward
        ctx.batch_size = input.size(0)
        ctx.num_latents = int(np.prod(np.array(input.size()[2:])))
        ctx.emb_dim = emb.size(0)
        ctx.num_emb = emb.size(1)
        ctx.input_type = type(input)
        ctx.dims = list(range(len(input.size())))

        # expand to be broadcast-able
        x_expanded = input.unsqueeze(-1)
        num_arbitrary_dims = len(ctx.dims) - 2
        if num_arbitrary_dims:
            emb_expanded = emb.view(
                emb.shape[0], *([1] * num_arbitrary_dims), emb.shape[1])
        else:
            emb_expanded = emb

        # find nearest neighbors
        # use distance on hyperbolic space
        #print(x_expanded.shape)
        #print(emb_expanded.shape)
        #dist = artanh(sqrt_c * pmath._mobius_add(x_expanded, torch.negative(emb_expanded), 1.0).norm(dim=1, p=2))
        # print("calculate dist to codebook")
        # print(x_expanded.shape)

        dist = pmath._dist_for_vqvae(x_expanded, emb_expanded, curvature)




        # print(torch.norm(emb_expanded, 2, 1)[0][0])
        # print(dist)
        # #dist = torch.norm(x_expanded - emb_expanded, 2, 1)
        # print(dist.shape)
        _, argmin = dist.min(-1)
        # print(argmin.shape)
        shifted_shape = [input.shape[0], *
                         list(input.shape[2:]), input.shape[1]]
        result = emb.t().index_select(0, argmin.view(-1)
                                      ).view(shifted_shape).permute(0, ctx.dims[-1], *ctx.dims[1:-1])

        ctx.save_for_backward(argmin)
        return result.contiguous(), argmin

    @staticmethod
    def backward(ctx, grad_output, argmin=None):
        grad_input = grad_emb = None
        if ctx.needs_input_grad[0]:
            # print(grad_output.shape)
            grad_input = grad_output
            #print(grad_output.shape)
        if ctx.needs_input_grad[1]:

            argmin, = ctx.saved_variables
            #print(argmin.shape)
            
            latent_indices = torch.arange(ctx.num_emb).type_as(argmin)

            idx_choices = (argmin.view(-1, 1) ==
                           latent_indices.view(1, -1)).type_as(grad_output.data)

            n_idx_choice = idx_choices.sum(0)
            # print(idx_choices)
            # print(n_idx_choice)
            #print(n_idx_choice)
            n_idx_choice[n_idx_choice == 0] = 1
            # print(argmin.shape)
            # print(latent_indices.shape)
            # print(idx_choices.shape)
            # print(n_idx_choice.shape)
            idx_avg_choices = idx_choices / n_idx_choice
            # print(idx_avg_choices)
            # print(idx_avg_choices.shape)
            # print(grad_output.shape)
            #####################################################################
            #idx_avg_choices = torch.ones_like(idx_avg_choices)

            grad_output = grad_output.permute(0, *ctx.dims[2:], 1).contiguous()
            # print(grad_output.shape)
            grad_output = grad_output.view(
                ctx.batch_size * ctx.num_latents, ctx.emb_dim)
            # print(grad_output.shape)
            # print((grad_output.data.view(-1, ctx.emb_dim, 1) * idx_avg_choices.view(-1, 1, ctx.num_emb)).shape)
            grad_emb = torch.sum(grad_output.data.view(-1, ctx.emb_dim, 1) *
                                 idx_avg_choices.view(-1, 1, ctx.num_emb), 0)
            # print(grad_emb.shape)

        return grad_input, grad_emb, None, None


def nearest_embed(x, emb, c):
    return NearestEmbedFunc().apply(x, emb, c)


class NearestEmbed(nn.Module):
    def __init__(self, num_embeddings, embeddings_dim, curvature):
        super(NearestEmbed, self).__init__()
        self.curvature = curvature
        self.weight = nn.Parameter(torch.rand(embeddings_dim, num_embeddings) * 1 / math.sqrt(embeddings_dim * (curvature + 1e-7)))
        #print(self.weight)

    def forward(self, x, weight_sg=False):
        """Input:
        ---------
        x - (batch_size, emb_size, *)
        """
        #print("&&&&&&&&&&&&&&&&&&&&")
        #print(self.weight)
        return nearest_embed(x, self.weight.detach() if weight_sg else self.weight, self.curvature)


# adapted from https://github.com/rosinality/vq-vae-2-pytorch/blob/master/vqvae.py#L25
# that adapted from https://github.com/deepmind/sonnet


class NearestEmbedEMA(nn.Module):
    def __init__(self, n_emb, emb_dim, decay=0.99, eps=1e-5):
        super(NearestEmbedEMA, self).__init__()
        self.decay = decay
        self.eps = eps
        self.embeddings_dim = emb_dim
        self.n_emb = n_emb
        self.emb_dim = emb_dim
        embed = torch.rand(emb_dim, n_emb)
        self.register_buffer('weight', embed)
        self.register_buffer('cluster_size', torch.zeros(n_emb))
        self.register_buffer('embed_avg', embed.clone())

    def forward(self, x):
        """Input:
        ---------
        x - (batch_size, emb_size, *)
        """

        dims = list(range(len(x.size())))
        x_expanded = x.unsqueeze(-1)
        num_arbitrary_dims = len(dims) - 2
        if num_arbitrary_dims:
            emb_expanded = self.weight.view(
                self.emb_dim, *([1] * num_arbitrary_dims), self.n_emb)
        else:
            emb_expanded = self.weight

        # find nearest neighbors
        dist = torch.norm(x_expanded - emb_expanded, 2, 1)
        _, argmin = dist.min(-1)
        shifted_shape = [x.shape[0], *list(x.shape[2:]), x.shape[1]]
        result = self.weight.t().index_select(
            0, argmin.view(-1)).view(shifted_shape).permute(0, dims[-1], *dims[1:-1])

        if self.training:
            latent_indices = torch.arange(self.n_emb).type_as(argmin)
            emb_onehot = (argmin.view(-1, 1) ==
                          latent_indices.view(1, -1)).type_as(x.data)
            n_idx_choice = emb_onehot.sum(0)
            n_idx_choice[n_idx_choice == 0] = 1
            flatten = x.permute(
                1, 0, *dims[-2:]).contiguous().view(x.shape[1], -1)

            self.cluster_size.data.mul_(self.decay).add_(
                1 - self.decay, n_idx_choice
            )
            embed_sum = flatten @ emb_onehot
            self.embed_avg.data.mul_(self.decay).add_(
                1 - self.decay, embed_sum)

            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) /
                (n + self.n_emb * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.weight.data.copy_(embed_normalized)

        return result, argmin
