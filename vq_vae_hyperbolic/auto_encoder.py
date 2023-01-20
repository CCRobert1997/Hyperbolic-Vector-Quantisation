from __future__ import print_function
import abc

import numpy as np
import logging
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
import torch.distributions as dist

from .nearest_embed import NearestEmbed, NearestEmbedEMA

import hyptorch.nn as hypnn
from hyptorch import pmath
import hyptorch.distributions as hypdist
import hyptorch.manifolds as manifolds
import math
from numpy import prod


class AbstractAutoEncoder(nn.Module):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def encode(self, x):
        return

    @abc.abstractmethod
    def decode(self, z):
        return

    @abc.abstractmethod
    def forward(self, x):
        """model return (reconstructed_x, *)"""
        return

    @abc.abstractmethod
    def sample(self, size):
        """sample new images from model"""
        return

    @abc.abstractmethod
    def loss_function(self, **kwargs):
        """accepts (original images, *) where * is the same as returned from forward()"""
        return

    @abc.abstractmethod
    def latest_losses(self):
        """returns the latest losses in a dictionary. Useful for logging."""
        return

#
# class VAE(nn.Module):
#     """Variational AutoEncoder for MNIST
#        Taken from pytorch/examples: https://github.com/pytorch/examples/tree/master/vae"""
#
#     def __init__(self, kl_coef=1, **kwargs):
#         super(VAE, self).__init__()
#
#         self.fc1 = nn.Linear(784, 400)
#         self.fc21 = nn.Linear(400, 20)
#         self.fc22 = nn.Linear(400, 20)
#         self.fc3 = nn.Linear(20, 400)
#         self.fc4 = nn.Linear(400, 784)
#
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#         self.kl_coef = kl_coef
#         self.bce = 0
#         self.kl = 0
#
#     def encode(self, x):
#         h1 = self.relu(self.fc1(x))
#         return self.fc21(h1), self.fc22(h1)
#
#     def reparameterize(self, mu, logvar):
#         if self.training:
#             std = logvar.mul(0.5).exp_()
#             eps = std.new(std.size()).normal_()
#             return eps.mul(std).add_(mu)
#         else:
#             return mu
#
#     def decode(self, z):
#         h3 = self.relu(self.fc3(z))
#         return torch.tanh(self.fc4(h3))
#
#     def forward(self, x):
#         mu, logvar = self.encode(x.view(-1, 784))
#         z = self.reparameterize(mu, logvar)
#         return self.decode(z), mu, logvar
#
#     def sample(self, size):
#         sample = torch.randn(size, 20)
#         if self.cuda():
#             sample = sample.cuda()
#         sample = self.decode(sample).cpu()
#         return sample
#
#     def loss_function(self, x, recon_x, mu, logvar):
#         #self.bce = F.binary_cross_entropy(
#         #    recon_x, x.view(-1, 784), size_average=False)
#         self.recons_loss = F.mse_loss(
#             recon_x, x.view(-1, 784))
#         batch_size = x.size(0)
#
#         # see Appendix B from VAE paper:
#         # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
#         # https://arxiv.org/abs/1312.6114
#         # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#         self.kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#
#         return self.recons_loss + self.kl_coef*self.kl
#
#     def latest_losses(self):
#         return {'bce': self.bce, 'kl': self.kl}


class VQ_VAE(nn.Module):
    """Vector Quantized AutoEncoder for mnist"""

    def __init__(self, hidden=200, k=10, vq_coef=0.2, comit_coef=0.4, **kwargs):
        super(VQ_VAE, self).__init__()

        self.emb_size = k
        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, hidden)

        self.fc3 = nn.Linear(hidden, 400)
        self.fc4 = nn.Linear(400, 784)

        self.emb = NearestEmbed(k, self.emb_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.vq_coef = vq_coef
        self.comit_coef = comit_coef
        self.hidden = hidden
        self.ce_loss = 0
        self.vq_loss = 0
        self.commit_loss = 0

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.fc2(h1)
        return h2.view(-1, self.emb_size, int(self.hidden / self.emb_size))

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return torch.tanh(self.fc4(h3))

    def forward(self, x):
        z_e = self.encode(x.view(-1, 784))
        z_q, _ = self.emb(z_e, weight_sg=True).view(-1, self.hidden)
        emb, _ = self.emb(z_e.detach()).view(-1, self.hidden)
        return self.decode(z_q), z_e, emb

    def sample(self, size):
        sample = torch.randn(size, self.emb_size,
                             int(self.hidden / self.emb_size))
        if self.cuda():
            sample = sample.cuda()
        emb, _ = self.emb(sample)
        sample = self.decode(emb(sample).view(-1, self.hidden)).cpu()
        return sample

    def loss_function(self, x, recon_x, z_e, emb):
        self.ce_loss = F.binary_cross_entropy(recon_x, x.view(-1, 784))
        self.vq_loss = F.mse_loss(emb, z_e.detach())
        self.commit_loss = F.mse_loss(z_e, emb.detach())

        return self.ce_loss + self.vq_coef*self.vq_loss + self.comit_coef*self.commit_loss

    def latest_losses(self):
        return {'cross_entropy': self.ce_loss, 'vq': self.vq_loss, 'commitment': self.commit_loss}


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super(ResBlock, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        layers = [
            nn.ReLU(),
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels,
                      kernel_size=1, stride=1, padding=0)
        ]
        if bn:
            layers.insert(2, nn.BatchNorm2d(out_channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.convs(x)


class CVAE(AbstractAutoEncoder):
    def __init__(self, d, kl_coef=0.1, bn=True, num_channels=3, transforms_size=256, **kwargs):
        super(CVAE, self).__init__()

        # self.encoder = nn.Sequential(
        #     nn.Conv2d(num_channels, d // 2, kernel_size=4,
        #               stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(d // 2),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(d // 2, d, kernel_size=4,
        #               stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(d),
        #     nn.ReLU(inplace=True),
        #     ResBlock(d, d, bn=True),
        #     nn.BatchNorm2d(d),
        #     ResBlock(d, d, bn=True),
        # )

        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=bn),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=bn),
            nn.BatchNorm2d(d),
        )

        # self.decoder = nn.Sequential(
        #     ResBlock(d, d, bn=True),
        #     nn.BatchNorm2d(d),
        #     ResBlock(d, d, bn=True),
        #     nn.BatchNorm2d(d),
        #
        #     nn.ConvTranspose2d(d, d // 2, kernel_size=4,
        #                        stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(d//2),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(d // 2, 3, kernel_size=4,
        #                        stride=2, padding=1, bias=False),
        # )
        self.decoder = nn.Sequential(
            ResBlock(d, d),
            nn.BatchNorm2d(d),
            ResBlock(d, d),
            nn.ConvTranspose2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                d, num_channels, kernel_size=4, stride=2, padding=1),
        )
        #self.f = int(math.sqrt(num_channels * transforms_size ** 2 / 3 * 2 / d))
        self.f = 8
        self.transforms_size  = transforms_size
        self.d = d
        #self.fc11 = nn.Linear(d * int(self.transforms_size / 4) ** 2, d * int(self.transforms_size / 4) ** 2)
        self.fc11 = nn.Sequential(
            nn.Linear(d * int(self.transforms_size / 4) ** 2, d),
            nn.BatchNorm1d(d),
        )
        #self.fc11 = nn.Linear(int(num_channels * transforms_size ** 2 / 3 * 2), d)
        #self.fc11 = nn.Linear(num_channels * self.f ** 2, d)
        #self.fc12 = nn.Linear(d * int(self.transforms_size / 4) ** 2, d * int(self.transforms_size / 4) ** 2)
        self.fc12 = nn.Sequential(
            nn.Linear(d * int(self.transforms_size / 4) ** 2, d),
            nn.BatchNorm1d(d),
        )

        self.fc21 = nn.Sequential(
            nn.Linear(d, d * int(self.transforms_size / 4) ** 2),
            nn.BatchNorm1d(d * int(self.transforms_size / 4) ** 2),
        )
        #self.fc22 = nn.Linear(d, d * int(self.transforms_size / 4) ** 2)


        #self.fc12 = nn.Linear(int(num_channels * transforms_size ** 2 / 3 * 2), d)
        #self.fc12 = nn.Linear(num_channels * self.f ** 2, d)
        self.kl_coef = kl_coef
        self.kl_loss = 0
        self.mse = 0


    def encode(self, x):
        #print(x.shape)
        h1 = self.encoder(x)
        #h1 = torch.flatten(h1, start_dim=1)
        #print(h1.shape)
        #print(h1.shape[1])
        #print(self.d)
        h2 = h1.view(-1, self.d * int(self.transforms_size / 4) ** 2)
        #print(h2.shape)
        #return self.fc11(h1), self.fc12(h1)
        return self.fc11(h2), self.fc12(h2)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        #z = z.view(-1, self.d, self.f, self.f)
        z = self.fc21(z)
        z = z.view(-1, self.d, int(self.transforms_size / 4), int(self.transforms_size / 4))
        #print(z.shape)
        #print("---------------------------")
        h3 = self.decoder(z)
        return torch.tanh(h3)

    def forward(self, x):
        #print(x.shape)
        mu, logvar = self.encode(x)
        #print(mu.shape)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def sample(self, size):
        sample = torch.randn(size, self.d * self.f ** 2, requires_grad=False)
        if self.cuda():
            sample = sample.cuda()
        return self.decode(sample).cpu()

    def loss_function(self, x, recon_x, mu, logvar):
        #print(recon_x.shape)
        #print(x.shape)
        self.mse = F.mse_loss(recon_x, x)
        batch_size = x.size(0)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        self.kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Normalise by same number of elements as in reconstruction
        self.kl_loss /= batch_size * 3 * 1024

        # return mse
        return self.mse + self.kl_coef * self.kl_loss

    def latest_losses(self):
        return {'mse': self.mse, 'kl': self.kl_loss}






class VQ_CVAE(nn.Module):
    def __init__(self, d, k=10, c=0.001, bn=True, vq_coef=1, commit_coef=0.5, num_channels=3, **kwargs):
        #def __init__(self, d, k=10, c=0.001, bn=True, vq_coef=4, commit_coef=4, num_channels=3, **kwargs):
        super(VQ_CVAE, self).__init__()
        self.c = c + 1e-7
        self.d = d
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=bn),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=bn),
            nn.BatchNorm2d(d),
        )


        """
        self.tp = hypnn.ToPoincare(
            c=args.c, train_x=args.train_x, train_c=args.train_c, ball_dim=args.dim
        )
        """
        self.tp = hypnn.ToPoincare(
            c=self.c, train_x=False, train_c=False, ball_dim=d
        )

        self.fp = hypnn.FromPoincare(
            c=self.c, train_x=False, train_c=False, ball_dim=d
        )


        self.decoder = nn.Sequential(
            ResBlock(d, d),
            nn.BatchNorm2d(d),
            ResBlock(d, d),
            nn.ConvTranspose2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                d, num_channels, kernel_size=4, stride=2, padding=1),
        )
        self.emb = NearestEmbed(k, d, self.c)
        #print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        #print(self.emb.weight)
        self.vq_coef = vq_coef
        self.commit_coef = commit_coef
        self.mse = 0
        self.vq_loss = torch.zeros(1)
        self.commit_loss = 0
        for l in self.modules():
            if isinstance(l, nn.Linear) or isinstance(l, nn.Conv2d):
                l.weight.detach().normal_(0, 0.02)
                torch.fmod(l.weight, 0.04)
                nn.init.constant_(l.bias, 0)


        #self.encoder[-1].weight.detach().fill_(1 / 40)

        #self.encoder[-1].weight.detach().fill_(1 / math.sqrt(80 * d * self.c))

        self.encoder[-1].weight.detach().fill_(1 / (40 * math.sqrt(d * self.c)))

        #self.emb.weight.detach().normal_(0, 0.02)
        self.emb.weight.detach().normal_(0, 0.02 / math.sqrt(d * self.c))
        self.emb.weight = nn.Parameter(torch.fmod(self.emb.weight, 1 / math.sqrt(d * self.c)))
        self.emb.weight.detach()
        #print(self.emb.weight)
        #print(1 / math.sqrt(d * self.c))

        #torch.fmod(self.emb.weight, 0.04)


    def encode(self, x):
        return self.encoder(x)

    def encode_index(self, x):
        z_e_euclidean = self.encode(x)
        z_e = self.tp(z_e_euclidean)
        #print(z_e)
        #z_q_poicare, argmin = self.emb(z_e, weight_sg=True)
        z_q_poicare, argmin = self.emb(z_e)
        #print(z_q_poicare)
        #print(argmin)
        return argmin



    def decode(self, x):
        return torch.tanh(self.decoder(x))

    def forward(self, x):
        #z_e = self.encode(x)

        z_e_euclidean = self.encode(x)

        z_e = self.tp(z_e_euclidean)

        self.f = z_e.shape[-1]

        z_q_poicare, argmin = self.emb(z_e, weight_sg=True)

        z_q = self.fp(z_q_poicare)
        #print(z_q.shape)
        emb, _ = self.emb(z_e.detach())
        return self.decode(z_q), z_e, emb, argmin

    def sample(self, size):
        sample = torch.randn(size, self.d, self.f,
                             self.f, requires_grad=False),
        if self.cuda():
            sample = sample.cuda()
        emb, _ = self.emb(sample)
        return self.decode(emb.view(size, self.d, self.f, self.f)).cpu()

    def loss_function(self, x, recon_x, z_e, emb, argmin):
        self.mse = F.mse_loss(recon_x, x)

        self.vq_loss = torch.mean(pmath._dist_for_vqvae(emb, z_e.detach(), self.c))
        self.commit_loss = torch.mean(
            pmath._dist_for_vqvae(emb.detach(), z_e, self.c))

        #self.vq_loss = torch.mean(torch.norm((emb - z_e.detach())**2, 2, 1))
        #self.commit_loss = torch.mean(
        #    torch.norm((emb.detach() - z_e)**2, 2, 1))

        return self.mse + self.vq_coef*self.vq_loss + self.commit_coef*self.commit_loss

    def latest_losses(self):
        #print(self.mse, self.vq_loss, self.commit_loss)
        return {'mse': self.mse, 'vq': self.vq_loss, 'commitment': self.commit_loss,
                'total': self.mse + self.vq_coef*self.vq_loss + self.commit_coef*self.commit_loss}

    def print_atom_hist(self, argmin):

        argmin = argmin.detach().cpu().numpy()
        unique, counts = np.unique(argmin, return_counts=True)
        logging.info(counts)
        logging.info(unique)


class VQ_CVAE2(nn.Module):
    def __init__(self, d, k=10, bn=True, vq_coef=1, commit_coef=0.5, num_channels=3, **kwargs):
        super(VQ_CVAE2, self).__init__()



























# class EncWrapped(nn.Module):
#     """ Usual encoder followed by an exponential map """
#     def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim, prior_iso):
#         super(EncWrapped, self).__init__()
#         self.manifold = manifold
#         self.data_size = data_size
#         modules = []
#         modules.append(nn.Sequential(nn.Linear(prod(data_size), hidden_dim), non_lin))
#         modules.extend([extra_hidden_layer(hidden_dim, non_lin) for _ in range(num_hidden_layers - 1)])
#         self.enc = nn.Sequential(*modules)
#         self.fc21 = nn.Linear(hidden_dim, manifold.coord_dim)
#         self.fc22 = nn.Linear(hidden_dim, manifold.coord_dim if not prior_iso else 1)
#
#     def forward(self, x):
#         e = self.enc(x.view(*x.size()[:-len(self.data_size)], -1))
#         mu = self.fc21(e)          # flatten data
#         mu = self.manifold.expmap0(mu)
#         return mu, F.softplus(self.fc22(e)) + Constants.eta,  self.manifold



# class DecWrapped(nn.Module):
#     """ Usual encoder preceded by a logarithm map """
#     def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim):
#         super(DecWrapped, self).__init__()
#         self.data_size = data_size
#         self.manifold = manifold
#         modules = []
#         modules.append(nn.Sequential(nn.Linear(manifold.coord_dim, hidden_dim), non_lin))
#         modules.extend([extra_hidden_layer(hidden_dim, non_lin) for _ in range(num_hidden_layers - 1)])
#         self.dec = nn.Sequential(*modules)
#         self.fc31 = nn.Linear(hidden_dim, prod(data_size))
#
#     def forward(self, z):
#         z = self.manifold.logmap0(z)
#         d = self.dec(z)
#         mu = self.fc31(d).view(*z.size()[:-1], *self.data_size)  # reshape data
#         return mu, torch.ones_like(mu)



class P_WRAPPED_VAE(AbstractAutoEncoder):


    def __init__(self, d, c=0.001, kl_coef=0.1, bn=True, num_channels=3, transforms_size=256, cuda=True, **kwargs):
        super(P_WRAPPED_VAE, self).__init__()



        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=bn),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=bn),
            nn.BatchNorm2d(d),
        )



        self.decoder = nn.Sequential(
            ResBlock(d, d),
            nn.BatchNorm2d(d),
            ResBlock(d, d),
            nn.ConvTranspose2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                d, num_channels, kernel_size=4, stride=2, padding=1),
        )
        #self.f = int(math.sqrt(num_channels * transforms_size ** 2 / 3 * 2 / d))
        self.f = 8
        self.transforms_size  = transforms_size
        self.d = d
        self.c = c

        self.tp = hypnn.ToPoincare(
            c=c, train_x=False, train_c=False, ball_dim=d * int(self.transforms_size / 4) ** 2
        )

        self.fp = hypnn.FromPoincare(
            c=c, train_x=False, train_c=False, ball_dim=d * int(self.transforms_size / 4) ** 2
        )

        #self.fc11 = nn.Linear(d * int(self.transforms_size / 4) ** 2, d * int(self.transforms_size / 4) ** 2)
        #self.fc11 = nn.Linear(int(num_channels * transforms_size ** 2 / 3 * 2), d)
        #self.fc11 = nn.Linear(num_channels * self.f ** 2, d)
        #self.fc12 = nn.Linear(d * int(self.transforms_size / 4) ** 2, d * int(self.transforms_size / 4) ** 2)
        #self.fc12 = nn.Linear(int(num_channels * transforms_size ** 2 / 3 * 2), d)
        #self.fc12 = nn.Linear(num_channels * self.f ** 2, d)

        # self.fc11 = nn.Linear(d * int(self.transforms_size / 4) ** 2, d)
        # self.fc12 = nn.Linear(d * int(self.transforms_size / 4) ** 2, d)
        #
        # self.fc21 = nn.Linear(d, d * int(self.transforms_size / 4) ** 2)

        self.fc11 = nn.Sequential(
            nn.Linear(d * int(self.transforms_size / 4) ** 2, d),
            nn.BatchNorm1d(d),
        )
        # self.fc11 = nn.Linear(int(num_channels * transforms_size ** 2 / 3 * 2), d)
        # self.fc11 = nn.Linear(num_channels * self.f ** 2, d)
        # self.fc12 = nn.Linear(d * int(self.transforms_size / 4) ** 2, d * int(self.transforms_size / 4) ** 2)
        self.fc12 = nn.Sequential(
            nn.Linear(d * int(self.transforms_size / 4) ** 2, d),
            nn.BatchNorm1d(d),
        )

        self.fc21 = nn.Sequential(
            nn.Linear(d, d * int(self.transforms_size / 4) ** 2),
            nn.BatchNorm1d(d * int(self.transforms_size / 4) ** 2),
        )


        self.kl_coef = kl_coef
        self.kl_loss = 0
        self.mse = 0

        # if (cuda):
        #     self._pz_mu = nn.Parameter(torch.zeros(1, self.d * int(self.transforms_size / 4) ** 2).cuda(), requires_grad=False)
        #     self._pz_logvar = nn.Parameter(torch.zeros(1, 1).cuda(), requires_grad=False)
        # else:
        #     self._pz_mu = nn.Parameter(torch.zeros(1, self.d * int(self.transforms_size / 4) ** 2),
        #                                requires_grad=False)
        #     self._pz_logvar = nn.Parameter(torch.zeros(1, 1), requires_grad=False)

        if (cuda):
            self._pz_mu = nn.Parameter(torch.zeros(1, self.d).cuda(), requires_grad=False)
            self._pz_logvar = nn.Parameter(torch.zeros(1, 1).cuda(), requires_grad=False)
        else:
            self._pz_mu = nn.Parameter(torch.zeros(1, self.d), requires_grad=False)
            self._pz_logvar = nn.Parameter(torch.zeros(1, 1), requires_grad=False)




        # self.pz = \
        #     hypdist.wrapped_normal.WrappedNormal(self._pz_mu.mul(1),
        #                                          F.softplus(self._pz_logvar).div(math.log(2)),
        #                                          manifolds.poincareball.PoincareBall(
        #                                              self.d * int(self.transforms_size / 4) ** 2, self.c))

        self.pz = \
            hypdist.wrapped_normal.WrappedNormal(self._pz_mu.mul(1),
                                                 F.softplus(self._pz_logvar).div(math.log(2)),
                                                 manifolds.poincareball.PoincareBall(self.d, self.c))






    def encode(self, x):
        #print(x.shape)
        #print(x)
        h1 = self.encoder(x)
        #h1 = torch.flatten(h1, start_dim=1)
        #print(h1.shape)
        #print("....................")
        #print(h1.shape[1])
        #print(self.d)
        h2 = h1.view(-1, self.d * int(self.transforms_size / 4) ** 2)
        #print(h2.shape)


        #h2 = self.tp(h2)

        #return self.fc11(h1), self.fc12(h1)
        #print("????????????????????????????????????????????????????????????")
        #print(x)

        return self.tp(self.fc11(h2)), self.fc12(h2)

    # def reparameterize(self, mu, logvar):
    #     if self.training:
    #         std = logvar.mul(0.5).exp_()
    #         eps = std.new(std.size()).normal_()
    #         return eps.mul(std).add_(mu)
    #     else:
    #         return mu

    def decode(self, z):
        z = self.fp(z)
        z = self.fc21(z)
        #z = z.view(-1, self.d, self.f, self.f)
        z = z.view(-1, self.d, int(self.transforms_size / 4), int(self.transforms_size / 4))
        #print(z.shape)
        #print("---------------------------")
        h3 = self.decoder(z)
        return torch.tanh(h3)

    def forward(self, x, K=1):
        #print(x.shape)
        mu, logvar = self.encode(x)
        #print(mu)
        #print(mu.shape)
        #print("ZZZZZZZZZZZZZZZ")
        #qz_x = hypdist.wrapped_normal.WrappedNormal(mu, F.softplus(logvar) + 1e-5,
        #                                            manifolds.poincareball.PoincareBall(
        #                                                self.d * int(self.transforms_size / 4) ** 2, self.c))

        qz_x = hypdist.wrapped_normal.WrappedNormal(mu, F.softplus(logvar) + 1e-5,
                                                    manifolds.poincareball.PoincareBall(self.d, self.c))
        #print("xxxxxxxxxxxxxxxxxxxx")
        #print(mu.shape)
        #print(K)
        zs = qz_x.rsample(torch.Size([K]))
        #print(zs.shape)
        #print(zs.shape)
        #print(mu.shape)
        #z = self.reparameterize(mu, logvar)
        xr = self.decode(mu)
        #print("xr shape is :")
        #print(xr.shape)
        #print(xr[0][0][0])
        #px_z = dist.RelaxedBernoulli(xr, torch.ones_like(xr))
        #px_z = self.px_z(*self.dec(zs))
        #return self.decode(z), mu, logvar
        #return qz_x, px_z, zs
        #return xr, zs, qz_x, px_z
        return xr, zs, qz_x, mu

    def sample(self, size):
        sample = torch.randn(size, self.d * self.f ** 2, requires_grad=False)
        if self.cuda():
            sample = sample.cuda()

        return self.decode(sample).cpu()

    #def loss_function(self, x, recon_x, mu, logvar):
    #def loss_function(self, x, xr, zs, qz_x, px_z):
    def loss_function(self, x, xr, zs, qz_x, mu):
        #print(recon_x.shape)
        #print(x.shape)
        self.mse = F.mse_loss(xr, x)
        #flat_rest = torch.Size([*px_z.batch_shape[:2], -1])
        #print(x.shape)
        #print(px_z.batch_shape)
        #print(x.expand(px_z.batch_shape).shape)
        #self.lpx_z = px_z.log_prob(x.expand(px_z.batch_shape)).view(flat_rest).sum(-1)
        #self.lpx_z = px_z.log_prob(x.expand(px_z.batch_shape))
        #print(x.expand(px_z.batch_shape)[0][0][0])
        #print(self.lpx_z[0][0][0])

        #self.lpx_z = self.lpx_z.mean(0).sum()
        #print("???????????????????????????????????????")
        #print(self.lpx_z)

        #batch_size = x.size(0)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        #print(mu.shape)
        #mu = mu[None, :]
        #print(mu.shape)

        #print((qz_x.log_prob(mu).sum(-1) - self.pz.log_prob(mu).sum(-1)).mean(0).sum())

        #print(zs.shape)
        #print(qz_x.log_prob(zs))
        #print(qz_x.log_prob(zs).shape)
        #print(qz_x.log_prob(zs).sum(-1).sum(0).shape)
        #print(torch.abs(qz_x.log_prob(zs).sum(-1).sum(0) - self.pz.log_prob(zs).sum(-1).sum(0)))
        #print(self.pz.log_prob(zs).shape)
        #self.kl_loss = qz_x.log_prob(zs).sum(-1) - self.pz.log_prob(zs).sum(-1)
        self.kl_loss = torch.abs(qz_x.log_prob(zs).sum(-1).sum(0) - self.pz.log_prob(zs).sum(-1).sum(0))
        #self.kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Normalise by same number of elements as in reconstruction
        self.kl_loss = self.kl_loss.mean(0).sum()
        #self.kl_loss /= batch_size * 3 * 1024

        #self.kl_loss = (self.pz.log_prob(mu).sum(-1) - qz_x.log_prob(mu).sum(-1)).mean(0).sum() ???? it should be correct

        #self.kl_loss = (qz_x.log_prob(mu).sum(-1) - self.pz.log_prob(mu).sum(-1)).mean(0).sum()

        #print(self.lpx_z)
        #print(self.kl_coef * self.kl_loss)
        #print(self.lpx_z + self.kl_coef * self.kl_loss)
        # return mse
        #return self.lpx_z + self.kl_coef * self.kl_loss
        return self.mse + self.kl_coef * self.kl_loss

    def latest_losses(self):
        return {'mse': self.mse, 'kl': self.kl_loss}

