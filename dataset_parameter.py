import os
import sys
import time
import logging
import argparse

from torchvision import datasets, transforms

from vq_vae_hyperbolic.auto_encoder import *



models = {
    'custom': {'vae': CVAE,
               'pvae': P_WRAPPED_VAE,
               'vqvae': VQ_CVAE,
               'vqvae2': VQ_CVAE2},
    'imagenet': {'vae': CVAE,
                 'pvae': P_WRAPPED_VAE,
                 'vqvae': VQ_CVAE,
                 'vqvae2': VQ_CVAE2},
    'cifar10': {'vae': CVAE,
                'pvae': P_WRAPPED_VAE,
                'vqvae': VQ_CVAE,
                'vqvae2': VQ_CVAE2},
    'mnist': {'vae': CVAE,
              'pvae': P_WRAPPED_VAE,
              'vqvae': VQ_CVAE},
    'svhn': {'vae': CVAE,
             'pvae': P_WRAPPED_VAE,
             'vqvae': VQ_CVAE},
}
datasets_classes = {
    'custom': datasets.ImageFolder,
    'imagenet': datasets.ImageNet,
    'cifar10': datasets.CIFAR10,
    'mnist': datasets.MNIST,
    'svhn': datasets.SVHN
}
dataset_train_args = {
    'custom': {},
    'imagenet': {},
    'cifar10': {'train': True, 'download': True},
    'mnist': {'train': True, 'download': True},
    'svhn': {'split': 'train', 'download': True},
}
dataset_test_args = {
    'custom': {},
    'imagenet': {},
    'cifar10': {'train': False, 'download': True},
    'mnist': {'train': False, 'download': True},
    'svhn': {'split': 'test', 'download': True},
}
dataset_n_channels = {
    'custom': 3,
    'imagenet': 3,
    'cifar10': 3,
    'mnist': 1,
    'svhn': 3,
}

dataset_transforms = {
    'custom': transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    'imagenet': transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    'cifar10': transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    'mnist': transforms.ToTensor(),
    'svhn': transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
}
default_hyperparams = {
    'custom': {'lr': 2e-4, 'k': 512, 'hidden': 128, 'c': 0.001, 'transforms_size': 256},
    'imagenet': {'lr': 2e-4, 'k': 512, 'hidden': 128, 'c': 0.001, 'transforms_size': 256},
    'cifar10': {'lr': 2e-4, 'k': 10, 'hidden': 256, 'c': 0.001, 'transforms_size': 32},
    'mnist': {'lr': 1e-4, 'k': 10, 'hidden': 64, 'c': 0.001, 'transforms_size': 28},
    'svhn': {'lr': 2e-4, 'k': 512, 'hidden': 128, 'c': 0.001, 'transforms_size': 256},
}
