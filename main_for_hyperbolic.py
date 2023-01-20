import os
import sys
import time
import logging
import argparse

import torch.utils.data
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

from vq_vae_hyperbolic.util import setup_logging_from_args
from vq_vae_hyperbolic.auto_encoder import *

import matplotlib.pyplot as plt

import matplotlib.colors as mcolors

import geoopt
import math

from dataset_parameter import *







# models = {
#     'custom': {'vqvae': VQ_CVAE,
#                'vqvae2': VQ_CVAE2},
#     'imagenet': {'vqvae': VQ_CVAE,
#                  'vqvae2': VQ_CVAE2},
#     'cifar10': {'vae': CVAE,
#                 'vqvae': VQ_CVAE,
#                 'vqvae2': VQ_CVAE2},
#     'mnist': {'vae': VAE,
#               'vqvae': VQ_CVAE},
#     'svhn': {'vae': CVAE,
#              'vqvae': VQ_CVAE},
# }

"""
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
"""

mse_loss_train = []
mse_loss_test = []

class MultipleScheduler(object):
    def __init__(self, sch):
        self.schedulers = sch

    def step(self):
        for sch in self.schedulers:
            sch.step()

class MultipleOptimizer(object):
    def __init__(self, *op):
        self.optimizers = op
        self.scheduler = []

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

    def optim_lr_schedulaer_StepLR(self, step_size, gamma=0.1, last_epoch=-1, verbose=False):
        for op in self.optimizers:
            self.scheduler.append(optim.lr_scheduler.StepLR(op, step_size, gamma, last_epoch, verbose))

        return MultipleScheduler(self.scheduler)

def main(args):
    parser = argparse.ArgumentParser(description='Variational AutoEncoders')

    model_parser = parser.add_argument_group('Model Parameters')
    model_parser.add_argument('--model', default='vae', choices=['vae', 'vqvae', 'pvae'],
                              help='autoencoder variant to use: vae | vqvae | pvae')
    model_parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                              help='input batch size for training (default: 128)')
    model_parser.add_argument('--hidden', type=int, metavar='d',
                              help='number of hidden channels')
    model_parser.add_argument('--k', '--dict-size', default=10, type=int, dest='k', metavar='K',
                              help='number of atoms in dictionary')
    model_parser.add_argument('--transforms-size', '--image-height-and-width', type=int, dest='transforms_size',
                              metavar='transforms_size',
                              help='image height and width, values are unique based on dataset: '
                                   'custom: 256 | imagenet: 256 | cifar10: 32 | mnist: 28 | svhn: 256'
                                   'the hidden layer of VAE has hidden * transforms-size/4 * transforms-size/4 features')
    model_parser.add_argument('--c', '--curvature', type=float, dest='c', metavar='C',
                              help='negative curvature of the hyperbolic space')
    model_parser.add_argument('--lr', type=float, default=None,
                              help='learning rate')
    model_parser.add_argument('--vq_coef', type=float, default=None,
                              help='vq coefficient in loss')
    model_parser.add_argument('--commit_coef', type=float, default=None,
                              help='commitment coefficient in loss')
    model_parser.add_argument('--kl_coef', type=float, default=0.1,
                              help='kl-divergence coefficient in loss')

    training_parser = parser.add_argument_group('Training Parameters')
    training_parser.add_argument('--dataset', default='cifar10', choices=['mnist', 'cifar10', 'imagenet', 'svhn', 'custom'], help='dataset to use: mnist | cifar10 | imagenet | svhn | custom')
    training_parser.add_argument('--dataset_dir_name', default='',
                                 help='name of the dir containing the dataset if dataset == custom')
    training_parser.add_argument('--train_dir', default='train',
                                 help='name of the dir containing the dataset of train')
    training_parser.add_argument('--val_dir', default='val',
                                 help='name of the dir containing the dataset of validation')
    training_parser.add_argument('--data-dir', default='/media/ssd/Datasets',
                                 help='directory containing the dataset')
    training_parser.add_argument('--epochs', type=int, default=20, metavar='N',
                                 help='number of epochs to train (default: 10)')
    training_parser.add_argument('--max-epoch-samples', type=int, default=50000,
                                 help='max num of samples per epoch')
    training_parser.add_argument('--no-cuda', action='store_true', default=False,
                                 help='enables CUDA training')
    training_parser.add_argument('--seed', type=int, default=1, metavar='S',
                                 help='random seed (default: 1)')
    training_parser.add_argument('--gpus', default='0',
                                 help='gpus used for training - e.g 0,1,3')

    logging_parser = parser.add_argument_group('Logging Parameters')
    logging_parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                                help='how many batches to wait before logging training status')
    logging_parser.add_argument('--results-dir', metavar='RESULTS_DIR', default='./results',
                                help='results dir')
    logging_parser.add_argument('--save-name', default='',
                                help='saved folder')
    logging_parser.add_argument('--data-format', default='json',
                                help='in which format to save the data')
    args = parser.parse_args(args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print("*******no_cuda is {}***********".format(args.no_cuda))
    print("*******cuda available is {}***********".format(torch.cuda.is_available()))

    dataset_dir_name = args.dataset if args.dataset != 'custom' else args.dataset_dir_name

    lr = args.lr or default_hyperparams[args.dataset]['lr']
    k = args.k or default_hyperparams[args.dataset]['k']
    c = args.c or default_hyperparams[args.dataset]['c']
    hidden = args.hidden or default_hyperparams[args.dataset]['hidden']
    transforms_size = args.transforms_size or default_hyperparams[args.dataset]['transforms_size']
    num_channels = dataset_n_channels[args.dataset]

    save_path = setup_logging_from_args(args)
    writer = SummaryWriter(save_path)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
        args.gpus = [int(i) for i in args.gpus.split(',')]
        torch.cuda.set_device(args.gpus[0])
        cudnn.benchmark = True
        torch.cuda.manual_seed(args.seed)

    model = models[args.dataset][args.model](hidden, k=k, c=c, kl_coef=args.kl_coef, num_channels=num_channels,
                                             transforms_size=transforms_size, cuda=args.cuda)
    if args.cuda:
        model.cuda()
    print("k is ")
    print(args.k)
    print(int(args.k))
    #print([para for para in model.emb.parameters()])

    #optimizer = optim.Adam(model.parameters(), lr=lr)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, 10 if args.dataset == 'imagenet' else 30, 0.5,)
    torch.autograd.set_detect_anomaly(True)

    if (args.model == 'vqvae'):
        optimizer_dec = optim.Adam(model.decoder.parameters(), lr=lr)
        optimizer_emb = geoopt.optim.RiemannianAdam(model.emb.parameters(), lr=lr)
        optimizer_enc = optim.Adam(model.encoder.parameters(), lr=lr)
        optimizer = MultipleOptimizer(optimizer_dec, optimizer_emb, optimizer_enc)

        #scheduler = optimizer.optim_lr_schedulaer_StepLR(10 if args.dataset == 'imagenet' else 30, 0.5,)
        scheduler = optimizer.optim_lr_schedulaer_StepLR(10 if args.dataset == 'imagenet' else 5, 0.5, )

        #scheduler_dec = optim.lr_scheduler.StepLR(optimizer_dec, 10 if args.dataset == 'imagenet' else 30, 0.5,)
        #scheduler_emb = optim.lr_scheduler.StepLR(optimizer_emb, 10 if args.dataset == 'imagenet' else 30, 0.5,)
        #scheduler_enc = optim.lr_scheduler.StepLR(optimizer_enc, 10 if args.dataset == 'imagenet' else 30, 0.5,)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 10 if args.dataset == 'imagenet' else 5, 0.5,)



    kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
    dataset_train_dir = os.path.join(args.data_dir, dataset_dir_name)
    dataset_test_dir = os.path.join(args.data_dir, dataset_dir_name)
    if args.dataset in ['imagenet', 'custom']:
        dataset_train_dir = os.path.join(dataset_train_dir, args.train_dir)
        dataset_test_dir = os.path.join(dataset_test_dir, args.val_dir)
    train_loader = torch.utils.data.DataLoader(
        datasets_classes[args.dataset](dataset_train_dir,
                                       transform=dataset_transforms[args.dataset],
                                       **dataset_train_args[args.dataset]),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets_classes[args.dataset](dataset_test_dir,
                                       transform=dataset_transforms[args.dataset],
                                       **dataset_test_args[args.dataset]),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    for epoch in range(1, args.epochs + 1):

        train_losses = train(epoch, model, train_loader, optimizer, args.cuda,
                             args.log_interval, save_path, args, writer)
        test_losses = test_net(epoch, model, test_loader, args.cuda, save_path, args, writer)

        for k in train_losses.keys():
            name = k.replace('_train', '')
            train_name = k
            test_name = k.replace('train', 'test')
            writer.add_scalars(name, {'train': train_losses[train_name],
                                      'test': test_losses[test_name],
                                      })
        scheduler.step()
        #scheduler_dec.step()
        #scheduler_emb.step()
        #scheduler_enc.step()

    #plt.show()

def train(epoch, model, train_loader, optimizer, cuda, log_interval, save_path, args, writer):
    model.train()
    loss_dict = model.latest_losses()
    losses = {k + '_train': 0 for k, v in loss_dict.items()}
    epoch_losses = {k + '_train': 0 for k, v in loss_dict.items()}
    start_time = time.time()
    batch_idx, data = None, None
    for batch_idx, (data, _) in enumerate(train_loader):
        if cuda:
            data = data.cuda()
        optimizer.zero_grad()
        outputs = model(data)
        #print(list(model.parameters()))
        loss = model.loss_function(data, *outputs)
        #print("loss1")
        loss.backward()
        #print("loss1---------------------------------------------------")
        #print(model.encoder)
        #print([param for param in model.encoder.parameters()])
        optimizer.step()
        latest_losses = model.latest_losses()
        #################################################################################################

        #################################

        """
        #print(model.emb.weight)
        for code_dim in range(model.emb.weight.shape[1]):
        #     #print(outputs[2].shape)
        #     #print(outputs[2][0:32, 0, 0, code_dim])
        #     first_five_dim_norm = outputs[2][0:32, 0, 0, code_dim].norm()
        #     second_five_dim_norm = outputs[2][32:, 0, 0, code_dim].norm()
        #     norm_code_book_vector = outputs[2][:,0,0,code_dim].norm()
        #     #print(mcolors.TABLEAU_COLORS.keys())
        #     mcolornames = list(mcolors.TABLEAU_COLORS.keys())
        #     plt.plot(first_five_dim_norm.detach().cpu(),
        #              second_five_dim_norm.cpu(), color=mcolornames[code_dim], marker='o',
        #              markersize=0.01)
        #     plt.xlim([0, 1])
        #     plt.ylim([0, 1])
            # print(first_five_dim_norm/norm_code_book_vector, second_five_dim_norm/norm_code_book_vector)
            # print("-------------------------------")

            # print(outputs[2].shape)
            # print(outputs[2][0:32, 0, 0, code_dim])
            # print(model.emb.weight.shape)
            first_five_dim_norm = model.emb.weight[0, code_dim]
            second_five_dim_norm = model.emb.weight[1, code_dim]
            #print(first_five_dim_norm, second_five_dim_norm)
            mcolornames = list(mcolors.TABLEAU_COLORS.keys())

            plt.plot(first_five_dim_norm.detach().cpu(),
                     second_five_dim_norm.detach().cpu(), color=mcolornames[code_dim], marker='o',
                     markersize=0.1)
            radius_of_poicare = 1 / math.sqrt(args.hidden * args.c)
            plt.xlim([-radius_of_poicare, radius_of_poicare])
            plt.ylim([-radius_of_poicare, radius_of_poicare])
            # plt.xlim([-0.5, 0.5])
            # plt.ylim([-0.5, 0.5])
            # print(first_five_dim_norm, second_five_dim_norm)
            # print("-------------------------------")
        if batch_idx == 0:
            #writer.add_figure(f'codebookvectorpath_{str(epoch)}', plt.gcf())
            plt.savefig(save_path + "/codebookvectorpath_" + str(epoch))
        #################################
        """

        mse_loss_train.append(float(latest_losses['mse']))


        for key in latest_losses:
            losses[key + '_train'] += float(latest_losses[key])
            epoch_losses[key + '_train'] += float(latest_losses[key])
        if (batch_idx+1) % log_interval == 0:

            for key in latest_losses:
                losses[key + '_train'] /= log_interval
            loss_string = ' '.join(['{}: {:.6f}'.format(k, v) for k, v in losses.items()])
            # logging.info("z_e shape is " + str(outputs[1].shape))
            logging.info('Train Epoch: {epoch} [{batch:5d}/{total_batch} ({percent:2d}%)]   time:'
                         ' {time:3.2f}   {loss}'
                         .format(epoch=epoch, batch=batch_idx * len(data), total_batch=len(train_loader) * len(data),
                                 percent=int(100. * batch_idx / len(train_loader)),
                                 time=time.time() - start_time,
                                 loss=loss_string))


            start_time = time.time()
            # logging.info('z_e norm: {:.2f}'.format(float(torch.mean(torch.norm(outputs[1][0].contiguous().view(256,-1),2,0)))))
            # logging.info('z_q norm: {:.2f}'.format(float(torch.mean(torch.norm(outputs[2][0].contiguous().view(256,-1),2,0)))))
            for key in latest_losses:
                losses[key + '_train'] = 0
        if batch_idx == (len(train_loader) - 1):
            save_reconstructed_images(data, epoch, outputs[0], save_path, 'reconstruction_train')

            write_images(data, outputs, writer, 'train')

        if args.dataset in ['imagenet', 'custom'] and batch_idx * len(data) > args.max_epoch_samples:
            break



    for key in epoch_losses:
        if args.dataset != 'imagenet':
            epoch_losses[key] /= (len(train_loader.dataset) / train_loader.batch_size)
        else:
            epoch_losses[key] /= (len(train_loader.dataset) / train_loader.batch_size)
    loss_string = '\t'.join(['{}: {:.6f}'.format(k, v) for k, v in epoch_losses.items()])
    logging.info('====> Epoch: {} {}'.format(epoch, loss_string))
    print(np.array([v for k, v in epoch_losses.items()]))
    writer.add_scalars('loss/train', epoch_losses, epoch)
    # *****************************************************************************************************************
    #print(mse_loss_train)
    logging.info(mse_loss_train)
    # *****************************************************************************************************************
    #print(outputs)
    try:
        writer.add_histogram('dict frequency', outputs[3], bins=range(args.k + 1))
        model.print_atom_hist(outputs[3])
    except:
        logging.info("VAE model doe not have codebook")
    return epoch_losses


def test_net(epoch, model, test_loader, cuda, save_path, args, writer):
    model.eval()
    loss_dict = model.latest_losses()
    losses = {k + '_test': 0 for k, v in loss_dict.items()}
    i, data = None, None
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            if cuda:
                data = data.cuda()
            outputs = model(data)
            model.loss_function(data, *outputs)
            latest_losses = model.latest_losses()
            mse_loss_test.append(float(latest_losses['mse']))
            for key in latest_losses:
                losses[key + '_test'] += float(latest_losses[key])
            if i == 0:
                write_images(data, outputs, writer, 'test')

                save_reconstructed_images(data, epoch, outputs[0], save_path, 'reconstruction_test')
                save_checkpoint(model, epoch, save_path)

                if (args.dataset in ['mnist', 'cifar10']):
                    hidden_height = 8
                    # visual_features = outputs[3].view(-1, 7*7)#data.view(-1, 28 * 28)
                    # writer.add_embedding(visual_features, label_img=data)

                    visual_embcode = model.emb.weight.contiguous() #outputs[2].contiguous()
                    encodings_all_multi = torch.zeros(args.k, hidden_height*hidden_height*args.k)
                    for ks in range(args.k):
                        for cks in range(hidden_height*hidden_height):
                            encodings_all_multi[ks][hidden_height*hidden_height*ks + cks] = 1.0
                    encodings_all_multi = encodings_all_multi.cuda()
                    #encodings_all_multi
                    label_embed = torch.matmul(visual_embcode, encodings_all_multi)
                    label_embed = label_embed.view(args.hidden, args.k, hidden_height, hidden_height)
                    label_embed = label_embed.permute(1, 0, 2, 3)
                    label_img = model.decode(model.fp(label_embed))
                    #print(label_embed.shape)

                    #coder = visual_embcode.t().index_select(0, i_c * torch.ones(7 * 7).int().cuda()).view(1, hidden, 7, 7)
                    #print(visual_embcode.shape)
                    visual_embcode = visual_embcode.transpose(0, 1)
                    writer.add_embedding(visual_embcode, label_img=label_img)
            if args.dataset == 'imagenet' and i * len(data) > 1000:
                break

    for key in losses:
        if args.dataset not in ['imagenet', 'custom']:
            losses[key] /= (len(test_loader.dataset) / test_loader.batch_size)
        else:
            losses[key] /= (i * len(data))
    loss_string = ' '.join(['{}: {:.6f}'.format(k, v) for k, v in losses.items()])
    print(np.array([v for k, v in losses.items()]))
    writer.add_scalars('loss/test', losses, epoch)
    # *****************************************************************************************************************
    # print(mse_loss_test)
    logging.info(mse_loss_test)
    # *****************************************************************************************************************
    logging.info('====> Test set losses: {}'.format(loss_string))
    return losses


def write_images(data, outputs, writer, suffix):
    original = data.mul(0.5).add(0.5)
    original_grid = make_grid(original[:6])
    writer.add_image(f'original/{suffix}', original_grid)
    reconstructed = outputs[0].mul(0.5).add(0.5)
    reconstructed_grid = make_grid(reconstructed[:6])
    writer.add_image(f'reconstructed/{suffix}', reconstructed_grid)


def save_reconstructed_images(data, epoch, outputs, save_path, name):
    size = data.size()
    n = min(data.size(0), 8)
    batch_size = data.size(0)
    comparison = torch.cat([data[:n],
                            outputs.view(batch_size, size[1], size[2], size[3])[:n]])
    save_image(comparison.cpu(),
               os.path.join(save_path, name + '_' + str(epoch) + '.png'), nrow=n, normalize=True)


def save_checkpoint(model, epoch, save_path):
    os.makedirs(os.path.join(save_path, 'checkpoints'), exist_ok=True)
    checkpoint_path = os.path.join(save_path, 'checkpoints', f'model_{epoch}.pth')
    torch.save(model.state_dict(), checkpoint_path)


if __name__ == "__main__":
    main(sys.argv[1:])
