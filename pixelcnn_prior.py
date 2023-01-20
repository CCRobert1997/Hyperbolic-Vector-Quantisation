import logging
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as transformer_F
import json
from torch import optim
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid


from pixelcnn.modules import VectorQuantizedVAE, GatedPixelCNN

import vq_vae_hyperbolic.auto_encoder as auto_encoder
from vq_vae_hyperbolic.auto_encoder import VQ_CVAE
#from datasets import MiniImagenet

#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

from dataset_parameter import *
import matplotlib.pyplot as plt

#from reconstruction import tensor2im, save_image





def train(data_loader, model, prior, optimizer, args, writer):
    #print(args.steps)
    transforms_size = default_hyperparams[args.dataset]['transforms_size']
    for images, labels in data_loader:
        with torch.no_grad():
            images = images.to(args.device)
            #print(images.shape)
            latents = model.encode_index(images)
            #print(latents.shape)
            #print("...............................")
            #print(latents[1])
            latents = latents.detach()


        labels = labels.to(args.device)


        #torch.randperm(labels.shape[0])

        # latents_generate = torch.zeros(
        #     (labels.shape[0], int(transforms_size/4), int(transforms_size/4)),
        #     dtype=torch.int64, device=args.device
        # )
        latents_generate = latents.contiguous()

        # print(labels.shape)
        # print(latents_generate.shape)
        # print(latents.shape)

        logits = prior(latents_generate, labels)
        logits = logits.permute(0, 2, 3, 1).contiguous()

        optimizer.zero_grad()
        loss = F.cross_entropy(logits.view(-1, args.k),
                               latents_generate.view(-1))


        loss.backward()

        # Logs
        writer.add_scalar('loss/train', loss.item(), args.steps)

        optimizer.step()
        args.steps += 1

def test(data_loader, model, prior, args, writer):
    transforms_size = default_hyperparams[args.dataset]['transforms_size']
    with torch.no_grad():
        loss = 0.
        for images, labels in data_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)

            latents = model.encode_index(images)
            latents = latents.detach()
            #print(latents)

            # latents_generate = torch.zeros(
            #     (labels.shape[0], int(transforms_size / 4), int(transforms_size / 4)),
            #     dtype=torch.int64, device=args.device
            # )
            latents_generate = latents.contiguous()

            logits = prior(latents_generate, labels)

            #logits = prior(latents, labels)
            logits = logits.permute(0, 2, 3, 1).contiguous()
            loss += F.cross_entropy(logits.view(-1, args.k),
                                    latents.view(-1))

        loss /= len(data_loader)

    # Logs
    writer.add_scalar('loss/valid', loss.item(), args.steps)

    return loss.item()



"""
def main(args):
    writer = SummaryWriter('./logs/{0}'.format(args.output_folder))
    save_filename = './models/{0}/prior.pt'.format(args.output_folder)

    if args.dataset in ['mnist', 'fashion-mnist', 'cifar10']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        if args.dataset == 'mnist':
            # Define the train & test datasets
            train_dataset = datasets.MNIST(args.data_folder, train=True,
                download=True, transform=transform)
            test_dataset = datasets.MNIST(args.data_folder, train=False,
                transform=transform)
            num_channels = 1
        elif args.dataset == 'fashion-mnist':
            # Define the train & test datasets
            train_dataset = datasets.FashionMNIST(args.data_folder,
                train=True, download=True, transform=transform)
            test_dataset = datasets.FashionMNIST(args.data_folder,
                train=False, transform=transform)
            num_channels = 1
        elif args.dataset == 'cifar10':
            # Define the train & test datasets
            train_dataset = datasets.CIFAR10(args.data_folder,
                train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10(args.data_folder,
                train=False, transform=transform)
            num_channels = 3
        valid_dataset = test_dataset
    elif args.dataset == 'miniimagenet':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(128),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # Define the train, valid & test datasets
        train_dataset = MiniImagenet(args.data_folder, train=True,
            download=True, transform=transform)
        valid_dataset = MiniImagenet(args.data_folder, valid=True,
            download=True, transform=transform)
        test_dataset = MiniImagenet(args.data_folder, test=True,
            download=True, transform=transform)
        num_channels = 3

    # Define the data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
        batch_size=args.batch_size, shuffle=False, drop_last=True,
        num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=16, shuffle=True)

    # Save the label encoder
    #with open('./models/{0}/labels.json'.format(args.output_folder), 'w') as f:
    #    json.dump(train_dataset._label_encoder, f)

    # Fixed images for Tensorboard
    fixed_images, _ = next(iter(test_loader))
    fixed_grid = make_grid(fixed_images, nrow=8, range=(-1, 1), normalize=True)
    writer.add_image('original', fixed_grid, 0)

    model = VectorQuantizedVAE(num_channels, args.hidden_size_vae, args.k).to(args.device)
    with open(args.model, 'rb') as f:
        state_dict = torch.load(f)
        model.load_state_dict(state_dict)
    model.eval()
    print(train_dataset)
    if args.dataset == 'miniimagenet':
        prior = GatedPixelCNN(args.k, args.hidden_size_prior,
                              args.num_layers, n_classes=len(train_dataset._label_encoder)).to(args.device)
    else:
        prior = GatedPixelCNN(args.k, args.hidden_size_prior,
                              args.num_layers).to(args.device)
    optimizer = torch.optim.Adam(prior.parameters(), lr=args.lr)

    best_loss = -1.
    for epoch in range(args.num_epochs):
        train(train_loader, model, prior, optimizer, args, writer)
        # The validation loss is not properly computed since
        # the classes in the train and valid splits of Mini-Imagenet
        # do not overlap.
        loss = test(valid_loader, model, prior, args, writer)

        if (epoch == 0) or (loss < best_loss):
            best_loss = loss
            with open(save_filename, 'wb') as f:
                torch.save(prior.state_dict(), f)
                
"""


#import torch
#import numpy as np

#import torchvision.transforms.functional as F


plt.rcParams["savefig.bbox"] = 'tight'


def show(imgs, output_folder):
    imgs = imgs.cpu()
    if not isinstance(imgs, list):
        imgs = imgs.numpy()
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    #print(imgs)
    for i, img in enumerate(imgs):
        #img = img.detach()
        #print(img.shape)
        img = img.astype(np.float32)
        img = transformer_F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        plt.savefig(output_folder + '/pixelcnn_generate_sample.png')


def generate_samples(data_loader, prior, vae_model, args, writer, output_folder):
    transforms_size = default_hyperparams[args.dataset]['transforms_size']
    with torch.no_grad():

        #for images, labels in data_loader:
        dataiter = iter(data_loader)
        images, labels = dataiter.next()
        images = images.to(args.device)
        labels = labels.to(args.device)
        #images = images.to(args.device)
        #generated_sampes = prior.forward(images, labels)
        #latent_shape = (int(transforms_size/4), int(transforms_size/4))
        #print(labels)
        #labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).to(args.device)
        #min_encoding_indices = prior.generate(labels, shape=(int(transforms_size/4), int(transforms_size/4)),
        #                                      batch_size=10)
        
        x_encoding_indices = vae_model.encode_index(images)
        x_encoding_indices = x_encoding_indices.detach()
        min_encoding_indices = prior.generate(x_encoding_indices, labels, shape=(int(transforms_size/4), int(transforms_size/4)),
                                              batch_size=x_encoding_indices.shape[0])
        #print(min_encoding_indices)

        min_encoding_indices = min_encoding_indices.view(-1).unsqueeze(1)
        #print(min_encoding_indices)
        #min_encodings = torch.zeros(min_encoding_indices.shape[0], vae_model.hidden-size-vae).to(device)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], args.k).to(args.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)
        #print(min_encodings)
        #print(vae_model.emb.weight)
        z_q = torch.matmul(vae_model.emb.weight, min_encodings.permute(1, 0).contiguous())
        #z_q = torch.matmul(min_encodings, vae_model.emb.weight)
        z_q = z_q.permute(1, 0).contiguous()
        print(z_q.is_contiguous())
        print(z_q.shape)
        #z_q = z_q.view(-1, int(transforms_size/4), int(transforms_size/4), vae_model.k)
        z_q = z_q.view(-1, int(transforms_size/4), int(transforms_size/4), args.hidden_size_vae)
        print(z_q.shape)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        z_q = vae_model.fp(z_q)
        #print(z_q)
        generated_sampes = vae_model.decode(z_q)
        #print(generated_sampes.shape)
        #generated_sampes = generated_sampes[None,]
        #print(generated_sampes.shape)
        #generated_sampes = generated_sampes.permute(1, 0, 2, 3)
        #print(generated_sampes.shape)
        #grid = make_grid(generated_sampes.cpu())#, nrow=8, range=(-1, 1), normalize=True)
        #print(grid.shape)
        #generated_sampes = generated_sampes.permute(0, 2, 3, 1).contiguous()
        genenrate_n = 8
        size = images.size()
        comparison = torch.cat([images[:genenrate_n], generated_sampes.view(args.batch_size, size[1], transforms_size, transforms_size)[:genenrate_n]])
        
        save_image(comparison.cpu(), output_folder + '/pixelcnn_generate_sample_all.png', nrow=genenrate_n, normalize=True)
        """
        for idx in range(generated_sampes.shape[0]):
            print(output_folder + '/pixelcnn_generate_sample' + str(idx) + '.png')
            save_image(tensor2im(generated_sampes[idx]), output_folder + '/pixelcnn_generate_sample' + str(idx) + '.png')
        """
        #show(generated_sampes, output_folder)
        #writer.add_image('reconstruction', generated_sampes, epoch + 1)


if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='PixelCNN Prior for VQ-VAE')

    # General
    parser.add_argument('--dataset', default='cifar10',
                                 choices=['mnist', 'cifar10', 'imagenet', 'svhn', 'custom'],
                                 help='dataset to use: mnist | cifar10 | imagenet | svhn | custom')
    parser.add_argument('--dataset_dir_name', default='',
                                 help='name of the dir containing the dataset if dataset == custom')
    parser.add_argument('--train_dir', default='train',
                                 help='name of the dir containing the dataset of train')
    parser.add_argument('--val_dir', default='val',
                                 help='name of the dir containing the dataset of validation')
    parser.add_argument('--data-dir', default='/media/ssd/Datasets',
                                 help='directory containing the dataset')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')

    parser.add_argument('--c', '--curvature', type=float, dest='c', metavar='C',
                        help='negative curvature of the hyperbolic space')
    parser.add_argument('--kl_coef', type=float, default=0.1,
                        help='kl-divergence coefficient in loss')

    parser.add_argument('--generate', default=False, action='store_true',
                        help='whether generate sample')


    # parser.add_argument('--data-folder', type=str,
    #     help='name of the data folder')
    # parser.add_argument('--dataset', type=str,
    #     help='name of the dataset (mnist, fashion-mnist, cifar10, miniimagenet)')
    parser.add_argument('--model', type=str,
        help='filename containing the model')

    # Latent space
    parser.add_argument('--hidden-size-vae', '--d', type=int, default=256,
                        help='size of the latent vectors (default: 256)')
    parser.add_argument('--hidden-size-prior', type=int, default=64,
                        help='hidden size for the PixelCNN prior (default: 64)')
    parser.add_argument('--k', '--dict-size', type=int, default=512,
                        help='number of latent vectors (default: 512)')
    parser.add_argument('--num-layers', type=int, default=15,
                        help='number of layers for the PixelCNN prior (default: 15)')


    # Optimization
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size (default: 128)')
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='learning rate for Adam optimizer (default: 3e-4)')

    # Miscellaneous
    # ==========================================================================================
    parser.add_argument('--output-folder', type=str, default='prior',
                        help='name of the output folder (default: prior)')
    # ==========================================================================================
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
                        help='number of workers for trajectories sampling (default: {0})'.format(mp.cpu_count() - 1))
    parser.add_argument('--device', type=str, default='cuda',
                        help='set the device (cpu or cuda, default: cpu)')

    args = parser.parse_args()




    args.cuda = not args.no_cuda and torch.cuda.is_available()

    print('***cuda is {}'.format(args.cuda))

    kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
    dataset_dir_name = args.dataset if args.dataset != 'custom' else args.dataset_dir_name
    num_channels = dataset_n_channels[args.dataset]


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





    # Create logs and models folder if they don't exist
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists('./models'):
        os.makedirs('./models')
    # Device
    args.device = torch.device(args.device
        if torch.cuda.is_available() else 'cpu')
    # Slurm
    if 'SLURM_JOB_ID' in os.environ:
        args.output_folder += '/pixelcnn_model_slurm_job_id-{0}'.format(os.environ['SLURM_JOB_ID'])
    # if not os.path.exists('./models/{0}'.format(args.output_folder)):
    #     os.makedirs('./models/{0}'.format(args.output_folder))
    if not os.path.exists('{0}'.format(args.output_folder)):
        os.makedirs('{0}'.format(args.output_folder))
        if not os.path.exists('{0}/writer_for_pixelcnn_prior'.format(args.output_folder)):
            os.makedirs('{0}/writer_for_pixelcnn_prior'.format(args.output_folder))
    args.steps = 0

    #main(args)

    # Save the label encoder
    # with open('./models/{0}/labels.json'.format(args.output_folder), 'w') as f:
    #    json.dump(train_dataset._label_encoder, f)

    #save_path = "demosavepath"
    #writer = SummaryWriter(save_path)

    writer = SummaryWriter('{0}/writer_for_pixelcnn_prior'.format(args.output_folder))

    save_filename = '{0}/pixelcnn_prior.pt'.format(args.output_folder)

    # Fixed images for Tensorboard
    fixed_images, _ = next(iter(test_loader))
    fixed_grid = make_grid(fixed_images, nrow=8, range=(-1, 1), normalize=True)
    writer.add_image('original', fixed_grid, 0)

    #model = VQ_CVAE(num_channels, args.hidden_size_vae, args.k).to(args.device)
    #model = models[args.dataset][args.model](hidden, k=k, c=c, kl_coef=args.kl_coef, num_channels=num_channels,
    #                                         transforms_size=transforms_size)
    model = VQ_CVAE(d=args.hidden_size_vae, k=args.k, c=args.c, num_channels=num_channels)
    if args.cuda:
        model.cuda()


    with open(args.model, 'rb') as f:
        state_dict = torch.load(f)
        model.load_state_dict(state_dict)
    #print([para for para in model.parameters()])



    #model.eval()
    #print(train_dataset)
    
    if args.dataset == 'custom':
        prior = GatedPixelCNN(args.k, args.hidden_size_prior,
                              args.num_layers, n_classes=len(train_dataset._label_encoder)).to(args.device)
    else:
        prior = GatedPixelCNN(args.k, args.hidden_size_prior,
                              args.num_layers).to(args.device)
    
    #prior = GatedPixelCNN(args.k, args.hidden_size_prior,
    #                      args.num_layers).to(args.device)
    optimizer = torch.optim.Adam(prior.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 15 if args.dataset == 'imagenet' else 30, 0.02,)

    if args.cuda:
        prior.cuda()
    if (args.generate):
        print("*************************generate****************************")
        # prior.load_state_dict(torch.load("/home/shangyu/ShangyuChen/vqvaeOT_Manifoldlearning/VQ-VAE-hyperbolic-master/results/2022-11-29_16-57-47_hyperbolic_data_mnist_model_name_vqvae_k_2_c_1.0_hidden_56/pixelcnn_prior.pt"))
        prior.load_state_dict(torch.load(save_filename))
        if args.cuda:
            prior.cuda()
        generate_samples(test_loader, prior, model, args, writer, args.output_folder)


    best_loss = -1.
    if (not args.generate):
        print("*************************prior-train****************************")
        for epoch in range(args.num_epochs):

            # train(train_loader, model, prior, optimizer, args, writer)
            train(train_loader, model, prior, optimizer, args, writer)
            # The validation loss is not properly computed since
            # the classes in the train and valid splits of Mini-Imagenet
            # do not overlap.
            loss = test(test_loader, model, prior, args, writer)
            #print(loss)
            #print("=============================>")
            print('====> Epoch: {}     loss {}'.format(epoch, loss))
            #generate_samples(test_loader, prior, model, args, writer, args.output_folder)
            if (epoch == 0) or (loss < best_loss):
                best_loss = loss
                with open(save_filename, 'wb') as f:
                    torch.save(prior.state_dict(), f)
            
            #loss_string = '\t'.join(['{}: {:.6f}'.format(k, v) for k, v in epoch_losses.items()])
            #logging.info('====> Epoch: {} {}'.format(epoch, loss_string))
            #loss_string = '\t'.join(['{:.6f}'.format(v) for v in loss])
            #logging.info('====> Epoch: {}     loss {}'.format(epoch, loss))
            scheduler.step()

        #python3 pixelcnn_prior.py --dataset=mnist --data-dir=~./datasets --k=2 --c=1.0 --hidden-size-vae=56 --model=results/2022-11-29_16-57-47_hyperbolic_data_mnist_model_name_vqvae_k_2_c_1.0_hidden_56/checkpoints/model_7.pth --output-folder=results/2022-11-29_16-57-47_hyperbolic_data_mnist_model_name_vqvae_k_2_c_1.0_hidden_56
        #python3 pixelcnn_prior.py --dataset=mnist --data-dir=~./datasets --k=2 --c=1.0 --hidden-size-vae=56 --model=results/2022-11-29_16-57-47_hyperbolic_data_mnist_model_name_vqvae_k_2_c_1.0_hidden_56/checkpoints/model_7.pth --output-folder=results/2022-11-29_16-57-47_hyperbolic_data_mnist_model_name_vqvae_k_2_c_1.0_hidden_56
