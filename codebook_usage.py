import os
import torch
import argparse
import imageio
from torchvision import datasets, transforms

from vq_vae_hyperbolic.auto_encoder import *
from collections import defaultdict

import matplotlib.pyplot as plt


# from models.vqvae import VQVAE
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# from torchvision.utils import make_grid
# import numpy as np
# import torch.nn.functional as F
# from scipy.signal import savgol_filter
# from utils import save_image, tensor2im, load_data_and_data_loaders

from dataset_parameter import *




def select_n_random(data, labels, n=100):
    '''
    Selects n random datapoints and their corresponding labels from a dataset
    '''
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]





if __name__ == "__main__":
    parser = argparse.ArgumentParser("argument for training")
    # parser.add_argument('--device', type=int, default=0)
    # parser.add_argument('--model_name', type=str, default='WS')
    parser.add_argument('--save_image', type=bool, default=True)
    # parser.add_argument('--recon', type=bool, default=False)

    model_parser = parser.add_argument_group('Model Parameters')

    model_parser.add_argument('--model', default='vae', choices=['vae', 'vqvae', 'pvae'],
                              help='autoencoder variant to use: vae | vqvae | pvae')

    model_parser.add_argument('--model-path',
                              default='/results/2022-10-10_11-17-37_hyperbolic_data_mnist_k_5_c_1.0_hidden_2',
                              help='the model summary folder in results file')

    model_parser.add_argument('--specific-epoch-model-file',
                              default='/checkpoints/model_100.pth',
                              help='the specific model package in model path, usually in /checkpoints dir')

    model_parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                              help='input batch size for training (default: 128)')
    model_parser.add_argument('--hidden', type=int, metavar='N',
                              help='number of hidden channels')
    model_parser.add_argument('--k', '--dict-size', type=int, dest='k', metavar='K',
                              help='number of atoms in dictionary')
    model_parser.add_argument('--c', '--curvature', type=float, dest='c', metavar='C',
                              help='negative curvature of the hyperbolic space')
    model_parser.add_argument('--lr', type=float, default=None,
                              help='learning rate')
    model_parser.add_argument('--kl_coef', type=float, default=0.1,
                              help='kl-divergence coefficient in loss')
    model_parser.add_argument('--transforms-size', '--image-height-and-width', type=int, dest='transforms_size',
                              metavar='transforms_size',
                              help='image height and width, values are unique based on dataset: '
                                   'custom: 256 | imagenet: 256 | cifar10: 32 | mnist: 28 | svhn: 256'
                                   'the hidden layer of VAE has hidden * transforms-size/4 * transforms-size/4 features')

    training_parser = parser.add_argument_group('Training Parameters')
    training_parser.add_argument('--dataset_dir_name', default='',
                                 help='name of the dir containing the dataset if dataset == custom')
    training_parser.add_argument('--train_dir', default='train',
                                 help='name of the dir containing the dataset of train')
    training_parser.add_argument('--val_dir', default='val',
                                 help='name of the dir containing the dataset of validation')
    training_parser.add_argument('--data-dir', default='/media/ssd/Datasets',
                                 help='directory containing the dataset')
    training_parser.add_argument('--dataset', default='cifar10',
                                 choices=['mnist', 'cifar10', 'imagenet', 'svhn', 'custom'],
                                 help='dataset to use: mnist | cifar10 | imagenet | svhn | custom')
    training_parser.add_argument('--no-cuda', action='store_true', default=False,
                                 help='enables CUDA training')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    # device_name = f"cuda:{args.device}"
    # device = torch.device(device_name)

    dataset_dir_name = args.dataset if args.dataset != 'custom' else args.dataset_dir_name
    lr = args.lr or default_hyperparams[args.dataset]['lr']
    k = args.k or default_hyperparams[args.dataset]['k']
    c = args.c or default_hyperparams[args.dataset]['c']
    hidden = args.hidden or default_hyperparams[args.dataset]['hidden']
    num_channels = dataset_n_channels[args.dataset]
    transforms_size = args.transforms_size or default_hyperparams[args.dataset]['transforms_size']

    print("parameter loaded")
    model = models[args.dataset][args.model](hidden, k=k, c=c, kl_coef=args.kl_coef, num_channels=num_channels,
                                             transforms_size=transforms_size)
    print("model loaded")
    if args.cuda:
        model.cuda()
    #    print("k is ")
    #    print(args.k)
    #    print(int(args.k))
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

    save_path = os.getcwd() + args.model_path
    # 2022-10-07_09-11-04_hyperbolic_data_cifar10_k_128_c_0.01"
    model_file = save_path + args.specific_epoch_model_file
    model = models[args.dataset][args.model](hidden, k=k, c=c, kl_coef=args.kl_coef, num_channels=num_channels,
                                             transforms_size=transforms_size)
    model.load_state_dict(torch.load(model_file))
    if args.cuda:
        model = model.cuda()
    # print(model)

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path + '/train/', exist_ok=True)
    os.makedirs(save_path + '/rec/', exist_ok=True)
    os.makedirs(save_path + '/single_code_vector_reconstruct_result/', exist_ok=True)

    demoreal = None

    print(model.emb.weight.shape)  #one colume is one code vector
    print(model.emb.weight[:, 0])
    emb_count = defaultdict(int)
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            if args.cuda:
                data = data.cuda()
            # print(model(data[0][None,:,:,:])[1].shape)
            # print(model.emb(model(data[0][None, :, :, :])[1])[0].shape)
            # if (args.model == "vqvae"):
            #    demoreal = model.emb(model(data[0][None,:,:,:])[1], weight_sg=True)


            argmin = model.encode_index(data)
            argminshape = argmin.shape
            for sample in argmin:
                for i in range(argminshape[1]):
                    for j in range(argminshape[2]):
                        emb_count[sample[i, j].item()]+=1
                        #print(sample[i, j].item())

            #print(argmin)
            # outputs = model(data)
            """
            if (i * args.batch_size >= (5000 + args.batch_size)):
                break
            if args.save_image:
                for idx in range(data.shape[0]):
                    save_image(tensor2im(data[idx]), save_path + '/train/' + str(i * args.batch_size + idx) + '.png')
                    save_image(tensor2im(outputs[0][idx]),
                               save_path + '/rec/' + str(i * args.batch_size + idx) + '.png')
            """

    print(emb_count)

    #objects = ('Python', 'C++', 'Java', 'Perl', 'Scala', 'Lisp')
    x_pos = np.arange(args.k)
    emb_count_plot = [0]*args.k

    for i in emb_count.keys():
        emb_count_plot[i] = emb_count[i]

    plt.bar(x_pos, emb_count_plot, align='center', alpha=0.99)
    #plt.xticks(x_pos, objects)
    plt.ylabel(args.model)
    plt.title('Codebook size ' + str(args.k))
    plt.savefig(save_path + "/codebookvusage.png")
    #plt.show()



    #for images, labels in test_loader:
    #    print(labels)
    #images, labels = select_n_random(trainset.data, trainset.targets)

    # get the class labels for each image
    #class_labels = [classes[lab] for lab in labels]


    #python3 codebook_usage.py --model=vqvae --data-dir=~./datasets --k=512 --hidden=64 --c=1.0 --dataset=cifar10 --model-path=/results/2022-12-20_13-55-42_hyperbolic_data_cifar10_model_name_vqvae_k_512_c_1.0_hidden_64 --specific-epoch-model-file=/checkpoints/model_20.pth
