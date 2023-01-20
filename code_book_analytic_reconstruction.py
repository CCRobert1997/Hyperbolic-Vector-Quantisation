import os
import torch
import argparse
import imageio
from torchvision import datasets, transforms

from vq_vae_hyperbolic.auto_encoder import *




# from models.vqvae import VQVAE
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# from torchvision.utils import make_grid
# import numpy as np
# import torch.nn.functional as F
# from scipy.signal import savgol_filter
# from utils import save_image, tensor2im, load_data_and_data_loaders

from dataset_parameter import *







def load_model(model_filename, device):

    path = os.getcwd() + '/results/'
    if torch.cuda.is_available():
        data = torch.load(path + model_filename)
    else:
        data = torch.load(path + model_filename, map_location=lambda storage, loc: storage)

    params = data["hyperparameters"]
    if 'MNIST' in model_filename:
        params['n_dims'] = 1
    else:
        params['n_dims'] = 3

    params['n_embeddings'] = 512
    params['n_hiddens'] = 128
    params['n_residual_hiddens'] = 32
    params['n_residual_layers'] = 2
    params['embedding_dim'] = 64

    # Optimizer
    params['learning_rate'] = 3e-4
    params['n_updates'] = 50000
    params['batch_size'] = 32
    params['loss_type'] = 'OR'

    model = VQVAE(params['n_dims'], params['n_hiddens'], params['n_residual_hiddens'],
                  params['n_residual_layers'], params['n_embeddings'],
                  params['embedding_dim'], params['beta'], params['loss_type'], params['loss_type']).to(device)

    model.load_state_dict(data['model'])
    return model, data, params


def evaluation(data_loader, model, params, args):
    save_path = 'results/' + args.model_name[:-4]
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path + '/train/', exist_ok=True)
    os.makedirs(save_path + '/rec/', exist_ok=True)

    len_data = len(data_loader.dataset)
    print('Evaluating on: ', len_data, 'samples')
    with torch.no_grad():
        recon_loss = 0.0
        histogram = torch.zeros(params['n_embeddings']).to(args.device)
        iterations = int(len(data_loader.dataset) / params['batch_size'])
        iter_loader = iter(data_loader)

        save_data = None
        for i in range(iterations):
            (x, y) = next(iter_loader)
            x = x.to(args.device)
            vq_encoder_output = model.pre_quantization_conv(model.encoder(x))
            z_q, min_encodings, e_indices, _ = model.vector_quantization.inference(vq_encoder_output)
            indices_numpy = e_indices.view(32, 8, 8, 1).cpu().numpy()
            if args.recon:
                if save_data is None:
                    save_data = indices_numpy
                    save_label = y.numpy()
                else:
                    save_data = np.concatenate([save_data, indices_numpy], 0)
                    save_label = np.concatenate([save_label, y.numpy()], 0)

            # import pdb; pdb.set_trace()

            x_recon = model.decoder(z_q)
            if args.save_image:
                for idx in range(x.shape[0]):
                    save_image(tensor2im(x[idx]), save_path + '/train/' + str(i * params['batch_size'] + idx) + '.png')
                # save_image(tensor2im(x_recon[idx]), save_path + '/rec/' + str(i*params['batch_size']+idx)+'.png')
            histogram += min_encodings.sum(0)
            recon_loss += ((x_recon - x) ** 2).mean(3).mean(2).mean(1).sum()
            if (i + 1) % 30 == 0:
                print((i + 1) * x.shape[0], 'sample done!')
    if args.recon:
        np.savez('data/' + args.model_name[:-4], data=save_data, label=save_label)

    recon_loss /= len_data
    e_mean = histogram / (len_data * 64.0)
    perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
    return histogram, recon_loss, perplexity


def _pairwise_distance(x, y, squared=False, eps=1e-16):
    cor_mat = torch.matmul(x, y.t())
    norm_mat = cor_mat.diag()
    distances = norm_mat.unsqueeze(1) - 2 * cor_mat + norm_mat.unsqueeze(0)
    distances = F.relu(distances)

    if not squared:
        mask = torch.eq(distances, 0.0).float()
        distances = distances + mask * eps
        distances = torch.sqrt(distances)
        distances = distances * (1.0 - mask)

    return distances


def get_results(args):
    model, vqvae_data, params = load_model(args.model_name, args.device)
    print('Model loaded')
    training_data, validation_data, training_loader, validation_loader, x_train_var = load_data_and_data_loaders(
        params['dataset'], params['batch_size'])
    print('Data loaded')

    vectors = model.vector_quantization.embedding.weight
    distance = _pairwise_distance(vectors, vectors)
    print('distance computed')

    if args.recon == False:
        histogram, recon_loss, perplexity = evaluation(validation_loader, model, params, args)
    else:
        histogram, recon_loss, perplexity = evaluation(training_loader, model, params, args)

    print(recon_loss, perplexity, distance.mean())

    return model, vqvae_data, params, training_loader, validation_loader, histogram.cpu().numpy()















def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    if image_numpy.shape[2] == 1:
        image_numpy = image_numpy.reshape(image_numpy.shape[0], image_numpy.shape[1])

    imageio.imwrite(image_path, image_numpy)

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        if image_tensor.dim() == 4:
            image_numpy = ((image_tensor[0]+1.0)/2.0).clamp(0,1).cpu().float().numpy()
        else:
            image_numpy = ((image_tensor+1.0)/2.0).clamp(0,1).cpu().float().numpy() # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)






if __name__ == "__main__":
    parser = argparse.ArgumentParser("argument for training")
    #parser.add_argument('--device', type=int, default=0)
    #parser.add_argument('--model_name', type=str, default='WS')
    parser.add_argument('--save_image', type=bool, default=True)
    #parser.add_argument('--recon', type=bool, default=False)

    model_parser = parser.add_argument_group('Model Parameters')
    model_parser.add_argument('--model', default='vae', choices=['vae', 'vqvae'],
                              help='autoencoder variant to use: vae | vqvae')

    model_parser.add_argument('--space', default='hyper', choices=['hyper', 'euc'],
                              help='the embedding space to use: hyper | euc')

    model_parser.add_argument('--model-path',
                              default='/results/2022-10-10_11-17-37_hyperbolic_data_mnist_k_5_c_1.0_hidden_2',
                              help='the model summary folder in results file')

    model_parser.add_argument('--specific-epoch-model-file',
                              default='/checkpoints/model_50.pth',
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
    training_parser.add_argument('--data-dir', default='/media/ssd/Datasets',
                                 help='directory containing the dataset')
    training_parser.add_argument('--dataset', default='cifar10', choices=['mnist', 'cifar10', 'imagenet',
                                                                          'custom'],
                                 help='dataset to use: mnist | cifar10 | imagenet | custom')
    training_parser.add_argument('--no-cuda', action='store_true', default=False,
                                 help='enables CUDA training')




    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    #device_name = f"cuda:{args.device}"
    #device = torch.device(device_name)



    dataset_dir_name = args.dataset if args.dataset != 'custom' else args.dataset_dir_name
    lr = args.lr or default_hyperparams[args.dataset]['lr']
    k = args.k or default_hyperparams[args.dataset]['k']
    c = args.c or default_hyperparams[args.dataset]['c']
    hidden = args.hidden or default_hyperparams[args.dataset]['hidden']
    num_channels = dataset_n_channels[args.dataset]
    transforms_size = args.transforms_size or default_hyperparams[args.dataset]['transforms_size']

    model = models[args.dataset][args.model](hidden, k=k, c=c, kl_coef=args.kl_coef, num_channels=num_channels,
                                             transforms_size=transforms_size)
    if args.cuda:
        model.cuda()
    #    print("k is ")
    #    print(args.k)
    #    print(int(args.k))
    kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}

    dataset_train_dir = os.path.join(args.data_dir, dataset_dir_name)
    dataset_test_dir = os.path.join(args.data_dir, dataset_dir_name)
    if args.dataset in ['imagenet', 'custom']:
        dataset_train_dir = os.path.join(dataset_train_dir, 'train')
        dataset_test_dir = os.path.join(dataset_test_dir, 'val')
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
    #2022-10-07_09-11-04_hyperbolic_data_cifar10_k_128_c_0.01"
    model_file = save_path + args.specific_epoch_model_file
    model = models[args.dataset][args.model](hidden, k=k, c=c, kl_coef=args.kl_coef, num_channels=num_channels,
                                             transforms_size=transforms_size)
    model.load_state_dict(torch.load(model_file))
    if args.cuda:
        model = model.cuda()
    #print(model)

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path + '/train/', exist_ok=True)
    os.makedirs(save_path + '/rec/', exist_ok=True)
    os.makedirs(save_path + '/single_code_vector_reconstruct_result/', exist_ok=True)


    demoreal = None
    with torch.no_grad():
        for i_c in range(k):
            if args.cuda:
                print("cuda is available")
                if (args.model == "vqvae"):
                    #output = model.decoder(model.fp(demoreal))
                    #print(model.emb)
                    #print(model.emb.weight.shape)
                    #print(model.emb.weight.t().shape)
                    #print(model.emb.weight.index_select(0, torch.ones(7*7).int().cuda()).view(1, 7, 7, 5))
                    coder = model.emb.weight.t().index_select(0, i_c*torch.ones(7 * 7).int().cuda()).view(1, hidden, 7, 7)
                    #print(model.emb.weight)
                    #print(model.emb.weight.t().index_select(0, torch.ones(7 * 7).int().cuda()))
                    #print(model.emb.shape)
                    #print(coder)
                    output = model.decoder(model.fp(coder))
                    #print(output.shape)
                    save_image(tensor2im(output), save_path + '/single_code_vector_reconstruct_result/' + str(i_c) + '.png')

                    #argmin.view(-1).view(shifted_shape).permute(0, ctx.dims[-1], *ctx.dims[1:-1])




    #print(model)

    #model, vqvae_data, params, training_loader, validation_loader, histogram = get_results(args)


    # python3 code_book_analytic_reconstruction.py --model=vqvae --k=5 --data-dir=~/.datasets --hidden=2 --dataset=mnist --c=1.0 --save_image=False
    # python3 code_book_analytic_reconstruction.py --model=vqvae --k=128 --data-dir=~/.datasets --dataset=cifar10 --c=1.0 --save_image=False --model-path=/results/2022-10-08_21-23-46_hyperbolic_data_cifar10_k_128_c_1.0_hidden_None  --specific-epoch-model-file=/checkpoints/model_50.pth