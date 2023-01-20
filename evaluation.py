import os
import os.path
import glob
import shutil
#import lpips
import numpy as np
import argparse
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


parser = argparse.ArgumentParser(description='Image quality evaluations on the dataset')
parser.add_argument('--gt_path', type=str, default='results/model_file_dir/train', help='path to original gt data')
parser.add_argument('--g_path', type=str, default='results/model_file_dir/rec', help='path to the generated data')
parser.add_argument('--save_path', type=str, default=None, help='path to save the best results')
parser.add_argument('--center', action='store_true',
                    help='only calculate the center masked regions for the image quality')
parser.add_argument('--num_test', type=int, default=100, help='how many examples to load for testing')
parser.add_argument('--device', type=int, default=0)


args = parser.parse_args()
device_name = f"cuda:{args.device}"
device = torch.device(device_name)
#lpips_alex = lpips.LPIPS(net='alex').to(device)




def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(path_files):
    if path_files.find('.txt') != -1:
        paths, size = make_dataset_txt(path_files)
    else:
        paths, size = make_dataset_dir(path_files)

    return sorted(paths), size


def make_dataset_txt(files):
    """
    :param path_files: the path of txt file that store the image paths
    :return: image paths and sizes
    """
    img_paths = []

    with open(files) as f:
        paths = f.readlines()

    for path in paths:
        path = path.strip()
        if is_image_file(path) and os.path.exists(path):
            img_paths.append(path)

    return img_paths, len(img_paths)


def make_dataset_dir(dir):
    """
    :param dir: directory paths that store the image
    :return: image paths and sizes
    """
    img_paths = []

    print(os.path.isdir("results/"))

    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in os.walk(dir):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                img_paths.append(path)

    return img_paths, len(img_paths)




def calculate_score(img_gt, img_test):
    """
    function to calculate the image quality score
    :param img_gt: original image
    :param img_test: generated image
    :return: mae, ssim, psnr
    """

    l1loss = np.mean(np.abs(img_gt-img_test))

    psnr_score = psnr(img_gt, img_test, data_range=1)

    ssim_score = ssim(img_gt, img_test, multichannel=True, data_range=1, win_size=11)

    #lpips_dis = lpips_alex(torch.from_numpy(img_gt).permute(2, 0, 1).to(device), torch.from_numpy(img_test).permute(2, 0, 1).to(device), normalize=True)

    return l1loss, ssim_score, psnr_score#, lpips_dis.data.cpu().numpy().item()

if __name__ == '__main__':
    gt_paths, gt_size = make_dataset(args.gt_path)
    g_paths, g_size = make_dataset(args.g_path)

    l1losses = []
    ssims = []
    psnrs = []
    lpipses = []
    size = args.num_test if args.num_test > 0 else gt_size
    print('Evaluating on ', size, 'samples')
    for i in range(size):
        gt_img = Image.open(gt_paths[i + 0*2000]).resize([256, 256]).convert('RGB')
        gt_numpy = np.array(gt_img).astype(np.float32) / 255.0

        l1loss_sample = 1000
        ssim_sample = 0
        psnr_sample = 0
        lpips_sample = 1000

        name = gt_paths[i + 0*2000].split('/')[-1].split(".")[0] + "*"
        g_paths = sorted(glob.glob(os.path.join(args.g_path, name)))
        num_files = len(g_paths)

        for j in range(num_files):
            if (j+1) % 100 == 0:
                print(j)
            index = j
            #import pdb; pdb.set_trace()

            try:
                g_img = Image.open(g_paths[j]).resize([256, 256]).convert('RGB')
                g_numpy = np.array(g_img).astype(np.float32) / 255.0
                if args.center:
                    gt_numpy = gt_numpy[64:192, 64:192, :]
                    g_numpy = g_numpy[64:192, 64:192, :]
                #l1loss, ssim_score, psnr_score, lpips_score = calculate_score(gt_numpy, g_numpy)
                l1loss, ssim_score, psnr_score = calculate_score(gt_numpy, g_numpy)
                # if l1loss - ssim_score - psnr_score + lpips_score < l1loss_sample - ssim_sample - psnr_sample + lpips_sample:
                #     l1loss_sample, ssim_sample, psnr_sample, lpips_sample = l1loss, ssim_score, psnr_score, lpips_score
                #     best_index = index

                if l1loss - ssim_score - psnr_score < l1loss_sample - ssim_sample - psnr_sample:
                    l1loss_sample, ssim_sample, psnr_sample = l1loss, ssim_score, psnr_score
                    best_index = index
            except:
                print(g_paths[index])

        if l1loss_sample != 1000 and ssim_sample !=0 and psnr_sample != 0:
            #print(g_paths[best_index])
            #print(l1loss_sample, ssim_sample, psnr_sample, lpips_sample)
            #print(l1loss_sample, ssim_sample, psnr_sample)
            l1losses.append(l1loss_sample)
            ssims.append(ssim_sample)
            psnrs.append(psnr_sample)
            #lpipses.append(lpips_sample)

            if args.save_path is not None:
                os.makedirs(args.save_path, exist_ok=True)
                shutil.copy(g_paths[best_index], args.save_path)

    # print('{:>10},{:>10},{:>10},{:>10}'.format('l1loss', 'SSIM', 'PSNR', 'LPIPS'))
    # print('{:10.4f},{:10.4f},{:10.4f},{:10.4f}'.format(np.mean(l1losses), np.mean(ssims), np.mean(psnrs), np.mean(lpipses)))
    # print('{:10.4f},{:10.4f},{:10.4f},{:10.4f}'.format(np.var(l1losses), np.var(ssims), np.var(psnrs), np.var(lpipses)))

    print('{:>10},{:>10},{:>10}'.format('l1loss', 'SSIM', 'PSNR'))
    print('{:10.4f},{:10.4f},{:10.4f}'.format(np.mean(l1losses), np.mean(ssims), np.mean(psnrs)))
    print('{:10.4f},{:10.4f},{:10.4f}'.format(np.var(l1losses), np.var(ssims), np.var(psnrs)))
    if args.save_path is not None:
        os.makedirs(args.save_path, exist_ok=True)
        with open(args.save_path + '/numerical_results.txt', 'w') as f:
            f.write('{:>10},{:>10},{:>10}\n'.format('l1loss', 'SSIM', 'PSNR'))
            f.write('{:10.4f},{:10.4f},{:10.4f}\n'.format(np.mean(l1losses), np.mean(ssims), np.mean(psnrs)))
            f.write('{:10.4f},{:10.4f},{:10.4f}\n'.format(np.var(l1losses), np.var(ssims), np.var(psnrs)))

    #python3 evaluation.py --gt_path=results/2022-12-15_15-56-07_hyperbolic_data_cifar10_model_name_pvae_k_128_c_1.0_hidden_64/train --g_path=results/2022-12-15_15-56-07_hyperbolic_data_cifar10_model_name_pvae_k_128_c_1.0_hidden_64/rec --save_path=results/2022-12-15_15-56-07_hyperbolic_data_cifar10_model_name_pvae_k_128_c_1.0_hidden_64/numersults_and_recdemo
    #pip3 install pytorch-fid
    #python3 -m pytorch_fid results/2022-12-15_15-56-07_hyperbolic_data_cifar10_model_name_pvae_k_128_c_1.0_hidden_64/train results/2022-12-15_15-56-07_hyperbolic_data_cifar10_model_name_pvae_k_128_c_1.0_hidden_64/rec

