import os
import cv2
import math
import numpy as np
from PIL import Image
from scipy import ndimage
from torchvision.utils import make_grid


def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / \
        (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(
            math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # cv2.imwrite(img_path, img)


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def calculate_uciqe(img):
    '''Calculate UCIQE for an RGB uint8 image.'''
    img = img.astype(np.float32) / 255.0
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]

    chroma = np.sqrt(a * a + b * b)
    sigma_c = np.std(chroma)
    contrast_l = np.percentile(l, 99) - np.percentile(l, 1)
    saturation = chroma / np.sqrt(chroma * chroma + l * l + 1e-12)
    mean_s = np.mean(saturation)

    return 0.4680 * sigma_c + 0.2745 * contrast_l + 0.2576 * mean_s


def _trimmed_mean_and_var(x, trim_ratio=0.1):
    x = np.sort(x.reshape(-1).astype(np.float64))
    n = len(x)
    trim = int(n * trim_ratio)
    if trim * 2 >= n:
        trimmed = x
    else:
        trimmed = x[trim:n - trim]
    return np.mean(trimmed), np.var(trimmed)


def _eme(channel, block_size=8):
    channel = channel.astype(np.float64)
    h, w = channel.shape
    score = 0.0
    count = 0
    eps = 1e-6

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = channel[y:min(y + block_size, h), x:min(x + block_size, w)]
            block_min = np.min(block)
            block_max = np.max(block)
            if block_max > eps and block_min >= 0:
                score += np.log((block_max + eps) / (block_min + eps))
                count += 1

    if count == 0:
        return 0.0
    return 20.0 * score / count


def _uicm(img):
    img = img.astype(np.float64)
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    rg = r - g
    yb = (r + g) / 2.0 - b

    mu_rg, var_rg = _trimmed_mean_and_var(rg)
    mu_yb, var_yb = _trimmed_mean_and_var(yb)
    mu = np.sqrt(mu_rg * mu_rg + mu_yb * mu_yb)
    sigma = np.sqrt(var_rg + var_yb)
    return -0.0268 * mu + 0.1586 * sigma


def _uism(img):
    img = img.astype(np.float64)
    weights = [0.299, 0.587, 0.114]
    score = 0.0
    for i, weight in enumerate(weights):
        channel = img[:, :, i]
        sx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
        edge = np.sqrt(sx * sx + sy * sy)
        score += weight * _eme(edge * channel)
    return score


def _uiconm(img):
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return _eme(gray)


def calculate_uiqm(img):
    '''Calculate UIQM for an RGB uint8 image.'''
    return 0.0282 * _uicm(img) + 0.2953 * _uism(img) + 3.5753 * _uiconm(img)
