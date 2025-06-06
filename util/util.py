"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import nibabel as nib
import pydicom
import matplotlib.pyplot as plt
import pdb

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
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = np.transpose(image_numpy, (1, 2, 0))  # transpose (C, H, W) to (H, W, C)
        image_numpy = (image_numpy + 1) / 2.0 * 255.0# scale [-1, 1] to [0, 255]
        # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return np.rot90(image_numpy.astype(imtype))


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)

def save_dicom_image(image_tensor, image_path, opt, B_path):
    """save a torch.tensor image to the disk as dicom
    Parameters:
        image_tensor (torch.tensor) -- input tensor array
        image_path (str)            -- the path of the image
    """
    im_np = image_tensor.to('cpu').detach().numpy().copy()
    im_np = im_np[:,int((opt.input_nc-1)/2)]
    # revert [-1, 1] to [opt.threshold_lower, opt.threshold_upper]
    im_np = (im_np + 1) / 2.0 * (opt.threshold_upper - opt.threshold_lower) + opt.threshold_lower
    # CT_dcm = pydicom.read_file(B_path)
    # CT_dcm.PixelData = im_np.tobytes()
    CT_dcm = pydicom.dcmread(B_path)
    CT_dcm.PixelData = im_np.astype(np.int16).tobytes()
    CT_dcm.Rows = im_np.shape[1]
    CT_dcm.Columns = im_np.shape[2]
    if "MRCAT" in B_path:
        rescale = CT_dcm.RescaleIntercept
        # PixelData をバイナリから数値配列に変換
        pixel_array = np.frombuffer(CT_dcm.PixelData, dtype=np.int16).reshape(CT_dcm.Rows, CT_dcm.Columns)
        # Rescale を適用 (整数型で加算)
        pixel_array = (pixel_array - int(rescale)).astype(np.int16)  # データ型を維持するために int に変換
        # 再びバイナリ形式に変換して PixelData に保存
        CT_dcm.PixelData = pixel_array.astype(np.int16).tobytes()
    CT_dcm.save_as(image_path)

def save_nifti_image(image_tensor, image_path, opt, affine):
    """save a torch.tensor image to the disk as nifti

    Parameters:
        image_tensor (torch.tensor) -- input tensor array
        image_path (str)            -- the path of the image 
    """

    im_np = image_tensor.to('cpu').detach().numpy().copy()
    # revert [-1, 1] to [opt.threshold_lower, opt.threshold_upper]
    im_np = (im_np + 1) / 2.0 * (opt.threshold_upper - opt.threshold_lower) + opt.threshold_lower
    im = nib.Nifti1Image(im_np.squeeze(), affine=affine)
    nib.save(im, image_path)
    

def save_eps_image(image_tensor, image_path, opt):
    im_np = tensor2im(image_tensor)
    im_np = im_np[:, :, :int((opt.input_nc-1)/2)]
    image_pil = Image.fromarray(im_np)
    image_pil = image_pil.convert("L")
    image_pil.save(image_path)

def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    if image_numpy.ndim == 2:
        h, w = image_numpy.shape
        c = 1
    elif image_numpy.ndim == 3:
        h, w, c = image_numpy.shape
    else:
        raise ValueError(f"Unexpected image shape: {image_numpy.shape}")


    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
