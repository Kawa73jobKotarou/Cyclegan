import os.path
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
import random
import util.util as util
import numpy as np
import torch
import nibabel as nib
import cv2

class NiftiDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        self.threshold_upper = opt.threshold_upper
        self.threshold_lower = opt.threshold_lower


    def open_nifti(self, filepath, affine=False):
        data = nib.load(filepath)
        fdata = data.get_fdata()
        if affine:
            affine = data.affine.copy()
            return fdata, affine
        else:
            return fdata


    def randomCrop(self, img, out_size):
        # img kara out_size wo 5% expand shita img kara out_size wo crop
        expand_ratio = 0.1
        edge_len = int(out_size[0] * (1 + expand_ratio))
        center = img.shape[0] // 2
        img = img[center - edge_len // 2: center + edge_len // 2, center - edge_len // 2: center + edge_len // 2]
        cur_h = img.shape[0] - out_size[0]
        cur_w = img.shape[1] - out_size[1]
        h = np.random.randint(0, cur_h)
        w = np.random.randint(0, cur_w)
        img = img[h : h + out_size[0], w : w + out_size[1]]
        return img

    def centerCrop(self, img, out_size):
        center = img.shape[0] // 2
        return img[center - out_size[0] // 2: center + out_size[0] // 2, center - out_size[1] // 2: center + out_size[1] // 2]


    def segmentation(self, tensor, tensor_pil):
        """ segmentate outside of the body contour.

        Parameters:
            tensor (ndarray)
            tensor_pil (jpg, png etc)
        """
        # read image
        thresh = cv2.threshold(tensor_pil, 0, 255, cv2.THRESH_OTSU)[1]
        thresh = cv2.dilate(thresh, np.ones((2, 2)),iterations = 1)

        # extract contours
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

        # pick the contour that has the largest area
        max_cnt = max(contours, key=lambda x: cv2.contourArea(x))

        # creat the mask of the largest area filled with white
        black = np.full_like(tensor, self.threshold_lower)
        mask = cv2.drawContours(black, max_cnt, -1, color=255, thickness=-1)
        cv2.fillPoly(mask, pts =[max_cnt], color=255)

        # synthesize image and mask
        out = np.where(mask==255, tensor, black)

        return out


    def preprocess(self, img, opt, is_CT=False):
        # resize the image to (256, 256). if phase is train, resize the image to (286, 286) and crop randomly to (256, 256)
        if opt.phase == 'train':
            img = cv2.resize(img, dsize=(opt.load_size, opt.load_size))
            # img = self.randomCrop(img, (opt.crop_size, opt.crop_size))
        else: # in test phase, just apply resize
            img = cv2.resize(img, dsize=(opt.load_size, opt.load_size))
            img = self.centerCrop(img, (opt.crop_size, opt.crop_size))
        
        # if the image is CT image, apply segmentation
        # if is_CT:
            # tmp_img = np.uint8((img - np.min(img)) / (np.max(img) - np.min(img)) * 255.0) # temporary image for segmentation
            # img = self.segmentation(img, tmp_img)
            # if np.max(img) > 2000:
            #     img -= 2048 # correct CT value error
            # img = np.clip(img, self.threshold_lower, self.threshold_upper) # eliminate outlier

        # normalize image to [-1, 1]
        # eps = 10 ** -6 # avoid zero division
        # img = (img - np.min(img)) / (np.max(img) - np.min(img) + eps) * 2.0 - 1
        # transform ndarray image to torch.tensor and (256, 256) to (1, 256, 256)
        img = torch.from_numpy(img).unsqueeze(0).float()

        return img


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        index_A = index % self.A_size
        index_B = None
        A_path = self.A_paths[index_A]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
            B_path = self.B_paths[index_B]
            while True:
                patient_num_A = int(A_path.split(sep='/')[4])
                patient_num_B = int(B_path.split(sep='/')[4])
                if patient_num_A != patient_num_B:
                    break
                else:
                    index_B = random.randint(0, self.B_size - 1)
                    B_path = self.B_paths[index_B]
                    patient_num_A = int(A_path.split(sep='/')[4])
                    patient_num_B = int(B_path.split(sep='/')[4])

        B_path = self.B_paths[index_B]

        # open nifti file as numpy.ndarray
        A, affine = self.open_nifti(A_path, affine=True) # use affine matrix in test phase
        B = self.open_nifti(B_path, affine=False)

        A = self.preprocess(A, self.opt, is_CT=False)
        B = self.preprocess(B, self.opt, is_CT=True)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'affine': affine}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
    


