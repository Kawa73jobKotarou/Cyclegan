import os.path
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
import random
import util.util as util
import numpy as np
import torch
import pydicom
import cv2
import pdb
import matplotlib.pyplot as plt
import re

def get_min_max_values(opt, dir_B):
    # 辞書を初期化
    min_max_dict = {}

    for subdir in os.listdir(dir_B):
        subdir_path = os.path.join(dir_B, subdir)

        # フォルダ名が「XX_YY」形式のときのみ処理
        if os.path.isdir(subdir_path) and re.match(r'^\d{2}_\d{2}$', subdir):
            min_value = float('inf')  # 初期値を無限大に設定
            max_value = float('-inf')  # 初期値を負の無限大に設定
            
            # サブフォルダ内のすべてのDICOMファイルを処理
            for fname in os.listdir(subdir_path):
                if fname.endswith('.dcm'):  # DICOMファイルの拡張子を確認
                    # DICOMファイルを読み込む
                    dcm_path = os.path.join(subdir_path, fname)
                    dcm_data = pydicom.dcmread(dcm_path)
                    img_array = dcm_data.pixel_array  # ピクセルデータを取得
                    img_array = cv2.resize(img_array, dsize=(opt.load_size, opt.load_size))
                    img_array = np.clip(img_array, opt.threshold_lower, opt.threshold_upper)
                    # 現在の画像の最小値と最大値を更新
                    min_value = min(min_value, np.min(img_array))
                    max_value = max(max_value, np.max(img_array))

            # 辞書にサブフォルダ名と最小・最大値を追加
            min_max_dict[subdir] = {'min': min_value, 'max': max_value}

    return min_max_dict

def decision_crop_start(img_path, crop_patch_size, threshold=0.5):

    dcm = pydicom.dcmread(img_path)
    img = dcm.pixel_array
    h, w = img.shape[:2]

    if h < crop_patch_size or w < crop_patch_size:
        raise ValueError("画像サイズがパッチサイズより小さいため、抽出できません。")

    while True:
        x = random.randint(0, w - crop_patch_size)
        y = random.randint(0, h - crop_patch_size)

        patch = img[y:y+crop_patch_size, x:x+crop_patch_size]
        zero_ratio = np.mean(patch == 0)

        if zero_ratio < threshold:
            break
    return x,y

class DicomExtentionDataset(BaseDataset):
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

        self.min_max_dict = get_min_max_values(opt, self.dir_B)
        # A_paths をロードし、05_?? フォルダのパスをフィルタリング
        all_A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.A_paths = [p for p in all_A_paths if not self._is_05_xx_folder(p)]
        # B_paths をロードし、05_?? フォルダのパスをフィルタリング
        all_B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        self.B_paths = [p for p in all_B_paths if not self._is_05_xx_folder(p)]
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        self.threshold_upper = opt.threshold_upper
        self.threshold_lower = opt.threshold_lower

        self.input_nc_G = opt.input_nc
        self.original_size = opt.original_size
        self.make_patch = opt.make_patch

    def _is_05_xx_folder(self, path_str):
        # パスに '/05_XX/' のパターンを検索
        # 例えば、'/path/to/data/trainA/05_01/image.dcm' のようなパスに対応
        match = re.search(r'/(05_\d{2})/', path_str)
        if match:
            return True
        return False

    def set_make_patch(self, flag: bool):
        self.make_patch = flag
        


    def randomCrop(self, img, out_size):
        cur_h = img.shape[0] - out_size[0]
        cur_w = img.shape[1] - out_size[1]
        h = np.random.randint(0, cur_h)
        w = np.random.randint(0, cur_w)
        img = img[h : h + out_size[0], w : w + out_size[1]]
        return img


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

    def preprocess(self, img, crop_start_width, crop_start_height, opt, is_CT=False, min_max = None):
        # resize the image to (256, 256). if phase is train, resize the image to (286, 286) and crop randomly to (256, 256)
        if opt.phase == 'train':
            if self.make_patch:
                img = img[crop_start_height : crop_start_height + opt.crop_patch_size, crop_start_width : crop_start_width + opt.crop_patch_size]
            else:
                img = cv2.resize(img, dsize=(opt.load_size, opt.load_size))
            # img = self.randomCrop(img, (256, 256))
        else: # in test phase, just apply resize
            img = cv2.resize(img, dsize=(opt.load_size, opt.load_size))
        
        # if the image is CT image, apply segmentation
        if is_CT:
            # tmp_img = np.uint8((img - np.min(img)) / (np.max(img) - np.min(img)) * 255.0) # temporary image for segmentation
            # img = self.segmentation(img, tmp_img)
            # if np.max(img) > 2000:
            #     img -= 2048 # correct CT value error
            img = np.clip(img, self.threshold_lower, self.threshold_upper) # eliminate outlier

        eps = 10 ** -6 # avoid zero division
        if is_CT:
            img = (img - min_max["min"]) / (min_max["max"] - min_max["min"] + eps) * 2.0 - 1
        else:
            # normalize image to [-1, 1]
            img = (img - np.min(img)) / (np.max(img) - np.min(img) + eps) * 2.0 - 1
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
        #  ランダムな変換を決定
        do_flip_lr = random.random() > 0.5  # 左右反転
        do_flip_ud = random.random() > 0.5  # 上下反転
        rotation = random.choice([0, 90, 180, 270])  # 回転角度        

        def transform(image, do_flip_lr, do_flip_ud, rotation, target_channels):
            """指定された変換を画像に適用"""
            # PyTorch Tensor なら NumPy に変換
            if isinstance(image, torch.Tensor):
                image = image.numpy()

            # (H, W) → (1, H, W) に変換 (もし image が (H, W) の場合)
            if image.ndim == 2:
                image = np.expand_dims(image, axis=0)  # (H, W) → (1, H, W)

            # NumPy → PyTorch Tensor に変換
            image = torch.from_numpy(image).float()

            # 変換適用
            if do_flip_lr:
                image = torch.flip(image, dims=[2])  # 左右反転
            if do_flip_ud:
                image = torch.flip(image, dims=[1])  # 上下反転
            if rotation != 0:
                k = rotation // 90
                image = torch.rot90(image, k, dims=[1, 2])  # PyTorch の rot90 を使用

            # チャンネル数をターゲットに合わせる
            if image.shape[0] != target_channels:
                image = image.repeat(target_channels, 1, 1)  # チャンネル数を調整

            return image

        if self.input_nc_G == 1:
            A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
            if self.opt.serial_batches:   # make sure index is within then range
                index_B = index % self.B_size
            else:   # randomize the index for domain B to avoid fixed pairs.
                index_B = random.randint(0, self.B_size - 1)
            B_path = self.B_paths[index_B]

            if self.opt.phase == "train":
                # 抽出するための正規表現パターン
                pattern = r'trainB/(\d{2}_\d{2})/'
            else:
                pattern = r'testB/(\d{2}_\d{2})/'
            # 各パスからパターンにマッチする部分を抽出
            extracted = re.search(pattern, B_path)
            match = re.search(pattern, B_path)
            extracted = match.group(1)
            # print("extracted"+extracted+" "+B_path)
            if self.opt.phase == "train" and self.make_patch:
                crop_start_width, crop_start_height = decision_crop_start(A_path, self.opt.crop_patch_size)
            else:
                crop_start_width, crop_start_height = 0, 0

            # open dicom file as numpy.ndarray
            A_dcm = pydicom.dcmread(A_path)
            B_dcm  =pydicom.dcmread(B_path)
            A = self.preprocess(A_dcm.pixel_array, crop_start_width, crop_start_height, self.opt, is_CT=False)
            B = self.preprocess(B_dcm.pixel_array, crop_start_width, crop_start_height, self.opt, is_CT=True, min_max = self.min_max_dict[extracted])
            if self.opt.phase == "train":
                do_flip_lr = random.random() > 0.5
                do_flip_ud = random.random() > 0.5
                rotation = random.choice([0, 90, 180, 270])
                A = transform(A, do_flip_lr, do_flip_ud, rotation, self.input_nc_G)
                B = transform(B, do_flip_lr, do_flip_ud, rotation, self.input_nc_G)

            # Determine bone presence
            # Assuming mask paths follow a similar structure to A_path/B_path
            # You'll need to adjust the mask_path generation based on your actual file structure
            bone_A_presence = 0 # Default to 2 (no bone)
            bone_B_presence = 0 # Default to 2 (no bone)

            # Example: Constructing mask path. Adjust this according to your dataset.
            # Assuming mask files are in a directory like 'masks/Bone/' and have the same name as the original images.
            # You might need to parse A_path/B_path to get the base filename.
            try:
                # For A (non-CT)
                A_filename = A_path.split('/')[-1]
                A_mask_path = A_path.replace(A_filename, f'masks/Bone/{A_filename.split(".")[0]}.png') # Adjust suffix if not .png
                A_mask = cv2.imread(A_mask_path, cv2.IMREAD_GRAYSCALE) # Assuming bone masks are images
                if A_mask is not None and np.any(A_mask == 1): # Check if any pixel is 1
                    bone_A_presence = 1
            except Exception as e:
                print(f"Could not load or process A bone mask for {A_path}: {e}")

            try:
                # For B (CT)
                B_filename = B_path.split('/')[-1]
                B_mask_path = B_path.replace(B_filename, f'masks/Bone/{B_filename.split(".")[0]}.png') # Adjust suffix if not .png
                B_mask = cv2.imread(B_mask_path, cv2.IMREAD_GRAYSCALE) # Assuming bone masks are images
                if B_mask is not None and np.any(B_mask == 1): # Check if any pixel is 1
                    bone_B_presence = 1
            except Exception as e:
                print(f"Could not load or process B bone mask for {B_path}: {e}")

            return {
                'A': A,
                'B': B,
                'A_paths': A_path,
                'B_paths': B_path,
                'A_patch_coords': [crop_start_width, crop_start_height],
                'B_patch_coords': [crop_start_width, crop_start_height], # Assuming same patch for B
                'patch_size': self.opt.crop_patch_size,
                'A_bone_presence': bone_A_presence,
                'B_bone_presence': bone_B_presence
            }
        else:
            # A_paths と B_paths のインデックス計算
            def get_slice_paths(paths, index, slice_range):
                slice_paths = []
                origin_number = int(paths[index].split('/')[-1].split('_')[-1].split('.')[0])
                upper = index
                lower = index
                for offset in range(slice_range+1):
                    if offset == 0:
                        slice_paths.append(paths[index])
                    else:
                        slice_index_minus =index-offset
                        slice_index_plus = index+offset
                        if slice_index_minus < 0:  # 境界処理: 最初のスライスを使う
                            slice_index_minus = 0
                        elif slice_index_plus >= len(paths):  # 境界処理: 最後のスライスを使う
                            slice_index_plus = len(paths) - 1

                        target_number = int(paths[slice_index_minus].split('/')[-1].split('_')[-1].split('.')[0])
                        if origin_number < target_number:
                            slice_index_minus = lower
                        else:
                            lower = slice_index_minus
                        slice_paths.insert(0,paths[int(slice_index_minus)])

                        target_number = int(paths[slice_index_plus].split('/')[-1].split('_')[-1].split('.')[0])
                        if origin_number > target_number:
                            slice_index_plus = upper
                        else:
                            upper = slice_index_plus
                        slice_paths.append(paths[int(slice_index_plus)])
                return slice_paths

            # A_paths と B_paths を取得
            A_slice_paths = get_slice_paths(self.A_paths, index % self.A_size, int((self.input_nc_G-1)/2))
            if self.opt.serial_batches:   # make sure index is within then range
                index_B = index % self.B_size
            else:   # randomize the index for domain B to avoid fixed pairs.
                index_B = random.randint(0, self.B_size - 1)
            B_path = self.B_paths[index_B] # This is just one path for B in this section, will need to be adapted for multiple B slices.

            if self.opt.phase == "train" and self.make_patch:
                crop_start_width, crop_start_height = decision_crop_start(A_slice_paths[int((self.input_nc_G-1)/2)], self.opt.crop_patch_size)
            else:
                crop_start_width, crop_start_height = 0, 0

            # A スライスを読み込んでテンソル化
            A_slices = []
            A_bone_presence_list = [] # To store bone presence for each A slice
            for path in A_slice_paths:
                A_dcm = pydicom.dcmread(path)
                A_img = self.preprocess(A_dcm.pixel_array, crop_start_width, crop_start_height, self.opt, is_CT=False)
                A_slices.append(A_img)

                # Determine bone presence for each A slice
                bone_A_presence = 0 
                if self.opt.phase == "train":
                    A_filename = path.split('/')[-1]
                    A_mask_path = path.replace(A_filename, f'masks/Bone/{A_filename.split(".")[0]}.png')
                    A_mask = cv2.imread(A_mask_path, cv2.IMREAD_GRAYSCALE)
                    A_mask = self.preprocess(A_mask, crop_start_width, crop_start_height, self.opt, is_CT=False)
                    if A_mask.max() == 1.0: # 画素値1が骨を示す場合
                        bone_A_presence = 1 # 骨あり
                A_bone_presence_list.append(bone_A_presence)

            A = torch.cat(A_slices, dim=0)  # (5, 256, 256)

            # B スライスを読み込んでテンソル化
            if self.opt.phase == "train":
                # 抽出するための正規表現パターン
                pattern = r'trainB/(\d{2}_\d{2})/'
            else:
                pattern = r'testB/(\d{2}_\d{2})/'
            # B スライスもAと同様に複数スライスを使う
            B_slice_paths = get_slice_paths(self.B_paths, index_B, int((self.input_nc_G-1)/2))
            # 各Bスライスを読み込んでテンソル化
            B_slices = []
            B_bone_presence_list = [] # To store bone presence for each B slice
            for path in B_slice_paths:
                match = re.search(pattern, path)
                extracted = match.group(1)
                B_dcm = pydicom.dcmread(path)
                B_img = self.preprocess(B_dcm.pixel_array, crop_start_width, crop_start_height, self.opt, is_CT=True, min_max=self.min_max_dict[extracted])
                B_slices.append(B_img)
                # Determine bone presence for each B slice
                bone_B_presence = 0 
                if self.opt.phase == "train":
                    B_filename = path.split('/')[-1]
                    B_mask_path = path.replace(B_filename, f'masks/Bone/{B_filename.split(".")[0]}.png')
                    B_mask = cv2.imread(B_mask_path, cv2.IMREAD_GRAYSCALE)
                    B_mask = self.preprocess(B_mask, crop_start_width, crop_start_height, self.opt, is_CT=False)
                    if B_mask.max() == 1.0: # 画素値1が骨を示す場合
                        bone_B_presence = 1
                B_bone_presence_list.append(bone_B_presence)

            B = torch.cat(B_slices, dim=0)  # (5, 256, 256) など

            if self.opt.phase == "train":
                do_flip_lr = random.random() > 0.5
                do_flip_ud = random.random() > 0.5
                rotation = random.choice([0, 90, 180, 270])
                A = [transform(img, do_flip_lr, do_flip_ud, rotation, 1) for img in A]
                A = torch.cat(A, dim=0)  # [5, 1, 128, 128]
                B = [transform(img, do_flip_lr, do_flip_ud, rotation, 1) for img in B]
                B = torch.cat(B, dim=0)  # [5, 1, H, W]

            return {
                'A': A,
                'B': B,
                'A_paths': A_slice_paths,
                'B_paths': B_slice_paths,
                'A_patch_coords': [crop_start_width, crop_start_height],
                'B_patch_coords': [crop_start_width, crop_start_height], # Assuming same patch for B
                'patch_size': self.opt.crop_patch_size,
                'A_bone_presence': A_bone_presence_list, # List of bone presence for each slice
                'B_bone_presence': B_bone_presence_list  # List of bone presence for each slice
            }


    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
    


