import os
import sys
import time
import torch
import torch.nn as nn

#from utils.config.read import readConfig
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from torchvision import transforms
from random import randint
# from utils.preprocessing.normalisation import standardization_intensity_normalization
from scipy import ndimage

from torch.utils.data import DataLoader
import glob
import SimpleITK as sitk
import random

args = {


    'train_patch_size_x': 96,
    'train_patch_size_y': 96,

    'train_patch_size_z': 96,



}
def standardization_intensity_normalization(dataset, dtype):
    mean = dataset.mean()
    std  = dataset.std()
    return ((dataset - mean) / std).astype(dtype)

def extractPatch(d, patch_size_x, patch_size_y, patch_size_z, x, y, z):
    patch = d[x - patch_size_x // 2:x + patch_size_x // 2, y - patch_size_y // 2:y + patch_size_y // 2,
            z - patch_size_z // 2:z + patch_size_z // 2]
    return patch


# dataset_size = (args['train'], args['valid'], args['test'])
patchs_size = (args["train_patch_size_x"], args["train_patch_size_y"], args['train_patch_size_z'])


def load_dataset(root_dir, train=True):
    images = []
    groundtruth = []
    if train:
        sub_dir = 'training'
    else:
        sub_dir = 'test'

    images_path = os.path.join(root_dir, sub_dir, 'HalfSSimage')
    groundtruth_path = os.path.join(root_dir, sub_dir, 'label')

    for file in glob.glob(os.path.join(images_path, '*.nii')):
        image_name = os.path.basename(file)[:-10]
        groundtruth_name = image_name + '.nii'

        images.append(file)
        groundtruth.append(os.path.join(groundtruth_path, groundtruth_name))

    return images, groundtruth


def RandomPatchCrop(image, label, patch_in_size, patch_gd_size):  # patch_in_size：[64,64,64]
    #
    # patchs_in = np.zeros((patch_in_size[0], patch_in_size[1], patch_in_size[2]),
    #                      dtype=image.dtype)
    # patchs_gd = np.zeros((patch_gd_size[0], patch_gd_size[1], patch_gd_size[2]),
    #                      dtype=label.dtype)

    if (patch_in_size[0] % patch_gd_size[0] != 0 or patch_in_size[1] % patch_gd_size[1] != 0 or patch_in_size[2] %
            patch_gd_size[2] != 0):
        sys.exit("ERROR : randomPatchsAugmented patchs size error 1")

    if (patch_in_size[0] < patch_gd_size[0] or patch_in_size[1] < patch_gd_size[1] or patch_in_size[2] <
            patch_gd_size[2]):
        sys.exit("ERROR : randomPatchsAugmented patchs size error 2")

    x = randint(patchs_size[0] // 2,
                image.shape[0] - patch_in_size[0] // 2)  # shape:38*224*224*96 x:randint(0,224-64) 0到160中的一个整型随机数
    y = randint(patchs_size[1] // 2, image.shape[1] - patch_in_size[1] // 2)  # y 和 x一样
    z = randint(patchs_size[2] // 2, image.shape[2] - patch_in_size[2] // 2)  # 96-64  0到32中的一个整型随机数

    r0 = randint(0, 3)
    r1 = randint(0, 3)
    r2 = randint(0, 3)
    patchs_in = extractPatch(image, patch_in_size[0], patch_in_size[1], patch_in_size[2], x, y, z)
    patchs_gd = extractPatch(label, patch_gd_size[0], patch_gd_size[1], patch_gd_size[2], x, y, z)

    patchs_in = np.rot90(patchs_in, r0, (0, 1))  # 旋转操作
    patchs_in = np.rot90(patchs_in, r1, (1, 2))
    patchs_in = np.rot90(patchs_in, r2, (2, 0))

    patchs_gd = np.rot90(patchs_gd, r0, (0, 1))
    patchs_gd = np.rot90(patchs_gd, r1, (1, 2))
    patchs_gd = np.rot90(patchs_gd, r2, (2, 0))

    return patchs_in, patchs_gd


class Data(Dataset):
    def __init__(self,
                 root_dir,
                 train=True,
                 rotate=40,
                 flip=True,
                 random_crop=True,
                 scale1=512):

        self.root_dir = root_dir
        self.train = train
        self.rotate = rotate
        self.flip = flip
        self.random_crop = random_crop
        self.transform = transforms.ToTensor()
        self.resize = scale1
        self.images, self.groundtruth = load_dataset(self.root_dir, self.train)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_path = self.images[idx]
        gt_path = self.groundtruth[idx]

        image = nib.load(img_path)
        image = image.get_data().astype(np.float32)
        # image = standardization_intensity_normalization(image, 'float32')
        # image = sitk.ReadImage(img_path)
        # image = sitk.GetArrayFromImage(image).astype(np.float32)  # [x,y,z] -> [z,y,x]
        # print(image.shape)

        label = nib.load(gt_path)
        label = label.get_data().astype(np.float32)
        # label = sitk.ReadImage(gt_path)
        # label = sitk.GetArrayFromImage(label).astype(np.float32).transpose(2, 1, 0)
        # print(n)

            # n = random.uniform(0.67, 1.5)
            # zoom_image = ndimage.zoom(image, n)
            # # print(zoom_image.max(), zoom_image.min())
            #
            # zoom_label = ndimage.zoom(label, n)
            # print(zoom_label.max(), zoom_label.min())

        ImagePatch, LablePatch = RandomPatchCrop(image, label, patchs_size,
                                                 patchs_size)  # patch_in_size：[64,64,64]
        ImagePatch = standardization_intensity_normalization(ImagePatch, 'float32')

        image = torch.from_numpy(np.ascontiguousarray(ImagePatch)).unsqueeze(0)
        label = torch.from_numpy(np.ascontiguousarray(LablePatch)).unsqueeze(0)

            # image = standardization_intensity_normalization(image, 'float32')
            # image = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0)
            # label = torch.from_numpy(np.ascontiguousarray(label)).unsqueeze(0)

        # image = image / 255
        # label = label / 255
        # print(torch.max(label))
        # print(1)

        return image, label
