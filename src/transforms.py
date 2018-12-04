
import math
import cv2
import random
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

from torch import FloatTensor as FT


def get_3d_mask(mask):
    # back, fore, contour (30, 110, 215)
    back_channel = (mask == 30)*1
    nuclei_channel = (mask == 102)*1
    contour_channel = (mask == 215)*1
    # stack depth-wise
    multiclass_out = np.dstack([back_channel, nuclei_channel, contour_channel])
    return multiclass_out

def get_2d_mask(mask):
    # back, fore, contour (30, 110, 215)
    single_channel_out = ((mask == 102)*1) + ((mask == 215)*1)
    return single_channel_out


def train_augment(image, mask, index, size=256):
    # apply some transforms
    if random.uniform(0,1) < 0.66:
        image, mask = random_horizontal_flip_transform2(image, mask)
        image, mask = random_vertical_flip_transform2(image, mask)
        image, mask = random_rotate_transform2(image, mask)
        #image, mask = random_angle_rotate_transform2(image, mask)
        image, mask = random_crop_transform2(image, mask)
    # format image
    mask = get_3d_mask(mask).astype(np.uint8)
    image = fix_resize_transform(image, size, size)
    mask = fix_resize_transform(mask, size, size)
    image = (image.transpose((2,0,1))) / 255 # rearrange channels and normalize
    mask = (mask.transpose((2,0,1))) # rearrange channels

    return FT(image.astype(np.float64)), FT(mask.astype(np.float64)), index


def valid_augment(image, mask, index, size=256):
    # apply some transforms
    if random.uniform(0,1) < 0.66:
        image, mask = random_horizontal_flip_transform2(image, mask)
        image, mask = random_vertical_flip_transform2(image, mask)
        image, mask = random_rotate_transform2(image, mask)
        image, mask = random_angle_rotate_transform2(image, mask)
        image, mask = random_crop_transform2(image, mask)
    # format image
    mask = get_3d_mask(mask).astype(np.uint8)
    image = fix_resize_transform(image, size, size)
    mask = fix_resize_transform(mask, size, size)
    image = (image.transpose((2,0,1))) / 255 # rearrange channels and normalize
    mask = (mask.transpose((2,0,1))) # rearrange channels

    return FT(image.astype(np.float64)), FT(mask.astype(np.float64)), index


def test_augment(image, index, size=256):
    image, mask = fix_resize_transform(image, size, size)
    # format image
    image = (image.transpose((2,0,1))) / 255 # rearrange channels and normalize
    return FT(image.astype(np.float64)), index


def fix_resize_transform2(image, mask, w, h):
    image = cv2.resize(image, (w,h),
                       interpolation = cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (w,h),
                      interpolation = cv2.INTER_LINEAR)
    thresh, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    return image, mask


def fix_resize_transform(image, w, h):
    image = cv2.resize(image, (w,h))
    return image

def random_horizontal_flip_transform2(image, mask, p=0.5):
    """
    give a random horizontal flip to the image and mask
    """
    if random.uniform(0,1) < p:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)
    return image, mask

def random_horizontal_flip_transform(image, p=0.5):
    """
    give a random horizontal flip to the image
    """
    if random.uniform(0,1) < p:
        image = cv2.flip(image, 0)
    return image

def random_vertical_flip_transform2(image, mask, p=0.5):
    """
    give a random vertical flip to the image and mask
    """
    if random.uniform(0,1) < p:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
    return image, mask

def random_vertical_flip_transform(image, p=0.5):
    """
    give a random vertical flip to the image and mask
    """
    if random.uniform(0,1) < p:
        image = cv2.flip(image, 1)
    return image

# 90,180,270 degrees
def random_rotate_transform2(image, mask, p=0.5):
    """
    rotate the image and mask with any of
    90,180 or 270 deg rotation
    """
    if random.uniform(0,1) < p:
        angle = random.choice([90,180,270])
        assert image.shape[:2] == mask.shape[:2]
        center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1)
        image = cv2.warpAffine(image, rot_mat, image.shape[1::-1])
        mask = cv2.warpAffine(mask, rot_mat, mask.shape[1::-1])#, flags=cv2.INTER_LINEAR)
        mask[mask == 0] = 30
    return image, mask

def random_rotate_transform(image, p=0.5):
    """
    rotate the image and mask with any of
    90,180 or 270 deg rotation
    """
    if random.uniform(0,1) < p:
        angle = random.choice([90,180,270])
        center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1)
        image = cv2.warpAffine(image, rot_mat, image.shape[1::-1])#, flags=cv2.INTER_LINEAR)
    return image

def random_angle_rotate_transform2(image, mask, p=0.5):
    """
    rotate the image and mask with any deg rotation
    """
    if random.uniform(0,1) < p:
        angle = random.randrange(0,360,10)
        assert image.shape[:2] == mask.shape[:2]
        center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1)
        image = cv2.warpAffine(image, rot_mat, image.shape[1::-1])

        # TODO seems like warpAffine messes a grayscale image during rotation,
        # TODO we have to rotate each layer of segmented image and stick them together at the end
        # back, fore, contour (30, 110, 215)
        #back_channel = (mask == 30)*1
        #nuclei_channel = (mask == 102)*1
        #contour_channel = (mask == 215)*1
        # stack depth-wise
        #multiclass_out = np.dstack([back_channel, nuclei_channel, contour_channel])

        mask = cv2.warpAffine(mask, rot_mat, mask.shape[1::-1])
        mask = cv2.threshold(mask, 128, 215, cv2.THRESH_BINARY)[1]
    return image, mask

def random_angle_rotate_transform(image, p=0.5):
    """
    rotate the image and mask with any deg rotation
    """
    if random.uniform(0,1) < p:
        angle = random.randrange(0,360,10)
        center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1)
        image = cv2.warpAffine(image, rot_mat, image.shape[1::-1])
    return image

def random_crop_transform2(image, mask, p=0.5):
    if random.uniform(0,1) < p:
        H,W,_ = image.shape
        i,j = (random.randrange(H), random.randrange(W))
        # UPPER LEFT
        if (j <= H / 2) & (i <= W / 2):
            image = image[j:H, i:H]
            mask = mask[j:H, i:H]
        # LOWER LEFT
        elif (j >= H / 2) & (i <= W / 2):
            image = image[:j, i:H]
            mask = mask[:j, i:H]
        # UPPER RIGHT
        elif (j <= H / 2) & (i >= W / 2):
            image = image[j:H, :i]
            mask = mask[j:H, :i]
        # LOWER RIGHT
        elif (j >= H / 2) & (i >= W / 2):
            image = image[:j, :i]
            mask = mask[:j, :i]
        else:
            image = image
            mask = mask
        image = cv2.resize(image, (H,W), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (H,W), interpolation=cv2.INTER_LINEAR)
    return image, mask

def random_crop_transform(image, p=0.5):
    if random.uniform(0,1) < p:
        H,W,_ = image.shape
        i,j = (random.randrange(H), random.randrange(W))
        # UPPER LEFT
        if (j <= H / 2) & (i <= W / 2):
            image = image[j:H, i:H]
        # LOWER LEFT
        elif (j >= H / 2) & (i <= W / 2):
            image = image[:j, i:H]
        # UPPER RIGHT
        elif (j <= H / 2) & (i >= W / 2):
            image = image[j:H, :i]
        # LOWER RIGHT
        elif (j >= H / 2) & (i >= W / 2):
            image = image[:j, :i]
        else:
            image = image
        image = cv2.resize(image, (H,W), interpolation=cv2.INTER_LINEAR)
    return image

# TODO random deformations
def random_deformations():
    pass
