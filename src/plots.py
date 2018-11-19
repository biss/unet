
import os
import glob
import numpy as np
import cv2

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def show_img_msk_frompath(img_path, msk_path=None, alpha=0.35, size=7):
    """
    Show original image and masked on top of image
    next to each other in desired size
    """
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if msk_path:
        fig=plt.figure(figsize=(size, size))
        fig.add_subplot(1, 2, 1)
        plt.imshow(image)
        plt.xlabel('original image')
        mask_image = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        masked_image = np.ma.masked_where(mask_image==0, mask_image)
        fig.add_subplot(1, 2, 2)
        plt.imshow(image)
        plt.imshow(masked_image, cmap='cool', alpha=alpha)
        plt.xlabel('masked image')
    else:
        plt.imshow(image)
    plt.show()
    plt.close()


def show_img_msk_fromarray(img_arr, msk_arr, alpha=0.35, size=7):
    fig=plt.figure(figsize=(size, size))
    fig.add_subplot(1, 2, 1)
    plt.imshow(img_arr)
    plt.xlabel('original image')
    masked_image = np.ma.masked_where(msk_arr==0, msk_arr)
    fig.add_subplot(1, 2, 2)
    plt.imshow(img_arr)
    plt.imshow(masked_image, cmap='cool', alpha=alpha)
    plt.xlabel('masked image')
    plt.show()
    plt.close()


def show_with_size(img, size=7):
    plt.figure(figsize=(size, size))
    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.show()
    plt.close()


def plot_img_with_mask(img_path, mask_paths):
    img = plt.imread(img_path)
    fig=plt.figure(figsize=(8, 8))
    fig.add_subplot(1, 2, 1)
    plt.imshow(img)

    mask = cv2.imread(mask_paths[0], cv2.IMREAD_GRAYSCALE)
    #mask = np.zeros(first_mask.shape, dtype=np.uint8)
    for i in mask_paths[1:]:
        mask = cv2.add(mask, cv2.imread(i, cv2.IMREAD_GRAYSCALE))
    fig.add_subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.show()

def plot_all_masks(mask_paths):
    """
    plot all masks in subplot
    """
    fig=plt.figure(figsize=(8, 8))
    size = len(mask_paths)
    l = h = math.ceil(math.sqrt(size))
    print(l,h)

    for i,mask in enumerate(mask_paths):
        mask_img = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
        fig.add_subplot(l, h, i+1)
        plt.imshow(mask_img, cmap="gray")
    plt.show()

def plot_pair_from_path(img, msk):
    img = cv2.imread(img, cv2.IMREAD_COLOR)
    msk = cv2.imread(msk, cv2.IMREAD_GRAYSCALE)
    fig=plt.figure(figsize=(8, 8))
    fig.add_subplot(1, 2, 1)
    plt.imshow(img)
    fig.add_subplot(1, 2, 2)
    plt.imshow(msk, cmap="gray")
    plt.show()
    plt.close()

def plot_pair_from_arr(img, msk):
    fig=plt.figure(figsize=(8, 8))
    fig.add_subplot(1, 2, 1)
    plt.imshow(img)
    fig.add_subplot(1, 2, 2)
    plt.imshow(msk, cmap="gray")
    plt.show()
    plt.close()


if __name__ == "__main__":
    pass
