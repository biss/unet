
import os
import glob
import numpy as np
from PIL import Image
import cv2

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

if __name__ == "__main__":
    from plots import *


def list_subdirectory(path):
    return glob.glob(path + "/*"), os.listdir(path)

def image_mask_paths(train_dir):
    """
    arg: train directory

    return a list of training image paths,
    multiclass mask path, and image id
    """
    image_dir_list = sorted(glob.glob(train_dir + '/*'))
    image_path_list = [item + '/images' + '/{}.png'.format(
                       item.split('/')[-1]) for item
                       in image_dir_list]
    mcmask_path_list = [item + "/multiclass_mask.png" for item in image_dir_list]
    image_id = sorted(os.listdir(train_dir))
    return image_path_list, mcmask_path_list, image_id


def one_mask_arr(msk_paths):
    mask = cv2.imread(msk_paths.pop(), cv2.IMREAD_GRAYSCALE)
    #mask = np.zeros(first_mask.shape, dtype=np.uint8)
    while msk_paths:
        mask = cv2.add(mask, cv2.imread(msk_paths.pop(),
                       cv2.IMREAD_GRAYSCALE))
    return mask

def one_mask_file(mask_paths, mask_name="one_mask"):
    image_root = os.path.dirname(os.path.dirname(mask_paths[0]))
    mask_dir = os.path.join(image_root, "mask")
    one_mask_path = os.path.join(image_root, "{}.png".format(mask_name))
    if not os.path.isfile(one_mask_path):
        mask_arr = one_mask_arr(mask_paths)
        mask_arr = np.dstack([mask_arr]*3).astype('uint8')
        im = Image.fromarray(mask_arr)
        im.save(one_mask_path)

def make_one_mask(train_dir, mask_name="one_mask"):
    for path in train_dir:
        path = os.path.dirname(os.path.dirname(path))
        mask_dir = os.path.join(path, "masks")
        mask_paths = glob.glob(mask_dir + "/*")
        one_mask_file(mask_paths)

def get_image_onemask_paths(path, onemask=True):
    image_paths = [p + '/images/' + p.split('/')[-1] + '.png'
                   for p in glob.glob(path+"/*")]
    if onemask:
        mask_paths =  [p + '/one_mask.png' for p in glob.glob(path+"/*")]
        return list(zip(image_paths, mask_paths))
    else:
        return image_paths

def to_numpy(x):
    return x.data.cpu().numpy()

def read2D(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def read3D(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)

def open_image(fn):
    flags = cv2.IMREAD_UNCHANGED+cv2.IMREAD_ANYDEPTH+cv2.IMREAD_ANYCOLOR
    if not os.path.exists(fn):
        raise OSError('No such file or directory: {}'.format(fn))
    elif os.path.isdir(fn):
        raise OSError('Is a directory: {}'.format(fn))
    else:
        try:
            im = cv2.imread(str(fn), flags).astype(np.float32)/255
            if im is None:
                raise OSError("File not recognized by opencv: {}".format(fn))
            return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise OSError('Error handling image at: {}'.format(fn)) from e

def multiclass_mask(mask_path):
    """
    Return an 2d numpy arraya representing the multiclass mask
    value 0: background, 1: inside the nuclei and 2: contour
    """
    mask = open_image(mask_path)
    gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    (t, binary) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    (_, contours, _) = cv2.findContours(binary.astype(np.uint8),
                           cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contoured_mask = cv2.drawContours(mask, contours, -1, (0, 0, 255), 2)
    contoured_mask2d = ((contoured_mask[:, :, 0]==1)*1) + ((contoured_mask[:, :, 2]==255)*2)
    return contoured_mask2d

def make_multiclass_mask(mask_path, always=False, mcm_name="multiclass_mask.png",):
    """
    mask multiclass masks and save them in the given path
    """
    mask_paths = glob.glob(mask_path + '/*')
    train_dir = os.path.dirname(mask_path)
    mc_mask_path = os.path.join(train_dir, mcm_name)
    if not os.path.isfile(mc_mask_path) or always:
        mc_mask = np.sum(np.array([multiclass_mask(item) for item in mask_paths]),0)
        out = np.clip(mc_mask, 0, 2)
        out[np.where(out == 0)] = 30
        out[np.where(out == 1)] = 110
        out[np.where(out == 2)] = 215
        plt.imsave(mc_mask_path, out)

if __name__ == "__main__":
    train_data_path = "/home/biswajit/Documents/bima/unet/data/train_fixed"
    img_path, mask_path, image_id = image_mask_paths(train_data_path)
    make_one_mask(img_path)

    # create multiclass mask
    image_dir_list = glob.glob(train_data_path + '/*')
    all_mask_paths = [item + "/masks" for item in image_dir_list]
    for item in all_mask_paths:
        make_multiclass_mask(item, always=True)
