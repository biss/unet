
import os
import glob
import numpy as np
from PIL import Image
import cv2

if __name__ == "__main__":
    from plots import *


def image_mask_paths(data_dir=None):
    if not data_dir:
        my_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(my_dir)
        train_data_dir = os.path.join(root_dir, "data", "train")
    image_dir_list = sorted(glob.glob(train_data_dir + '/*'))
    image_id = sorted(os.listdir(train_data_dir))
    return image_dir_list, image_id


def image_and_masks(img_dir):
    img_path = os.path.join(img_dir, "image")
    img_path = glob.glob(mask_path + '/*')
    masks_path = os.path.join(img_dir, "masks")
    masks_path = glob.glob(mask_path + '/*')
    return img_path, masks_path

def one_mask_arr(msk_paths):
    mask = cv2.imread(msk_paths.pop(), cv2.IMREAD_GRAYSCALE)
    #mask = np.zeros(first_mask.shape, dtype=np.uint8)
    while msk_paths:
        mask = cv2.add(mask, cv2.imread(msk_paths.pop(),
                       cv2.IMREAD_GRAYSCALE))
    return mask

def make_one_mask_file(mask_paths, mask_name="one_mask"):
    image_root = os.path.dirname(os.path.dirname(mask_paths[0]))
    mask_dir = os.path.join(image_root, "mask")
    one_mask_path = os.path.join(image_root, "{}.png".format(mask_name))
    if not os.path.isfile(one_mask_path):
        mask_arr = one_mask_arr(mask_paths)
        mask_arr = np.dstack([mask_arr]*3).astype('uint8')
        im = Image.fromarray(mask_arr)
        im.save(one_mask_path)

def make_one_mask(train_dirs, mask_name="one_mask"):
    for train_dir in train_dirs:
        mask_dir = os.path.join(train_dir, "masks")
        mask_paths = glob.glob(mask_dir + "/*")
        make_one_mask_file(mask_paths)

def get_image_onemask_paths(path, onemask=True):
    image_paths = [p + '/images/' + p.split('/')[-1] + '.png'
                   for p in glob.glob(path+"/*")]
    if onemask:
        mask_paths =  [p + '/one_mask.png' for p in glob.glob(path+"/*")]
        return list(zip(image_paths, mask_paths))
    else:
        return image_paths

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
                raise OSError(f'File not recognized by opencv: {fn}')
            return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise OSError('Error handling image at: {}'.format(fn)) from e


if __name__ == "__main__":
    img_dirs, msk_dirs = image_mask_paths()
    make_one_mask(img_dirs)
