
import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader

#from .utils import *



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


def read2D(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

class ImageLoader(Dataset):

    def __init__(self, train_dir, image_ids, mode='train', transform=None):
        """
        inputs:
        path is a list containing all training data paths
        """
        self.mode = mode
        self.train_dir = train_dir
        self.image_ids = image_ids
        self.transform = transform

    def __getitem__(self, index):

        img_id = self.image_ids[index]
        train_path = os.path.join(self.train_dir, img_id, "images", img_id)
        train_file = train_path + ".png"
        image = open_image(train_file)
        image_ids = self.image_ids[index]

        if self.mode in ['train', 'valid']:
            mclass_mask_path = os.path.join(self.train_dir, img_id, "multiclass_mask.png")
            mask = read2D(mclass_mask_path)
            if self.transform:
                return self.transform(image, mask, image_ids)
            else:
                return image, mask, image_ids
        else:
            if self.transform:
                return self.transform(image, image_ids)
            else:
                return image, image_ids


    def __len__(self):
        return len(self.image_ids)


if __name__ == "__main__":
    from utils import list_subdirectory

    rootdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_paths, image_ids = list_subdirectory(path=os.path.join(rootdir, "data","train"))
    train_set = ImageLoader(image_paths, image_ids, mode='train', transform=None)
    def get_train_loader(batch_size):
        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  sampler=train_sampler, num_workers=2)
        return(train_loader)
