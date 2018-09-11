
import cv2
import numpy as np

from .utility.utils import *


class TrainImageLoader(object):

    def __init__(self, X_data, Y_data, input_img_resize=None,
                 output_img_size=None, X_transform=None,
                 Y_transform=None):
        """
        inputs:
        X_data is a list containing all training data paths
        y_data contains masks for training data in the same order
        """
        self.X_train = X_data
        self.y_train_masks = Y_data
        self.input_img_resize = input_img_resize
        self.output_img_resize = output_img_resize
        self.X_transform = X_transform
        self.Y_transform = Y_transform

    def __getitem__(self, index):

        image = open_image(self.X_train[index])
        if self.input_img_resize:
            image = cv2.resize(image, self.input_img_resize,
                               interpolation = cv.INTER_LINEAR)

        mask = cv2.imread(self.y_train_masks[index])
        if self.output_img_resize:
            mask = cv2.resize(mask, self.output_img_resize,
                              interpolation = cv.INTER_LINEAR)

        if self.X_transform:
            image = self.X_transform(image)

        if self.Y_transform:
            mask = self.Y_transform(mask)

        return image, mask


    def __len__(self):
        assert len(self.X_data) == len(self.y_train_masks)
        return len(self.X_data)


class TrainImageLoader(object):

    def __init__(self, X_data, input_img_resize=None,
                 X_transform=None):
        self.X_train = X_data
        self.input_img_resize = input_img_resize
        self.X_transform = X_transform

    def __getitem__(self, index):
        image = open_image(self.X_train[index])
        if self.input_img_resize:
            image = cv2.resize(image, self.input_img_resize,
                               interpolation = cv.INTER_LINEAR)

        if self.X_transform:
            image = self.X_transform(image)

        return image

    def __len__(self):
        assert len(self.X_data) == len(self.y_train_masks)
        return len(self.X_data)

