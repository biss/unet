
import numpy as np
import pandas as pd
from collections import defaultdict
import cv2
import os
import glob
import random
from torch.utils.data import DataLoader
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from src import utils
from src import plots
from src import transforms
from src import validation
from src import loaddata
from src import loss
from src import metric
from src import unet256_3x3
from src.classifier import NucleiClassifier


TRAIN_DIR = "/home/biswajit/Documents/bima/unet/data/train/"
TEST_DIR = "/home/biswajit/Documents/bima/unet/data/test/"
FIXED_TRAIN_DIR = "/home/biswajit/Documents/bima/unet/data/train_fixed/"
CLASS_DISTB_DIR = "/home/biswajit/Documents/bima/unet/data/classes.csv"
DUMMY = "/home/biswajit/Documents/bima/unet/data/dummy/"

test_ids = os.listdir(TEST_DIR)
print(len(test_ids))
dummy_ids = os.listdir(DUMMY)
print(len(dummy_ids))

image_dirs, image_masks, image_ids = utils.image_mask_paths(FIXED_TRAIN_DIR)
class_distb = pd.read_csv(CLASS_DISTB_DIR)
train_ids, valid_ids = validation.get_stratified_valid_dirs(class_distb)

image_p = "{0}{1}/images/{1}.png".format(FIXED_TRAIN_DIR, train_ids[1])
mclass_p = "{}{}/multiclass_mask.png".format(FIXED_TRAIN_DIR, train_ids[1])
assert os.path.isfile(image_p)
assert os.path.isfile(mclass_p)
image = utils.open_image(image_p)

train_data = loaddata.ImageLoader(FIXED_TRAIN_DIR, train_ids, mode='train', transform=transforms.train_augment)
train_dl = DataLoader(train_data, batch_size=32, shuffle=True)
valid_data = loaddata.ImageLoader(FIXED_TRAIN_DIR, valid_ids, mode='valid', transform=transforms.train_augment)
valid_dl = DataLoader(valid_data, batch_size=32, shuffle=True)
test_data = loaddata.ImageLoader(TEST_DIR, test_ids, mode='test', transform=transforms.test_augment)
test_dl = DataLoader(test_data, batch_size=1, shuffle=False)
dummy_data = loaddata.ImageLoader(DUMMY, dummy_ids, mode='train', transform=transforms.train_augment)
dummy_dl = DataLoader(dummy_data, batch_size=1, shuffle=True)

in_shape = (3,256,256)
classes = 3
net = unet256_3x3.UNet256_3x3(in_shape, classes)
criteria = loss.MulticlassBCELoss()
metric = metric.dice_score
optimizer = optim.RMSprop(net.parameters(), lr=0.1)

classifier = NucleiClassifier(net, criteria, metric, optimizer, gpu=1)
classifier.train(dummy_dl, valid_dl, 10)
classifier.save_model('./models/mclass_1')
