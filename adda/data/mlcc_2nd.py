import operator
import os
import struct
from functools import reduce
from urllib.parse import urljoin

import numpy as np

from adda.data import DatasetGroup
from adda.data import ImageDataset
from adda.data import util
from adda.data.dataset import register_dataset

@register_dataset("mlcc_2nd")
class MLCC_2ND(DatasetGroup):
    """MNIST dataset used in sualab for DA team"""

    data_files = {
        "train_images" : "target_2nd_train_X.npy",
        "train_labels" : "target_2nd_train_y.npy",
        "valid_images" : "target_2nd_val_X.npy",
        "valid_labels" : "target_2nd_val_y.npy",
        "test_images" : "target_2nd_test_X.npy",
        "test_labels" : "target_2nd_test_y.npy",
    }

    num_classes = 10

    def __init__(self, path=None, shuffle=True):
        data_path = "D:/Dataset/Domain_Adaptation/"
        DatasetGroup.__init__(self, 'MLCC_1st_to_2nd', data_path)
        self.image_shape = (160, 160, 3)
        self.label_shape = ()
        self.shuffle = shuffle
        # self.download()
        self._load_datasets()

    def _load_datasets(self):
        abspaths = {name: self.get_path(path)
                    for name, path in self.data_files.items()}
        train_images = self._read_images(abspaths['train_images'])
        train_images = train_images.astype(np.float32) / 255
        train_labels = self._read_labels(abspaths['train_labels'])
        test_images = self._read_images(abspaths['test_images'])
        test_images = test_images.astype(np.float32) / 255
        test_labels = self._read_labels(abspaths['test_labels'])
        valid_images = self._read_images(abspaths['valid_images'])
        valid_images = valid_images.astype(np.float32) / 255
        valid_labels = self._read_images(abspaths['valid_labels'])
        # print(np.shape(train_images))
        # print(np.shape(train_labels))
        self.train = ImageDataset(train_images, train_labels,
                                  image_shape=self.image_shape,
                                  label_shape=self.label_shape,
                                  shuffle=self.shuffle)
        self.test = ImageDataset(test_images, test_labels,
                                 image_shape=self.image_shape,
                                 label_shape=self.label_shape,
                                 shuffle=self.shuffle)
        self.valid = ImageDataset(valid_images, valid_labels,
                                 image_shape=self.image_shape,
                                 label_shape=self.label_shape,
                                 shuffle=self.shuffle)

    def _read_images(self, path):
        return np.load(path)

    def _read_labels(self, path):
        label = np.load(path)
        return np.array([0 for r in label if r[0] == 1 or 1], dtype=np.uint8) #np.array([np.where(r==1)[0][0] for r in label],dtype=np.uint8)