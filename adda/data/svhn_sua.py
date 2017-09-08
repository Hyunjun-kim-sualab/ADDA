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

@register_dataset("svhn_sua")
class SVHN_SUA(DatasetGroup):
    """MNIST dataset used in sualab for DA team"""

    data_files = {
        "train_images" : "source_svhn_X_train.npy",
        "train_labels" : "source_svhn_Y_train.npy",
        "test_images" : "source_svhn_X_val.npy",
        "test_labels" : "source_svhn_Y_val.npy",
    }

    num_classes = 10

    def __init__(self, path=None, shuffle=True):
        data_path = "D:/Dataset/Domain_Adaptation/"
        DatasetGroup.__init__(self, 'SVHN-to-MNIST', data_path)
        self.image_shape = (32, 32, 3)
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
        print(train_labels.shape)
        test_images = self._read_images(abspaths['test_images'])
        test_images = test_images.astype(np.float32) / 255
        test_labels = self._read_labels(abspaths['test_labels'])
        print(np.size(train_labels))
        self.train = ImageDataset(train_images, train_labels,
                                  image_shape=self.image_shape,
                                  label_shape=self.label_shape,
                                  shuffle=self.shuffle)
        self.test = ImageDataset(test_images, test_labels,
                                 image_shape=self.image_shape,
                                 label_shape=self.label_shape,
                                 shuffle=self.shuffle)

    def _read_images(self, path):
        return np.load(path)

    def _read_labels(self, path):
        label = np.load(path)
        return np.array([np.where(r==1)[0][0] for r in label],dtype=np.uint8)