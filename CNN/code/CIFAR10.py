import numpy as np
import pickle
import os
from torchvision.datasets.vision import VisionDataset
from typing import Any, Tuple
from PIL import Image


class cifar10(VisionDataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(cifar10, self).__init__(root, transform=transform, target_transform=target_transform)
        self.train = train  # training set or test set
        # load the picked numpy arrays
        self.data = []
        self.labels = []

        if self.train:
            for batch in range(5):
                file_name = 'data_batch_{}'.format(batch + 1)
                file_path = os.path.join(self.root, file_name)
                with open(file_path, 'rb')as fo:
                    data_batch = pickle.load(fo, encoding='latin1')
                self.data.append(data_batch['data'])
                self.labels.extend(data_batch['labels'])
        else:
            file_path = os.path.join(self.root, 'test_batch')
            with open(file_path, 'rb') as fo:
                data_batch = pickle.load(fo, encoding='latin1')
            self.data.append(data_batch['data'])
            self.labels.extend(data_batch['labels'])
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # ?convert to HWC
        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, 'batches.meta')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data['label_names']
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
