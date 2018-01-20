import os
import cv2
import logging
import numpy as np
import glob
#  from PIL import Image


class Dataset(object):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

class TrainDataListDataset(Dataset):
    def __init__(self,
                 train_data_list,
                 dtype=np.float32,
                 data_transform_method=None,
                 target_transform_method=None,
                 skip_failed=False):
        super(TrainDataListDataset, self).__init__()

        self.dtype = dtype
        self.data_transform_method = data_transform_method
        self.target_transform_method = target_transform_method
        self.skip_failed = skip_failed
        self.image_list = train_data_list
        self.n_samples = len(self.image_list)


    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        while True:
            try:
                sample = np.load(self.image_list[index])
                blob = (sample[0],)
                target = (sample[1], sample[2], sample[3])
                break
            except:
                logging.warning('Failed to load {0}'.format(self.image_list[index]))
                if self.skip_failed:
                    # if skip_faied is True, we have to repeat loading samples until we get one
                    index = np.random.randint(0, self.n_samples)
                    continue
                else:
                    # if skip_failed is False, we just return None and let the dataloader to handle it
                    return (None, None, -1)

        return blob, target, index

class ImageListDataset(Dataset):
    def __init__(self,
                 data_dir,
                 list_file,
                 image_idx=0,
                 label_idx=1,
                 dtype=np.float32,
                 data_transform_method=None,
                 target_transform_method=None,
                 skip_failed=False):
        super(ImageListDataset, self).__init__()

        self.data_dir = data_dir
        self.image_list = []
        if label_idx < 0:
            self.label_list = None
        else:
            self.label_list = []
        self.dtype = dtype
        self.data_transform_method = data_transform_method
        self.target_transform_method = target_transform_method
        self.skip_failed = skip_failed

        for l in open(list_file).readlines():
            t = l.strip().split()
            self.image_list.append(t[image_idx])
            if label_idx >= 0:
                self.label_list.append(int(t[label_idx]))

        self.n_samples = len(self.image_list)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        while True:
            try:
                im = cv2.imread(os.path.join(self.data_dir, self.image_list[index])).astype(self.dtype)
                #  im = np.asarray(Image.open(os.path.join(self.data_dir, self.image_list[index]))).astype(self.dtype)
                target = self.label_list[index]
                break
            except:
                logging.warning('Failed to load {0}'.format(self.image_list[index]))
                if self.skip_failed:
                    # if skip_faied is True, we have to repeat loading samples until we get one
                    index = np.random.randint(0, self.n_samples)
                    continue
                else:
                    # if skip_failed is False, we just return None and let the dataloader to handle it
                    return (None, None, -1)

        if self.data_transform_method is not None:
            im = self.data_transform_method(im)

        if self.target_transform_method is not None:
            target = self.target_transform_method(target)

        blob = np.array(im)
        return (blob, ), (target, ), index

class RandomImageListDataset(Dataset):
    def __init__(self,
                 data_dir,
                 list_file,
                 image_idx=0,
                 label_idx=1,
                 dtype=np.float32,
                 data_transform_method=None,
                 target_transform_method=None,
                 skip_failed=False,
                 class_weights=None):
        super(RandomImageListDataset, self).__init__()

        self.data_dir = data_dir
        self.label2imgs = {}
        self.image_list = []
        self.dtype = dtype
        self.data_transform_method = data_transform_method
        self.target_transform_method = target_transform_method
        self.skip_failed = skip_failed
        self.class_weights = class_weights

        for line in open(list_file).readlines():
            l_split = line.strip().split()
            label = int(l_split[label_idx])
            img = l_split[image_idx]
            self.image_list.append(img)
            if label not in self.label2imgs:
                self.label2imgs[label] = []
            self.label2imgs[label].append(img)

        self.n_classes = 1+max(self.label2imgs.keys())
        if isinstance(self.class_weights, (list, np.ndarray)):
            self.class_weights = np.array(self.class_weights)
            assert len(self.class_weights) >= self.n_classes
        elif self.class_weights == 'avg_samples':
            self.class_weights = np.zeros((self.n_classes,), dtype=np.float32)
            for label in self.label2imgs.keys():
                self.class_weights[label] += 1
            self.class_weights /= np.sum(self.class_weights)
        elif self.class_weights == 'avg_classes':
            self.class_weights = np.ones((self.n_classes,), dtype=np.float32)
            self.class_weights /= np.sum(self.class_weights)
        else:
            raise ValueError('Unknown class_weights: %s' % self.class_weights)

        self.n_samples = 0
        for label in self.label2imgs.keys():
            self.n_samples += len(self.label2imgs[label])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        while True:
            try:
                target = np.random.choice(range(self.n_classes),
                                          p=self.class_weights)
                im_file = np.random.choice(self.label2imgs[target])
                im = cv2.imread(os.path.join(self.data_dir, im_file)).astype(self.dtype)
                index = self.image_list.index(im_file)
                break
            except:
                logging.warning('Failed to load {0}'.format(self.image_list[index]))
                if self.skip_failed:
                    # if skip_faied is True, we have to repeat loading samples until we get one
                    continue
                else:
                    # if skip_failed is False, we just return None and let the dataloader to handle it
                    return (None, None, -1)

        if self.data_transform_method is not None:
            im = self.data_transform_method(im)

        if self.target_transform_method is not None:
            target = self.target_transform_method(target)

        blob = np.array(im)
        return (blob, ), (target, ), index
