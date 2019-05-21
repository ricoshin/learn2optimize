import itertools
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from datasets.imagenet import ImageNet
from models.meprop import SBLinear, UnifiedSBLinear
from models.model_helpers import ParamsFlattener
from torch.utils import data
from torchvision import datasets, transforms
from utils import utils

C = utils.getCudaManager('default')


class IterDataLoader(object):
  """Just a simple custom dataloader to load data whenever neccessary
  without forcibley using iterative loops.
  """

  def __init__(self, dataset, batch_size, sampler=None):
    self.dataset = dataset
    self.sampler = sampler
    self.dataloader = data.DataLoader(
        dataset, batch_size, sampler)
    self.iterator = iter(self.dataloader)

  def __len__(self):
    return len(self.dataloader)

  def load(self, eternal=True):
    if eternal:
      try:
        return next(self.iterator)
      except StopIteration:
        self.iterator = iter(self.dataloader)
        return next(self.iterator)
      except AttributeError:
        import pdb
        pdb.set_trace()
        print('a')
    else:
      return next(self.iterator)

  @property
  def batch_size(self):
    return self.dataloader.batch_size

  @property
  def full_size(self):
    return self.batch_size * len(self)

  def new_batchsize(self):
    self.dataloader

  @classmethod
  def from_dataset(cls, dataset, batch_size, rand_with_replace=True):
    if rand_with_replace:
      sampler = data.sampler.RandomSampler(dataset, replacement=True)
    else:
      sampler = None
    # loader = data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return cls(dataset, batch_size, sampler)


class ImageNetData:
  """ImageNet 1K dataset:
    - meta-train data(600 classes -> 10 sampling)
      - inner-train data(75% for each class)
      - inner-test data(35% for each class)
    - meta-valid data(200 classes -> 10 sampling)
      - inner-train data(75% for each class)
      - inner-test data(35% for each class)
    - meta-test data(200 classes -> 10 sampling)
      - inner-train data(75% for each class)
      - inner-test data(35% for each class)
  """

  def __init__(self, batch_size=64, fixed=False):
    self.fixed = fixed
    self.batch_size = batch_size
    path = '/v9/whshin/imagenet_resized_32_32'
    # path = '/v9/whshin/imagenet_resized_32_32'
    print(f'Dataset path: {path}')
    # path ='./imagenet'
    composed_transforms = transforms.Compose([
        # transforms.RandomResizedCrop(32),
        # transforms.Resize([32, 32], interpolation=2),
        # transforms.RandomHorizontalFlip(0.5),
        # transforms.ToTensor(),
    ])
    whole_data = ImageNet(path, splits=['train', 'val'], download=True,
                          transform=composed_transforms)
    # test_data = ImageNet(path, split='val', download=True,
    #                      transform=composed_transforms)
    self.m_train_d, self.m_valid_d, self.m_test_d = \
        self._meta_data_split(whole_data)  # , test_data)
    self.meta_train = self.meta_valid = self.meta_test = {}

  def sample_meta_train(self, n_sample=10, ratio=0.75, fixed=None):
    fixed = fixed if fixed is not None else self.fixed
    if fixed and self.meta_train:
      return self.meta_train
    m_train_sampled = self.m_train_d.class_sample(n_sample, preload=True)
    inner_train, inner_test = m_train_sampled.intra_class_split(ratio, True)
    self.meta_train['in_train'] = IterDataLoader.from_dataset(
        inner_train, self.batch_size)
    self.meta_train['in_test'] = IterDataLoader.from_dataset(
        inner_test, self.batch_size)
    return self.meta_train

  def sample_meta_valid(self, n_sample=10, ratio=0.75, fixed=None):
    fixed = fixed if fixed is not None else self.fixed
    if fixed and self.meta_valid:
      return self.meta_valid
    m_valid_sampled = self.m_valid_d.class_sample(n_sample)
    inner_train, inner_test = m_valid_sampled.intra_class_split(ratio, True)
    self.meta_valid['in_train'] = IterDataLoader.from_dataset(
        inner_train, self.batch_size)
    self.meta_valid['in_test'] = IterDataLoader.from_dataset(
        inner_test, self.batch_size)
    return self.meta_valid

  def sample_meta_test(self, n_sample=10, ratio=0.75, fixed=None):
    fixed = fixed if fixed is not None else self.fixed
    if fixed and self.meta_test:
      return self.meta_test
    m_test_sampled = self.m_test_d.class_sample(n_sample)
    inner_train, inner_test = m_test_sampled.intra_class_split(ratio, True)
    self.meta_test['in_train'] = IterDataLoader.from_dataset(
        inner_train, self.batch_size)
    self.meta_test['in_test'] = IterDataLoader.from_dataset(
        inner_test, self.batch_size)
    return self.meta_test

  def _meta_data_split(self, meta_data):  # , test_data):
    # whole_data = ConcatDatasetFolder([train_data, test_data])
    meta_train, meta_valid_test = meta_data.inter_class_split(
        600 / 1000, False)
    meta_valid, meta_test = meta_valid_test.inter_class_split(200 / 400, False)
    return meta_train, meta_valid, meta_test

  def _random_split(self, dataset, ratio=0.5):
    assert isinstance(dataset, data.dataset.Dataset)
    n_total = len(dataset)
    n_a = int(n_total * ratio)
    n_b = n_total - n_a
    data_a, data_b = data.random_split(dataset, [n_a, n_b])
    return data_a, data_b

  def _fixed_split(self, dataset, ratio=0.5):
    assert isinstance(dataset, data.dataset.Dataset)
    n_total = len(dataset)
    thres = int(n_total * ratio)
    id_a = range(len(dataset))[:thres]
    id_b = range(len(dataset))[thres:]
    data_a = data.Subset(dataset, id_a)
    data_b = data.Subset(dataset, id_b)
    return data_a, data_b

  def _get_wrapped_dataloader(self, dataset, batch_size):
    sampler = data.sampler.RandomSampler(dataset, replacement=True)
    loader = data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return IterDataLoader(loader)

  def pseudo_sample(self):
    """return pseudo sample for checking activation size."""
    return (torch.zeros(1, 1, 28, 28), None)
