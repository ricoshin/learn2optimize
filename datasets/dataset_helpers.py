import os
import pdb
import random
import sys
from collections import OrderedDict as odict
from os import path

import _pickle as pickle
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset, dataset
from tqdm import tqdm


# scanning function
def scandir(dir):
  if sys.version_info >= (3, 5):
    return [d.name for d in os.scandir(dir) if d.is_dir()]
  else:
    return [d for d in os.listdir(dir) if path.isdir(path.join(dir, d))]


class Metadata(object):
  def __init__(self, **kwargs):
    # must_have_keys = [
    #   'classes', 'class_to_idx', 'idx_to_class', 'idx_to_samples']
    # assert all([k in kwargs.keys() for k in must_have_keys])
    self.dnames = []
    self._add_metadata(**kwargs)

    # self.classes = classes
    # self.class_to_idx = class_to_idx
    # self.idx_to_class = idx_to_class
    # self.idx_to_samples = idx_to_samples

  def __len__(self):
    return len(self.classes)

  def __eq__(self, other):
    return set(self.classes) == set(other.classes)

  def __add__(self, other):
    return self.merge([self, other])

  def _add_metadata(self, **kwargs):
    import pdb; pdb.set_trace()
    for name, data in kwargs.items():
      setattr(self, name, data)
      self.dnames.append(name)

  def _del_metadata(self, names):
    for name in names:
      if name in self.dnames:
        delattr(self, name)
        self.dnames.remove(name)

  def _change_dname(self, from_to):
    assert isinstance(from_to, dict)
    self._add_metadata(**{from_to[n]: getattr(self, n) for n in self.dnames
                          if n in from_to.keys()})
    self._del_metadata([v for v in from_to.keys()])

  def _add_metadata_imagenet(self, **kwargs):
    # must_have_keys = ['classes', 'class_to_wnid', 'wnid_to_class']
    # assert all([k in kwargs.keys() for k in must_have_keys])
    from_to = {
        'classes': 'wnids',
        'class_to_idx': 'wnid_to_idx',
        'idx_to_class': 'idx_to_wnid',
    }
    self._change_dname(from_to)
    self._add_metadata(kwargs)

  def relative_index(self, rel_indices):
    assert isinstance(rel_indices, (tuple, list))
    abs_indices = list(self.idx_to_samples.keys())
    indices = [abs_indices[rel_idx] for rel_idx in rel_indices]
    classes = [self.idx_to_class[idx] for idx in indices]
    # class_to_idx = odict({cls_: self.class_to_idx[cls_] for cls_ in classes})
    # idx_to_class = odict({self.class_to_idx[cls_]: cls_ for cls_ in classes})
    idx_to_samples = odict({idx: self.idx_to_samples[idx] for idx in indices})
    return Metadata(
        classes, self.class_to_idx, self.idx_to_class, idx_to_samples)

  @classmethod
  def merge(cls, others):
    assert len(others) > 1
    # assert all([others[0] == other for other in others])
    classes = [set(other.classes) for other in others]
    classes = list(classes[0].union(*classes[1:]))
    # import pdb; pdb.set_trace()
    classes.sort()
    class_to_idx = odict({classes[i]: i for i in range(len(classes))})
    idx_to_class = odict({i: classes[i] for i in range(len(classes))})
    idx_to_samples = odict()
    for idx, class_ in idx_to_class.items():
      samples = []
      for other in others:
        samples.extend(other.idx_to_samples[idx])
      idx_to_samples[idx] = list(set(samples))
    return cls(classes, class_to_idx, idx_to_class, idx_to_samples)

  @classmethod
  def get_filepath(cls, root, postfix=''):
    return path.normpath(path.join(
      root, f"{'_'.join(['meta', postfix])}.pickle"))

  @classmethod
  def is_loadable(cls, root, name):
    return path.exists(cls.get_filepath(root, name))

  @classmethod
  def load(cls, root, name):
    filepath = cls.get_filepath(root, name)
    with open(filepath, 'rb') as f:
      meta_data = pickle.load(f)
    print(f'Loaded preprocessed dataset dictionaries: {filepath}')
    return meta_data

  def save(self, root, name):
    filepath = self.get_filepath(root, name)
    with open(filepath, 'wb') as f:
      pickle.dump(self, f)
    print(f'Saved processed dataset dictionaries: {filepath}')

  @classmethod
  def new(cls, *args):
    import pdb; pdb.set_trace()
    print('a')
    return cls(cls._template_base(args))

  @classmethod
  def _template_base(cls, classes, class_to_idx, idx_to_class, idx_to_samples):
    return dict(classes=classes, class_to_idx=class_to_idx,
                idx_to_class=idx_to_class, idx_to_samples=idx_to_samples)

  @classmethod
  def _template_imagenet(cls, classes, class_to_wnid, wnid_to_class):
    return dict(classes=classes, class_to_wnid=class_to_wnid,
                wnid_to_class=wnid_to_class)

  def to_imagenet(self, classes, class_to_wnid, wnid_to_class):
    # class -> wninds
    from_to = {
        'classes': 'wnids',
        'class_to_idx': 'wnid_to_idx',
        'idx_to_class': 'idx_to_wnid',
    }
    self._change_dname(from_to)
    self._add_metadata(self._template_imagenet(
      classes, class_to_wnid, wnid_to_class))
    return self

  @staticmethod
  def load_or_make(meta_dir, name, remake=False, *args, **kwargs):
    import pdb; pdb.set_trace()
    if Metadata.is_loadable(meta_dir, name) and not remake:
      metadata = Metadata.load(meta_dir, name)
    else:
      metadata = Metadata.make(*args, **kwargs)
      metadata.save(meta_dir, name)
    return metadata

  @classmethod
  def make(cls, data_dir, visible_subdirs=None, extensions=None,
           is_valid_file=None, imagenet_devkit_dir=None):

    print("Generating metadata..")
    import pdb; pdb.set_trace()
    data_dir = path.expanduser(data_dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
      raise ValueError("Both extensions and is_valid_file cannot be None "
                       "or not None at the same time")
    if extensions is not None:
      def is_valid_file(filename):
        return filename.lower().endswith(extensions)

    classes, class_to_idx, idx_to_class = cls.scan_classes(
        data_dir, visible_subdirs, extensions)

    idx_to_samples = cls.scan_files(
        data_dir, class_to_idx, visible_subdirs, is_valid_file)

    if any([len(v) == 0 for v in idx_to_samples.values()]):
      raise (RuntimeError(
          "Found 0 files in subfolders of: " + self.root + "\n"
          "Supported extensions are: " + ",".join(extensions)))

    metadata = cls.new(classes, class_to_idx, idx_to_class, idx_to_samples)

    if imagenet_devkit_dir:
      metadata.to_imagenet(
        *cls.scan_imagenet_devkit(data_dir, imagenet_devkit_dir))

    return metadata

  @staticmethod
  def scan_classes(data_dir, visible_subdirs=None, extensions=None):
    """Scan class directories.
      Returns:
        classes (list)
        class_to_idx (OrderedDict)
        idx_to_class (OrderedDict)
    """
    subdirs = visible_subdirs if visible_subdirs else scandir(data_dir)
    classes_subdirs = []
    scan_pbar = tqdm(subdirs)
    for subdir in scan_pbar:
      subdir = path.join(data_dir, subdir)
      scan_pbar.set_description(f'Scanning subdir: {subdir}')
      classes = scandir(subdir)
      # deterministic shuffle to maintain consistency in data splits
      #   between multiple runs
      classes.sort()
      random.Random(1234).shuffle(classes)
      classes_subdirs.append(classes)
    any_classes = classes_subdirs[0]
    if not all([any_classes == classes_subdirs[i]
                for i in range(len(classes_subdirs))]):
      raise Exception("'train' and 'val' splits have different classes.")
    class_to_idx = odict({classes[i]: i for i in range(len(any_classes))})
    idx_to_class = odict({i: classes[i] for i in range(len(any_classes))})
    return any_classes, class_to_idx, idx_to_class

  @staticmethod
  def scan_files(
          data_dir, class_to_idx, visible_subdirs=None, is_valid_file=None):
    """Scan files in each class directories.
      Returns:
        idx_to_samples (OrderedDict): This will help us to maintain class group
          information so that class-level sub-sampling can be easier.
    """
    subdirs = visible_subdirs if visible_subdirs else scandir(data_dir)
    idx_to_samples = odict()  # the order must be preserved!
    for class_, idx in tqdm(class_to_idx.items()):
      samples = []
      for subdir in subdirs:
        dir = path.join(data_dir, subdir, class_)
        if not path.isdir(dir):
          continue
        for base, _, fnames in sorted(os.walk(dir)):
          for fname in sorted(fnames):
            fpath = path.join(base, fname)
            if is_valid_file(fpath):
              samples.append((path, idx))
      idx_to_samples[idx] = samples
    return idx_to_samples

  @staticmethod
  def scan_imagenet_devkit(data_dir, devkit_dir):
    devkit_dir = path.join(data_dir, devkit_dir)
    # load mat 'data/meta.mat'
    mat_fpath = path.join(devkit_dir, 'data', 'meta.mat')
    meta = sio.loadmat(mat_fpath, squeeze_me=True)['synsets']
    nums_children = list(zip(*meta))[4]
    meta = [meta[idx] for idx, num_children in enumerate(nums_children)
            if num_children == 0]
    idcs, wnids, classes = list(zip(*meta))[:3]
    classes = [tuple(clss.split(', ')) for clss in classes]
    idx_to_wnid = {idx: wnid for idx, wnid in zip(idcs, wnids)}
    wnid_to_class = {wnid: clss for wnid, clss in zip(wnids, classes)}
    class_to_wnid = {clss: wnid for wnid, clss in zip(wnids, classes)}
    # load 'data/ILSVRC2012_validation_ground_truth.txt'
    val_gt_fname = 'ILSVRC2012_validation_ground_truth.txt'
    val_gt_fpath = path.join(devkit_dir, 'data', val_gt_fname)
    with open(val_gt_fpath, 'r') as f:
      val_idcs = f.readlines()
    val_idcs = [int(val_idx) for val_idx in val_idcs]
    val_wnids = [idx_to_wnid[idx] for idx in val_idcs]
    return classes, wnid_to_class, class_to_wnid, val_wnids


class ConcatDatasetFolder(dataset.ConcatDataset):
  """Dataset to concatenate multiple 'DatasetFolder's"""

  def __init__(self, datasets):
    super(ConcatDatasetFolder, self).__init__(datasets)
    # if not all([isinstance(dataset, DatasetFolder) for dataset in datasets]):
    #   raise TypeError('All the datasets have to be DatasetFolders.')
    # assert all([others[0] == dataset.meta for dataset in datasets])
    self.meta = Metadata.merge([dset.meta for dset in self.datasets])


class SubsetClass(Dataset):
  def __init__(self, dataset, idx=None, load_all_at_once=True, debug=False):
    self.valid_dataset(dataset)
    self.dataset = dataset

    if idx is None:
      self.indices = list(range(len(dataset)))
      self.meta = dataset.meta
    else:
      prev = 0
      self.indices = []
      self.idxx = []
      self.meta = dataset.meta.relative_index(idx)

      for k, v in self.meta.idx_to_samples.items():
        curr = prev + len(v)
        self.indices.extend(list(range(prev, curr)))
        self.idxx.append(list(range(prev, curr)))
        # import pdb; pdb.set_trace()
        prev = curr

      if debug:
        pdb.set_trace()

  def __getitem__(self, idx):
    return self.dataset[self.indices[idx]]

  def __len__(self):
    return len(self.indices)

  @staticmethod
  def valid_dataset(dataset):
    assert isinstance(dataset, Dataset)
    if not (hasattr(dataset, 'meta') and isinstance(dataset.meta, Metadata)):
      pdb.set_trace()
      raise Exception("Dataset should have attributes 'meta', instance of "
                      "datasets.dataset_helper.Metadata.")

  @classmethod
  def random_sample(cls, dataset, num):
    cls.valid_dataset(dataset)
    sampled_idx = np.random.choice(
        len(dataset.meta.classes), num, replace=False)
    return cls(dataset, sampled_idx.tolist(), debug=False)

  @classmethod
  def split(cls, dataset, ratio):
    cls.valid_dataset(dataset)
    n_classes = len(dataset.meta.classes)
    idx = list(range(n_classes))
    thres = int(n_classes * ratio)
    return cls(dataset, idx[:thres]), cls(dataset, idx[thres:])
