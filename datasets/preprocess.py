"""Preprocess ImageNet 1K in advance, in order to avoid transform
overhead one would have to amortize at each batch collection using dataloader.

Recommend you to install Pillow-SIMD first.
  $ pip uninstall pillow
  $ CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
"""

import multiprocessing as mp
from time import sleep
import PIL
from PIL import Image
from tqdm import tqdm
import sys
import os
from os import path


RESIZE = (32, 32)
IMAGENET_DIR = '/v9/whshin/imagenet/'
VISIBLE_SUBDIRS = ['train', 'val']
RESIZE_FILTER = {
  0: Image.NEAREST,
  1: Image.BILINEAR,
  2: Image.BICUBIC,
  3: Image.ANTIALIAS,
}[3]

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp',
                  '.pgm', '.tif', '.tiff', '.webp')
EXIST_OK = False  # Enable overwriting if it's set True
DEBUG = False


def is_image_file(filepath):
  return filepath.lower().endswith(IMG_EXTENSIONS)


def chunkify(list_, n):
  return [[list_[i::n], i] for i in range(n)]


def scandir(supdir, subdirs):
  image_paths = []
  for subdir in subdirs:
    dir = path.join(supdir, subdir)
    print(f'Scanning subdir: {subdir}')
    if sys.version_info >= (3, 5):
      # Faster and available in Python 3.5 and above
      classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    else:
      classes = [d for d in os.listdir(dir) \
                 if path.isdir(path.join(dir, d))]

    for class_ in tqdm(classes):
      d = path.join(dir, class_)
      if not path.isdir(d):
        continue  # dir only
      for root, _, fnames in sorted(os.walk(d)):
        for fname in sorted(fnames):
          image_path = path.join(root, fname)
          if is_image_file(image_path):
            image_paths.append(image_path)

  return image_paths


def split_path(filepath):
  """Split full filepath into 4 different chunks.
    Args:
      filepath (str): "/../imagenet/train/n01632777/n01632777_1000.JPEG"

    Returns:
      base (str): '..'
      dataset (str): 'imagenet'
      subdirs (str): 'train/n01632777'
      filename (str): 'n01632777_1000.JPEG'
  """
  splits = filepath.split('/')
  filename = splits[-1]
  subdirs = path.join(*splits[-3:-1])
  dataset = splits[-4]
  base = path.join('/', *splits[:-4])
  return base, dataset, subdirs, filename


def process(chunk):
  filepaths, i = chunk
  process_desc = '[Process #%2d] ' % i
  for filepath in tqdm(filepaths, desc=process_desc, position=i):
    with Image.open(filepath) as img:
      # processing part
      img = img.resize(RESIZE, RESIZE_FILTER)
      # make new path ready
      base, dataset, subdirs, filename = split_path(filepath)
      dataset_new = "_".join([dataset, 'resized', *map(str, RESIZE)])
      classpath_new = path.join(base, dataset_new, subdirs)
      os.makedirs(classpath_new, exist_ok=EXIST_OK)
      filepath_new = path.join(classpath_new, filename)
      # save resized image
      img.save(filepath_new)
  return 1


def run():
  print('Image preprocessing with multi workers.')
  print(f'RESIZE: {RESIZE}')
  print(f'IMAGENET_DIR: {IMAGENET_DIR}')
  print(f'VISIBLE_SUB_DIRS: {VISIBLE_SUBDIRS}')

  print(f'\nScanning dirs..')
  subdirs = ['val'] if DEBUG else VISIBLE_SUBDIRS
  paths = scandir(IMAGENET_DIR, subdirs)
  print('Done.')

  num_process = 1 if DEBUG else mp.cpu_count()
  chunks = chunkify(paths, num_process)

  if DEBUG:
    print('Start single processing for debugging.')
    process(chunks[0])  # for debugging
  else:
    print(f'Start {num_process} processes.')
    pool = mp.Pool(processes=num_process)
    result = pool.map(process, chunks)
    pool.close()  # no more task
    pool.join()  # wrap up current tasks
    print("Preprocessing completed.")


if __name__ == '__main__':
  run()
