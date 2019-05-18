"""Preprocess the whole dataset in advance, in order to avoid transform
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
EXIST_OK = True
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


def process(chunk):
  filepaths, pos = chunk
  for filepath in tqdm(filepaths, position=pos):
    img = Image.open(filepath)
    img = img.resize(RESIZE, RESIZE_FILTER)
    # img.show()
    split_path = path.join('/', *filepath.split('/')[:-2])
    class_name = filepath.split('/')[-2]
    split_name = path.basename(split_path)  # train or val
    split_name_new = split_name + "_{}_{}".format(*RESIZE)
    split_path_new = path.join(split_path, os.pardir, split_name_new)
    split_path_new = path.normpath(split_path_new)
    os.makedirs(path.join(split_path_new, class_name), exist_ok=EXIST_OK)
    file_path_new = path.join(split_path_new, *filepath.split('/')[-2:])
    img.save(file_path_new)
  return 1


def run():
  print(f'Scanning dir: {IMAGENET_DIR}')
  paths = scandir(IMAGENET_DIR, VISIBLE_SUBDIRS)
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
