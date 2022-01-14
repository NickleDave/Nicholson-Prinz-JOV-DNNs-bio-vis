#!/usr/bin/env python
# coding: utf-8
"""script used to split CelebA-Spoof dataset for training with train_image_classify.py

to set up to run this, first download CelebA-Spoof dataset.

$ cd ~/${CELEBA_SPOOF_DATASET_DIR}  # below it's "~/Documents/data/CelebA-Spoof"
$ brew install gdrive
$ gdrive download --recursive --skip 1OW_1bawO79pRqdVEVmBzp8HSxdSwln_Z
$ cat CelebA_Spoof.zip.* > CelebA_Spoof.zip
$ 7z x CelebA_spoof.zip
$ mv CelebA_Spoof/* .
"""
import math
from pathlib import Path
import random
import shutil

from tqdm import tqdm


CLASSES = ('live', 'spoof')


def _get_img_paths(root, ext):
    img_paths = {}
    for class_ in CLASSES:
        print(
            f'getting {class_} images with glob, may take a moment'
        )
        img_paths_class = sorted(root.glob(f'*/{class_}/*.{ext}'))
        print(
            f'found {len(img_paths_class)} images for {class_} class'
        )
        img_paths[class_] = img_paths_class

    new_img_paths = {}
    for class_ in CLASSES:
        print(
            f'generating new image paths for {class_} class'
        )
        new_img_paths_class = []
        for img_path in tqdm(img_paths[class_]):
            new_img_paths_class.append(
                root / class_ / img_path.name
            )
        new_img_paths[class_] = new_img_paths_class
    
    return img_paths, new_img_paths


def separate_live_and_spoof_into_classes(dataset_root, split_train=True, val_frac=0.1, dryrun=True):
    """prepare CelebA-Spoof dataset to be used with ImageFolder,
    by making 'live' and 'spoof' sub-directories that will be the two classes,
    and moving images into those
    """
    for split in ('train', 'test'):
        split_root = dataset_root / split

        for class_ in CLASSES:
            class_dir = split_root / class_
            class_dir.mkdir(exist_ok=True)

        if split == 'train':
            ext = 'jpg'  # why in God's name do you have a different image format for your two splits. I bet this affects performance

            if split_train:
                val_root = dataset_root / 'val'
                val_root.mkdir(exist_ok=True)
                for class_ in CLASSES:
                    class_dir = val_root / class_
                    class_dir.mkdir(exist_ok=True)

                # then we need to carefully split by celeb IDs 
                # instead of blindly moving all images to live / spoof dirs
                id_dirs = sorted([dir_ for dir_ in split_root.iterdir() if dir_.is_dir()])
                random.shuffle(id_dirs)
                val_end_ind = math.ceil(len(id_dirs) * val_frac)
                val_id_dirs = id_dirs[:val_end_ind]
                train_id_dirs = id_dirs[val_end_ind:]

                img_paths_by_train_split = {}
                new_img_paths_by_train_split = {}

                for train_split, id_dirs in zip(('train', 'val'), (train_id_dirs, val_id_dirs)):
                    train_split_root = DATASET_ROOT / train_split

                    img_paths_this_train_split = {}
                    for class_ in CLASSES:
                        print(
                            f'getting {class_} images with glob for training set split "{train_split}", may take a moment'
                        )
                        img_paths_class = [sorted(id_dir.glob(f'{class_}/*.{ext}')) for id_dir in id_dirs]
                        # need to flatten
                        img_paths_class = [img_path 
                                           for id_dir_img_paths in img_paths_class
                                           for img_path in id_dir_img_paths]
                        print(
                            f'found {len(img_paths_class)} images for {class_} class'
                        )
                        img_paths_this_train_split[class_] = img_paths_class
                    img_paths_by_train_split[train_split] = img_paths_this_train_split

                    new_img_paths_this_train_split = {}
                    for class_ in CLASSES:
                        print(
                            f'generating new image paths for {class_} class'
                        )
                        new_img_paths_class = []
                        for img_path in tqdm(img_paths_this_train_split[class_]):
                            new_img_paths_class.append(
                                train_split_root / class_ / img_path.name
                            )
                        new_img_paths_this_train_split[class_] = new_img_paths_class
                    new_img_paths_by_train_split[train_split] = new_img_paths_this_train_split

                # unpack everything back into `img_path` and `new_img_paths` to match the rest of this function
                img_paths = {class_: [] for class_ in CLASSES}
                new_img_paths = {class_: [] for class_ in CLASSES}
                for train_split in ('train', 'val'):
                    for img_path_dict, img_path_dict_by_train_split in zip(
                        (img_paths, new_img_paths),
                        (img_paths_by_train_split, new_img_paths_by_train_split),
                    ):
                        img_path_dict_this_train_split = img_path_dict_by_train_split[train_split]
                        for class_, img_path_list in img_path_dict_this_train_split.items():
                            img_path_dict[class_].extend(img_path_list)
            else:
                 img_paths, new_img_paths = _get_img_paths(split_root, ext)

        elif split == 'test':
            ext = 'png'  # why in God's name do you have a different image format for your two splits
            img_paths, new_img_paths = _get_img_paths(split_root, ext)

        if dryrun:
            for class_ in CLASSES:
                print(
                    f'dry run, printing first 5 img_paths for class {class_} that will be moved to:'
                    f'{root / class_}.\n'
                    f'{img_paths[class_][:5]}\n\n'
                    f'{new_img_paths[class_][:5]}\n\n'
                )
        else:
            for class_ in CLASSES:
                print(
                    f'moving {class_} images'
                )
                for img_path, new_img_path in tqdm(zip(img_paths[class_], new_img_paths[class_])):
                    shutil.move(img_path, new_img_path)


DATASET_ROOT = Path('~/Documents/data/CelebA-Spoof/Data/')

separate_live_and_spoof_into_classes(DATASET_ROOT, dryrun=False)
