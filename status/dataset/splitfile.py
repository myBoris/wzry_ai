import os
import shutil
import random
from tqdm import tqdm


def split_images(source_dir, train_dir, val_dir, train_ratio=0.8):
    for root, dirs, files in os.walk(source_dir):
        for dir_name in dirs:
            sub_dir = os.path.join(root, dir_name)
            images = [f for f in os.listdir(sub_dir) if os.path.isfile(os.path.join(sub_dir, f))]
            random.shuffle(images)

            train_count = int(len(images) * train_ratio)
            train_images = images[:train_count]
            val_images = images[train_count:]

            train_sub_dir = sub_dir.replace(source_dir, train_dir)
            val_sub_dir = sub_dir.replace(source_dir, val_dir)

            os.makedirs(train_sub_dir, exist_ok=True)
            os.makedirs(val_sub_dir, exist_ok=True)

            print(f"Processing directory: {sub_dir}")
            print(
                f"Total images: {len(images)}, Training images: {len(train_images)}, Validation images: {len(val_images)}")

            for image in tqdm(train_images, desc=f"Copying to {train_sub_dir}", unit="image"):
                shutil.copy(os.path.join(sub_dir, image), os.path.join(train_sub_dir, image))

            for image in tqdm(val_images, desc=f"Copying to {val_sub_dir}", unit="image"):
                shutil.copy(os.path.join(sub_dir, image), os.path.join(val_sub_dir, image))


source_directory = 'sss'
train_directory = 'train'
val_directory = 'val'

split_images(source_directory, train_directory, val_directory)
