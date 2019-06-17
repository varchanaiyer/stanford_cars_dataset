'''
This is a script to augment all images and save them to the aug_images folder.
'''
from imgaug import augmenters as iaa
import os
from tqdm import tqdm
import numpy as np
import glob
import cv2

image_paths=glob.glob(os.path.join('data', '*', '*', '*.jpg'))

bright=iaa.Sequential(iaa.Add((-20, 20)))
mult_bright=iaa.Sequential(iaa.Multiply((1.15, 1.45)))
mult_dark=iaa.Sequential(iaa.Multiply((0.15, 0.45)))
flip=iaa.Sequential(iaa.Fliplr(1))
blur=iaa.Sequential(iaa.GaussianBlur(sigma=(0.0, 2.0)))
rotate=iaa.Sequential(iaa.Affine(rotate=(-5, 5)))
shear=iaa.Sequential(iaa.Affine(shear=(-2, 2)))

for i, image_path in tqdm(enumerate(image_paths)):
    aug_images=[]
    img=cv2.imread(image_path)
    aug_images.append(bright.augment_images([img]))
    aug_images.append(mult_bright.augment_images([img]))
    aug_images.append(mult_dark.augment_images([img]))
    aug_images.append(blur.augment_images([img]))
    aug_images.append(rotate.augment_images([img]))
    aug_images.append(shear.augment_images([img]))
    aug_images.append(flip.augment_images([img]))

    for j, image in enumerate(aug_images):
        if not os.path.isdir(os.path.join('aug_images', image_path.split('/')[2])):
            os.makedirs(os.path.join('aug_images', image_path.split('/')[2]))
        cv2.imwrite(os.path.join('aug_images', image_path.split('/')[2], f"{i}_{j}_{image_path.split('/')[-1]}"), np.asarray(image[0]))

