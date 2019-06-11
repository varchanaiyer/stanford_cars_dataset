import numpy as np
import scipy.io as sio
import os
from shutil import copyfile
from utils.helpers import split

def get_image_path_and_class(annotations_path='cars_annos.mat'):
    annotations=sio.loadmat(annotations_path)['annotations']
    image_path=[]
    image_class=[]

    for annotation in annotations[0]:
        image_path.append(annotation[0][0].split('/')[1])
        image_class.append(str(annotation[5][0][0]))
    return image_path, image_class

def save_data(images_path, images_class, train_test='train'):
    for path, cls in zip(images_path, images_class):
        if not os.path.isdir(os.path.join('data', train_test, cls)):
            os.makedirs(os.path.join('data', train_test, cls))
        
        copyfile(os.path.join('car_ims', path), os.path.join('data', train_test, cls, path))

if __name__=='__main__':
    images_path, images_class=get_image_path_and_class('cars_annos.mat')
    #import ipdb; ipdb.set_trace()
    x_path, x_class, y_path, y_class=split(zip(images_path, images_class))
    save_data(x_path, x_class, 'train')
    save_data(y_path, y_class, 'test')

