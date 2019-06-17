import argparse
import matplotlib.pyplot as plt
import os
import sys
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score, roc_curve
from collections import Counter
from sklearn.utils import class_weight

from utils.helpers import get_data_paths, get_labels, one_hot_encoding, initialise_model
from tqdm import tqdm

from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing import image
from keras.optimizers import SGD

from keras import backend as K
K.set_image_dim_ordering('tf')

def image_generator(paths, labels, target_size, preprocess_input):
    '''
    A generator that loads and returns images and their labels given the path to the image
    Args:
        paths: List of path to all images
        labels: List of labels for all the images
        target_size: The target size output for the image
        preprocess_input: Function to preprocess the image

    Returns:
        A generator which gives a Tuple of the form (numpy array of images, labels for those images)
    '''
    batch_size=32
    while True:
        for i in range(int(len(paths)/32)):
            images=[]
            image_labels=[]
            data_batch=paths[i*batch_size:(i*batch_size)+batch_size]
            data_labels=labels[i*batch_size:(i*batch_size)+batch_size]
            for path in data_batch:
                img = image.load_img(path, target_size=target_size)
                x = image.img_to_array(img)
                x = preprocess_input(x)
                images.append(x)

            yield (np.asarray(images), data_labels)

def get_images(paths, target_size, preprocess_input):
    '''
    Similar to image_generator except that it does not return a generator and instead returns all the 
    images for the list of paths given.
    Args:
        paths: List of path to all the images
        target_size: The target size output for the image
        preprocess_input: Function to preprocess the image
    Returns:
        numpy array of all images
    '''
    images = []
    for path in paths:
        img = image.load_img(path, target_size=target_size)
        x = image.img_to_array(img)
        x = preprocess_input(x)
        images.append(x)

    return np.asarray(images)

def fine_tune(model, ft_layers):
    '''
    Compile the model for finetuning
    Args:
        model: The model to compile
        ft_layers: Number of layers to freeze
    Returns:
        NA
    '''
    for layer in model.layers[:ft_layers]:
        layer.trainable = False
    for layer in model.layers[ft_layers:]:
        layer.trainable = True

    model.compile(
        optimizer=SGD(lr=0.0001, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

def add_new_last_layer(base_model, nb_classes):
    """Add last layer to the convnet
    Args:
        base_model: keras model excluding top
        nb_classes: # of classes
    Returns:
        new keras model with last layer
    """
    x = base_model.output
    #x = Flatten(input_shape=base_model.output_shape[1:])(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(5000, activation='relu')(x)  #new FC layer, random init
    x = Dense(2500, activation='relu')(x)  #new FC layer, random init
    x = Dense(5000, activation='relu')(x)  #new FC layer, random init
    x = Dense(2500, activation='relu')(x)  #new FC layer, random init
    predictions = Dense(
        nb_classes, activation='softmax')(x)  #new softmax layer
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def setup_to_transfer_learn(model, base_model):
    """Freeze all layers and compile the model"""
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])


def train(args):
    '''
    Trains the models
    '''
    nb_classes = 196 
    nb_epoch = int(args.nb_epoch)
    batch_size = int(args.batch_size)

    print(f"Training {args.model_type} model")

    train_paths, test_paths, valid_paths = get_data_paths(os.path.join('aug_images', '*'))

    print(f"No. of Train samples = {len(train_paths)} \n")
    print(f"No. of Test samples = {len(test_paths)} \n")
    print(f"No. of Valid samples = {len(valid_paths)} \n")

    train_labels = get_labels(train_paths)
    print(f'For Train = {Counter(train_labels)} \n')
    train_labels = np.asarray(one_hot_encoding(train_labels))

    test_labels = get_labels(test_paths)
    print(f'For Test = {Counter(test_labels)} \n')
    test_labels = np.asarray(one_hot_encoding(test_labels))

    valid_labels = get_labels(valid_paths)
    print(f'For Valid = {Counter(valid_labels)} \n')
    valid_labels = np.asarray(one_hot_encoding(valid_labels))

    # setup model
    base_model, target_size, preprocess_input, ft_layers = initialise_model(args.model_type) 
    model = add_new_last_layer(base_model, nb_classes)

    for layer in base_model.layers:
        layer.trainable = False
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    print(model.summary())

    history=model.fit_generator(image_generator(train_paths, train_labels, target_size, preprocess_input),
            steps_per_epoch=1, nb_epoch=nb_epoch, validation_data=image_generator(valid_paths, valid_labels, target_size, preprocess_input), validation_steps=1)
    if args.model:
        model.save(args.model)

    test_images = get_images(test_paths[:10], target_size, preprocess_input)

    y_pred_class = model.predict(test_images, verbose=1)

    y_pred_class = [np.argmax(r) for r in y_pred_class]
    test_y = [np.argmax(r) for r in test_labels[:10]]

    print('Confusion matrix is \n', confusion_matrix(test_y, y_pred_class))

    if args.ft:
        print("Fine Tuning the model")
        ft_epochs=int(args.epoch_ft)

        fine_tune(model, ft_layers)

        history=model.fit_generator(image_generator(train_paths, train_labels, target_size, preprocess_input),
                steps_per_epoch=1, nb_epoch=ft_epochs, validation_data=image_generator(valid_paths, valid_labels, target_size, preprocess_input), validation_steps=1)
        
        model.save(args.model_ft)

        y_pred_class = model.predict(test_images, verbose=1)

        y_pred_class = [np.argmax(r) for r in y_pred_class]
        test_y = [np.argmax(r) for r in test_labels[:10]]

        print('Confusion matrix is \n', confusion_matrix(test_y, y_pred_class))

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--model_type",
            default= 'inception_resnet',
            help='Which model to use. Choose one of: inception_resnet, inception, mobilenet, \
            resnet, vgg16, vgg19, xception. Default = inception_resnet')
    a.add_argument(
        "--nb_epoch",
        default=1,
        help='Number of epochs for Transfer Learning. Default = 1.')
    a.add_argument(
        "--batch_size",
        default=32,
        help='Batch size for training. Default = 32.')
    a.add_argument("--model", help='Path to save model to.')
    a.add_argument("--model_ft", help='Path to save fine tuned model')
    a.add_argument(
        "--ft", action="store_true", help='Whether to fine tune model or not')
    a.add_argument(
        '--epoch_ft',
        default=1,
        help='Number of epochs for Fine-Tuning for model. Default = 1.')

    args = a.parse_args()

    if args.ft:
        print("Please make sure that you have added fine tuning epochs value")

    train(args)


    '''
    Sample Command - sudo python3 inception_transfer.py --nb_epoch 1 --model models/resnet/resnet50_1.h5 --ft --model_ft models/resnet/resnet50_1_ft.h5 --epoch_ft 2 --nb_epoch 2
    '''
