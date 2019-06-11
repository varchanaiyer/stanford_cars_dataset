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

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input, InceptionV3
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing import image
from keras.optimizers import SGD

IM_WIDTH, IM_HEIGHT = 229, 229
NB_EPOCHS = 1
BAT_SIZE = 32


def get_images(paths):
    images = []
    for path in paths:
        img = image.load_img(os.path.join('car_ims', path), target_size=(229, 229))
        x = image.img_to_array(img)
        #x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        images.append(x)

    return np.asarray(images)


def one_hot_encoding(labels):
    labels=np.eye(25)[labels]
    return labels


def split(files):
    X_train, X_test = train_test_split(files, test_size=0.20, random_state=42)
    X_train, X_valid = train_test_split(
        X_train, test_size=0.10, random_state=42)

    return X_train, X_test, X_valid


def get_labels(data_paths):
    from preprocess import get_image_path_and_class
    _, labels = get_image_path_and_class()
    return labels


def fine_tune(model):

    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True

    model.compile(
        optimizer=SGD(lr=0.0001, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy'])


def get_data_paths():
    from preprocess import get_image_path_and_class
    #import ipdb; ipdb.set_trace()
    paths, _ = get_image_path_and_class()
    
    return paths

def add_new_last_layer(base_model, nb_classes):
    """Add last layer to the convnet
    Args:
        base_model: keras model excluding top
        nb_classes: # of classes
    Returns:
        new keras model with last layer
    """
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)  #new FC layer, random init
    x = Dense(512, activation='relu')(x)  #new FC layer, random init
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
    #import ipdb; ipdb.set_trace()
    nb_classes = 196 
    nb_epoch = int(args.nb_epoch)
    batch_size = int(args.batch_size)

    train_generator=ImageDataGenerator(rescale=1./255, shear_range=0.1, zoom_range=0.2, horizontal_flip=True)
    test_generator=ImageDataGenerator(rescale=1./255)

    training_set=train_generator.flow_from_directory(os.path.join('data', 'train'), target_size=(299, 299), batch_size=32, class_mode='categorical')
    test_set=test_generator.flow_from_directory(os.path.join('data', 'test'), target_size=(299, 299), batch_size=32, class_mode='categorical')

    # setup model
    base_model = InceptionV3(
        weights='imagenet', include_top=False)  #Not Icluding the FC layer
    model = add_new_last_layer(base_model, nb_classes)

    for layer in base_model.layers:
        layer.trainable = False
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])


    history = model.fit_generator(
        training_set,
        epochs=int(nb_epoch),
        use_multiprocessing=True,
        workers=4,
        verbose=1,
        shuffle=True,
        validation_data=test_set)

    model.save(args.model)

    y_pred_class = model.predict(test_images, verbose=1)

    y_pred_class = [np.argmax(r) for r in y_pred_class]
    test_y = [np.argmax(r) for r in test_labels]

    print('Confusion matrix is \n', confusion_matrix(test_y, y_pred_class))
    print('tn, fp, fn, tp =')
    print(confusion_matrix(test_y, y_pred_class).ravel())
    # Precision
    print('Precision = ', precision_score(test_y, y_pred_class))
    # Recall
    print('Recall = ', recall_score(test_y, y_pred_class))
    # f1_score
    print('f1_score = ', f1_score(test_y, y_pred_class))
    # cohen_kappa_score
    print('cohen_kappa_score = ', cohen_kappa_score(test_y, y_pred_class))
    # roc_auc_score
    print('roc_auc_score = ', roc_auc_score(test_y, y_pred_class))

    if args.ft:

        ft_epochs=args.epoch_ft

        fine_tune(model)
        history = model.fit(
            x=train_images,
            y=train_labels,
            batch_size=batch_size,
            epochs=int(ft_epochs),
            verbose=1,
            shuffle=True,
            validation_data=(valid_images, valid_labels))

        model.save(args.model_ft)

        y_pred_class = model.predict(test_images, verbose=1)

        y_pred_class = [np.argmax(r) for r in y_pred_class]
        test_y = [np.argmax(r) for r in test_labels]

        print('Confusion matrix is \n', confusion_matrix(test_y, y_pred_class))
        print('tn, fp, fn, tp =')
        print(confusion_matrix(test_y, y_pred_class).ravel())
        # Precision
        print('Precision = ', precision_score(test_y, y_pred_class))
        # Recall
        print('Recall = ', recall_score(test_y, y_pred_class))
        # f1_score
        print('f1_score = ', f1_score(test_y, y_pred_class))
        # cohen_kappa_score
        print('cohen_kappa_score = ', cohen_kappa_score(test_y, y_pred_class))
        # roc_auc_score
        print('roc_auc_score = ', roc_auc_score(test_y, y_pred_class))


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument(
        "--nb_epoch",
        default=NB_EPOCHS,
        help='Number of epochs for Transfer Learning. Default = 1.')
    a.add_argument(
        "--batch_size",
        default=BAT_SIZE,
        help='Batch size for training. Default = 32.')
    a.add_argument("--model", help='Path to save model to.')
    a.add_argument("--model_ft", help='Path to save fine tuned model')
    a.add_argument(
        "--ft", action="store_true", help='Whether to fine tune model or not')
    a.add_argument(
        '--epoch_ft',
        default=NB_EPOCHS,
        help='Number of epochs for Fine-Tuning for model. Default = 1.')

    args = a.parse_args()

    if args.ft:
        print("Please make sure that you have added fine tuning epochs value")

    train(args)


    '''
    Sample Command - sudo python3 inception_transfer.py --nb_epoch 1 --model models/resnet/resnet50_1.h5 --ft --model_ft models/resnet/resnet50_1_ft.h5 --epoch_ft 2 --nb_epoch 2
    '''
