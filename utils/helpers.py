from sklearn.model_selection import train_test_split
import pandas as pd
import os
import numpy as np
import glob

def initialise_model(model_type):
    if model_type == 'inception_resnet':
        from keras.applications.inception_resnet_v2 import preprocess_input, InceptionResNetV2

        IM_WIDTH, IM_HEIGHT = 224, 224
        ft_layers = 780

        base_model = InceptionResNetV2(weights='imagenet', include_top=False)  #Not Icluding the FC layer
        return base_model, (IM_WIDTH, IM_HEIGHT), preprocess_input, ft_layers
    elif model_type == 'inception':
        from keras.applications.inception_v3 import preprocess_input, InceptionV3

        IM_WIDTH, IM_HEIGHT = 229, 229
        ft_layers=249

        base_model = InceptionV3(weights='imagenet', include_top=False)  #Not Icluding the FC layer
        return base_model, (IM_WIDTH, IM_HEIGHT), preprocess_input, ft_layers
    elif model_type == 'mobilenet':
        from keras.applications.mobilenet import preprocess_input, MobileNet

        IM_WIDTH, IM_HEIGHT = 224, 224
        ft_layers = 95

        base_model = MobileNet(input_shape=(224, 224, 3), weights='imagenet', include_top=False)  #Not Icluding the FC layer
        return base_model, (IM_WIDTH, IM_HEIGHT), preprocess_input, ft_layers
    elif model_type == 'resnet':
        from keras.applications.resnet50 import ResNet50, preprocess_input

        IM_WIDTH, IM_HEIGHT = 224, 224
        ft_layers = 172

        base_model = ResNet50(weights='imagenet', include_top=False) #Not Icluding the FC layer
        return base_model, (IM_WIDTH, IM_HEIGHT), preprocess_input, ft_layers
    elif model_type == 'vgg16':
        from keras.applications.vgg16 import preprocess_input, VGG16
        
        IM_WIDTH, IM_HEIGHT = 224, 224
        ft_layers = 19

        base_model = VGG16(weights='imagenet', include_top=False)  #Not Icluding the FC layer
        return base_model, (IM_WIDTH, IM_HEIGHT), preprocess_input, ft_layers
    elif model_type == 'vgg19':
        from keras.applications.vgg19 import preprocess_input, VGG19

        IM_WIDTH, IM_HEIGHT = 224, 224
        ft_layers= 22

        base_model = VGG19(weights='imagenet', include_top=False)  #Not Icluding the FC layer
        return base_model, (IM_WIDTH, IM_HEIGHT), preprocess_input, ft_layers
    elif model_type == 'xception':
        from keras.applications.xception import preprocess_input, Xception

        IM_WIDTH, IM_HEIGHT = 299, 299
        ft_layers = 126

        base_model = Xception(weights='imagenet', include_top=False)  #Not Icluding the FC layer
        return base_model, (IM_WIDTH, IM_HEIGHT), preprocess_input, ft_layers
    else:
        raise ValueError("Please Specify a model type by using --model-type argument. See --help for more information.")

def get_data_paths(path):
    '''
    Returns train, test and validation data paths in a given folder.
    The split is done based on the classes and takes into account the class
    imbalances that occur in a dataset. For example, if there are 10 datapoints
    for class A and 90 for class B, a vanilla split might not have a proportional
    number of classes in test set for class A (it might not have any at all). This
    leads to incorrect testing and training of models. This functions prevents that
    by keeping a proportional amount of all classes in training, testing and validation
    dataset.

    By default, the split is train: 70%, test: 20%, validation: 10%.

    Args:
        path: Path to where all your data is. The directory structure of the path given
              should be of the form - <type>/<class>/<data.*>. Where type is train, test, valid etc
    Returns:
        Three numpy arrays for train_paths, test_paths and valid_paths.
    '''
    data_folders = glob.glob(path)

    train_paths = []
    test_paths = []
    valid_paths = []

    for folder in data_folders:
        files = glob.glob(os.path.join(folder, '*.*'))
        train, test, valid = split_sets(files)

        train_paths = train_paths + train
        test_paths = test_paths + test
        valid_paths = valid_paths + valid

    np.random.shuffle(train_paths)
    np.random.shuffle(test_paths)
    np.random.shuffle(valid_paths)

    return np.asarray(train_paths), np.asarray(test_paths), np.asarray(
        valid_paths)

def get_labels(data_paths):
    '''
    Returns the labels for a datapoint by looking at its path.
    Args:
        data_paths: List of data paths of the form- <type>/<label>/<data>.*
                    where type is train, test, valid.
    Returns:
        List of labels for the input.
    '''
    labels = []
    for path in data_paths:
        labels.append(os.path.basename(os.path.dirname(path)))

    return labels

def one_hot_encoding(labels):
    '''
    One Hot Encodes the labels
    Args:
        labels: list of all the labels to one hot encode
    Returns:
        List of one hot encoded labels
    '''
    labels = pd.Series(labels).str.get_dummies()

    return labels

def split_sets(files):
    '''
    Splits the input list into train(70%), test(20%) and validation(10%) datasets
    Args:
        files: List of data to split
    Returns:
        train, test and validation split lists
    '''
    X_train, X_test = train_test_split(files, test_size=0.20, random_state=42)
    X_train, X_valid = train_test_split(
        X_train, test_size=0.10, random_state=42)

    return X_train, X_test, X_valid


def split(data, percentage=0.20):
    x, y = train_test_split(list(data), test_size=percentage, random_state=42)
    X_path, X_class=zip(*x)
    Y_path, Y_class=zip(*y)

    return X_path, X_class, Y_path, Y_class
