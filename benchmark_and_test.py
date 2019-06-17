import argparse
import os
import time
import glob
import numpy as np
from tqdm import tqdm

from keras.models import Model, load_model
from keras.applications.inception_v3 import preprocess_input
from keras.utils import multi_gpu_model
from keras.preprocessing import image

from utils.helpers import initialise_model

def get_data_paths(folder):
    '''
    To get the path of all the images in a folder
    Args:
        folder: Path of the folder containing images
    Returns:
        list of all files in a directory
    '''
    paths = glob.glob(os.path.join(folder, '*'))

    return np.asarray(paths)

def get_image(path, target_size, preprocess_input):
    '''
    Loads, preprocesses and returns image for prediction
    Args:
        path: Path to the image
        target_size: Size of the image during loading
        preprocess_input: Function to preprocess loaded image
    Returns:
        numpy array of loaded image
    '''
    img = image.load_img(path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    return np.asarray(x)

def inference(args):
    '''
    Function to get the inference results for trained models
    '''

    if not args.gpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    predictions = []

    paths = get_data_paths(args.images)

    if args.model_type == 'mobilenet':
        from keras.utils.generic_utils import CustomObjectScope
        from keras.applications import mobilenet
        model = load_model(args.model, custom_objects={'relu6': mobilenet.relu6})
    else:
        model = load_model(args.model)

    if args.multi:
        model = multi_gpu_model(model, gpus=None)

    _, target_size, preprocess_input, _ = initialise_model(args.model_type)

    for path in paths:
        img = get_image(path, target_size, preprocess_input)

        pred_class=model.predict(img)
        predictions.append(np.argmax(pred_class))
        print(f"For Image {path}, predicted class is: {np.argmax(pred_class)}")

    if not os.path.exists(args.loc):
        os.makedirs(os.path.join(f'{args.loc}'))
        print(f'Saving to file {args.loc}/{args.model}_predictions')
    
    np.save(os.path.join(f'{args.loc}', f'predictions'), predictions)

def benchmark(args):
    '''
    Function to get the benchmarking results for trained models
    '''

    if not args.gpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    pp_time = []
    p_time = []
    t_time = []

    paths = get_data_paths(args.images)

    load_time = time.time()
    
    if args.model_type == 'mobilenet':
        from keras.utils.generic_utils import CustomObjectScope
        from keras.applications import mobilenet
        model = load_model(args.model, custom_objects={'relu6': mobilenet.relu6})
    else:
        model = load_model(args.model)
    
    print(f'Time taken to load model is : {time.time()-load_time}')
    print('Sleeping')
    time.sleep(5)

    if args.multi:
        model = multi_gpu_model(model, gpus=None)

    _, target_size, preprocess_input, _ = initialise_model(args.model_type)

    for _ in tqdm(range(int(args.i))):
        for path in tqdm(paths):
            total_time = time.time()
            preprocess_time = time.time()
            img = get_image(path, target_size, preprocess_input)
            pp_time.append(time.time() - preprocess_time)

            prediction_time = time.time()
            model.predict(img)
            p_time.append(time.time() - prediction_time)

            t_time.append(time.time() - total_time)

    if not os.path.exists(args.loc):
        os.makedirs(os.path.join(f'{args.loc}'))
        print(f'Saving to dir {args.loc}')
    
    np.save(os.path.join(f'{args.loc}', f'pp_time'), pp_time)
    np.save(os.path.join(f'{args.loc}', f'p_time'), p_time)
    np.save(os.path.join(f'{args.loc}', f't_time'), t_time)


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--loc', help='The path where you want to save the prediction or benchamrking results')
    a.add_argument('--model', help='The path of the saved model file')
    a.add_argument("--model_type", default= 'inception_resnet',
        help='Which model to use. Choose one of: inception_resnet, inception, mobilenet,\
                resnet, vgg16, vgg19, xception. Default = inception_resnet')
    a.add_argument('--gpu', action='store_true', help='To use GPU')
    a.add_argument('--benchmark', action="store_true", help='Use this flag if you want to perform benchmarking instead of predicting')
    a.add_argument('--multi', action='store_true', help='To use multiple GPU')
    a.add_argument('--images', help='Location of test images')
    a.add_argument('--i', default=1, help="Test images will be iterated i'th many times. Default = 1")
    args = a.parse_args()
    
    if args.benchmark:
        benchmark(args)
    else:
        inference(args)

