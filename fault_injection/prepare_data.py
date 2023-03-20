import tensorflow as tf
import config
import pathlib
from config import *
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np

def load_and_preprocess_image(img_path):
    # read pictures
    img_raw = tf.io.read_file(img_path)
    # decode pictures
    img_tensor = tf.image.decode_jpeg(img_raw, channels=channels)
    # resize
    img_tensor = tf.image.resize(img_tensor, [image_height, image_width])
    img_tensor = tf.cast(img_tensor, tf.float32)
    # normalization
    img = img_tensor / 255.0
    return img

def get_images_and_labels(data_root_dir):
    # get all images' paths (format: string)
    data_root = pathlib.Path(data_root_dir)
    all_image_path = [str(path) for path in list(data_root.glob('*/*'))]
    # get labels' names
    label_names = sorted(item.name for item in data_root.glob('*/'))
    # dict: {label : index}
    label_to_index = dict((label, index) for index, label in enumerate(label_names))
    # get all images' labels
    all_image_label = [label_to_index[pathlib.Path(single_image_path).parent.name] for single_image_path in all_image_path]

    return all_image_path, all_image_label


def get_dataset(images, labels):
    #all_image_path, all_image_label = get_images_and_labels(data_root_dir=dataset_root_dir)
    # print("image_path: {}".format(all_image_path[:]))
    # print("image_label: {}".format(all_image_label[:]))
    # load the dataset and preprocess images

    return tf.data.Dataset.from_tensor_slices((images, labels)), len(labels)

    '''
    image_dataset = tf.data.Dataset.from_tensor_slices(images)
    label_dataset = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((image_dataset, label_dataset))
    image_count = len(labels)

    return dataset, image_count
    '''

def load_cifar10() :
    (train_data, train_labels), (test_data, test_labels) = cifar10.load_data()
    # train_data = train_data / 255.0
    # test_data = test_data / 255.0

    train_data, test_data = normalize(train_data, test_data)

    #train_labels = to_categorical(train_labels, 10)
    #test_labels = to_categorical(test_labels, 10)

    '''
    seed = 777
    np.random.seed(seed)
    np.random.shuffle(train_data)
    np.random.seed(seed)
    np.random.shuffle(train_labels)
    '''

    return train_data, train_labels, test_data, test_labels


def generate_datasets(seed):
    x_train, y_train, x_test, y_test = load_cifar10()

    train_dataset, train_count = get_dataset(x_train, y_train)
    valid_dataset, valid_count = get_dataset(x_test, y_test)
    #test_dataset, test_count = get_dataset(dataset_root_dir=config.test_dir)

    # read the original_dataset in the form of batch
    train_dataset = train_dataset.shuffle(buffer_size=train_count, seed=seed)
    #train_dataset = train_dataset.shuffle(buffer_size=train_count)
    train_dataset = train_dataset.repeat().batch(batch_size=BATCH_SIZE)
    #train_dataset = train_dataset.repeat().batch(batch_size=BATCH_SIZE)
    valid_dataset = valid_dataset.batch(batch_size=VALID_BATCH_SIZE)
    #test_dataset = test_dataset.batch(batch_size=config.BATCH_SIZE)

    #return train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count
    return train_dataset, valid_dataset, train_count, valid_count

def normalize(X_train, X_test):

    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_test

