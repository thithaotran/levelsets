
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np

def get_datasets_MPG_FF():

    dataset_path = keras.utils.get_file("auto-mpg.data",
                                        "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
    # dataset_path
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                    'Acceleration', 'Model Year', 'Origin']
    dataset = pd.read_csv(dataset_path, names=column_names,
                          na_values="?", comment='\t',
                          sep=" ", skipinitialspace=True)

    # dataset.tail()
    dataset = dataset.dropna()
    dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
    dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)
    train_labels = train_dataset.pop('MPG')
    test_labels = test_dataset.pop('MPG')

    def norm(x):
        return (x - train_dataset.mean(axis=0)) / train_dataset.std(axis=0)

    normed_train_data = norm(train_dataset)
    normed_test_data = norm(test_dataset)
    X_train, y_train, X_test, y_test = normed_train_data, train_labels, normed_test_data, test_labels

    X_train, y_train, X_test, y_test = map(lambda x: x.astype(np.float32), [X_train, y_train, X_test, y_test])
    X_train, y_train, X_test, y_test = map(lambda x: tf.convert_to_tensor(x), [X_train, y_train, X_test, y_test])

    ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    ds_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    return (ds_train, ds_test)


def get_datasets_IRIS_FF():
    dataset_path_train = keras.utils.get_file("iris_train.csv",
                                              "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
    dataset_path_test = keras.utils.get_file("iris_test.csv",
                                             "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']

    train_dataset = pd.read_csv(dataset_path_train, names=column_names, skiprows=1)
    train_labels = pd.get_dummies(train_dataset.pop('species'))
    test_dataset = pd.read_csv(dataset_path_test, names=column_names, skiprows=1)
    test_labels = pd.get_dummies(test_dataset.pop('species'))

    def norm(x):
        return (x - train_dataset.mean(axis=0)) / train_dataset.std(axis=0)

    normed_train_data = norm(train_dataset)
    normed_test_data = norm(test_dataset)
    X_train, y_train, X_test, y_test = normed_train_data, train_labels, normed_test_data, test_labels

    X_train, y_train, X_test, y_test = map(lambda x: x.astype(np.float32), [X_train, y_train, X_test, y_test])
    X_train, y_train, X_test, y_test = map(lambda x: tf.convert_to_tensor(x), [X_train, y_train, X_test, y_test])

    ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    ds_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    return (ds_train, ds_test)


def get_datasets_MNIST_CONV():
    handle_dataset = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = handle_dataset.load_data()
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    # standardizing the pixels to a [0, 1.0] range, as suggested in the literature
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

    X_train, y_train, X_test, y_test = map(lambda x: x.astype(np.float32), [X_train, y_train, X_test, y_test])
    X_train, y_train, X_test, y_test = map(lambda x: tf.convert_to_tensor(x), [X_train, y_train, X_test, y_test])

    ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    ds_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    return (ds_train, ds_test)

def get_datasets_MNIST_FF():
    handle_dataset = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = handle_dataset.load_data()
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    # standardizing the pixels to a [0, 1.0] range, as suggested in the literature
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    X_train, y_train, X_test, y_test = map(lambda x: x.astype(np.float32), [X_train, y_train, X_test, y_test])
    X_train, y_train, X_test, y_test = map(lambda x: tf.convert_to_tensor(x), [X_train, y_train, X_test, y_test])

    ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    ds_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    return (ds_train, ds_test)


def get_datasets_CIFAR10_CONV():
    handle_dataset = tf.keras.datasets.cifar10
    (X_train, y_train), (X_test, y_test) = handle_dataset.load_data()
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    # standardizing the pixels to a [0, 1.0] range, as suggested in the literature
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    X_train = X_train.reshape((X_train.shape[0], 32, 32, 3))
    X_test = X_test.reshape((X_test.shape[0], 32, 32, 3))

    X_train, y_train, X_test, y_test = map(lambda x: x.astype(np.float32), [X_train, y_train, X_test, y_test])
    X_train, y_train, X_test, y_test = map(lambda x: tf.convert_to_tensor(x), [X_train, y_train, X_test, y_test])

    ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    ds_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    return (ds_train, ds_test)

if __name__ == "__main__":
    pass