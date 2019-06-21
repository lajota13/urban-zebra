import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Reshape, UpSampling2D

from sklearn import metrics
import tensorflow as tf


class Autoencoder:
    def __init__(self, anomalous_digits: list):
        assert type(anomalous_digits) is list
        assert not not anomalous_digits
        assert set(anomalous_digits) < set(range(10))
        self.anomalous_digits = np.unique(anomalous_digits)
        self.x_train_normal, self.x_test_normal, self.x_test_anomaly = self.get_dataset()
        self.graph = None
        self.roc_curve = None

    def get_dataset(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train_normal_indeces = (y_train[:, None] != self.anomalous_digits[None, :]).all(axis=1)
        x_train_normal = x_train[x_train_normal_indeces, :, :]

        x_test_normal_indeces = (y_test[:, None] != self.anomalous_digits[None, :]).all(axis=1)
        x_test_normal = x_test[x_test_normal_indeces, :, :]
        x_test_anomaly = x_test[np.logical_not(x_test_normal_indeces), :, :]
        return x_train_normal, x_test_normal, x_test_anomaly

    def build_graph(self):
        pass

    def train(self):
        pass

    def get_roc(self):
        pass

    def show_prediction(self):
        pass

    def save_weights(self):
        pass

    def load_weights(self):
        pass


if __name__ == "__main__":
    ae = Autoencoder(anomalous_digits=[0])

