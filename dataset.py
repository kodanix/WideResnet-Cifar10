from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import StratifiedShuffleSplit

class DataSet(ABC):
    
    def __init__(self, multi) -> None:
        self.multi_flag = multi

    def is_multi(self):
        return self.multi_flag

    @abstractmethod
    def get_train(self):
        pass

    @abstractmethod
    def get_test(self):
        pass

    @abstractmethod
    def get_val(self):
        pass

    @abstractmethod
    def get_input_shape(self):
        pass

    @abstractmethod
    def get_nb_classes(self):
        pass

    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def get_bound(self):
        pass


class CIFAR10(DataSet):
    
    def __init__(self, val_size=1000, onehot=False, seed=9) -> None:
        super().__init__(False)
        self.rnd = np.random.RandomState(seed)
        cifar10 = tf.keras.datasets.cifar10
        self.onehot = onehot
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train, x_test = x_train.astype(np.float32), x_test.astype(np.float32)
        y_train = y_train.flatten()
        y_test = y_test.flatten()
        self.x_train = x_train
        if onehot:
            self.y_train = keras.utils.to_categorical(y_train)
            self.y_test = keras.utils.to_categorical(y_test)
        else:
            self.y_train = np.array(y_train, np.int64)
            self.y_test = np.array(y_test, np.int64)
        self.x_test = x_test
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
        train_idxs = np.arange(0, x_train.shape[0])
        train_idxs, val_idxs = next(sss.split(train_idxs, y_train))
        self.x_val = self.x_train[val_idxs]
        self.y_val = self.y_train[val_idxs]
        self.x_train = self.x_train[train_idxs]
        self.y_train = self.y_train[train_idxs]

    def get_bound(self):
        return (0., 255.)

    def get_input_shape(self):
        return self.x_train.shape[1:]

    def get_nb_classes(self):
        return np.unique(self.y_train).shape[0] if not self.onehot else self.y_train.shape[1]

    def get_train(self):
        return self.x_train, self.y_train

    def get_test(self):
        return self.x_test, self.y_test

    def get_val(self):
        return self.x_val, self.y_val

    def get_name(self):
        return 'CIFAR10'