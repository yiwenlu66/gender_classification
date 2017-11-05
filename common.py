import numpy as np


def load_data():
    data = np.load('data_preprocessed.npz')
    X_train, X_test, y_train, y_test, X_train_small, y_train_small, \
    X_train_2feat, X_train_small_2feat, X_test_2feat = \
        data['X_train'], data['X_test'], data['y_train'], data['y_test'], \
        data['X_train_small'], data['y_train_small'], \
        data['X_train_2feat'], data['X_train_small_2feat'], data['X_test_2feat']

    datasets = [
        (20, 10, X_train_small, y_train_small, X_test, y_test),
        (20, 2, X_train_small_2feat, y_train_small, X_test_2feat, y_test),
        ('all', 10, X_train, y_train, X_test, y_test),
        ('all', 2, X_train_2feat, y_train, X_test_2feat, y_test)
    ]

    return datasets