import numpy as np
from numpy.random import permutation

SMALL_SAMPLES_PER_CLASS = 10
SELECTED_FEATURES = (0, 2)

if __name__ == '__main__':
    np.random.seed(42)      # make the result reproducible
    data = np.load('data.npz')
    X_train, X_test, y_train, y_test = \
        data['X_train'], data['X_test'], data['y_train'], data['y_test']

    # select 10 samples from each class
    def select_samples(clazz):
        X_sub = X_train[y_train == clazz]
        return X_sub[permutation(len(X_sub))[:SMALL_SAMPLES_PER_CLASS]]

    X_train_small = np.concatenate([select_samples(i) for i in [0, 1]])
    y_train_small = np.concatenate([np.zeros((10, 1)), np.ones((10, 1))])

    # select 2 features from each dataset
    X_train_2feat, X_train_small_2feat, X_test_2feat = map(
        lambda X: X[:, SELECTED_FEATURES],
        [X_train, X_train_small, X_test]
    )

    np.savez('data_preprocessed.npz',
             X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
             X_train_small=X_train_small, y_train_small=y_train_small,
             X_train_2feat=X_train_2feat,
             X_train_small_2feat=X_train_small_2feat,
             X_test_2feat=X_test_2feat)
