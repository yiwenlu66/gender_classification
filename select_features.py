import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score as accuracy


if __name__ == '__main__':
    data = np.load('data.npz')
    X_train, X_test, y_train, y_test = \
        data['X_train'], data['X_test'], data['y_train'], data['y_test']
    n_features = X_train.shape[1]

    error_rate = np.full((n_features, n_features), np.nan)

    for i in range(n_features):
        for j in range(i + 1, n_features):
            X_train_2feat = X_train[:, ((i, j))]
            X_test_2feat = X_test[:, ((i, j))]
            clf = GaussianNB(priors=[0.5, 0.5])
            clf.fit(X_train_2feat, y_train)
            error_rate[i, j] = 1 - accuracy(y_test, clf.predict(X_test_2feat))
            print("{}, error rate: {}".format((i, j), error_rate[i, j]))

    np.save('error_rate_2feat.npy', error_rate)
    best_combo = np.unravel_index(np.nanargmin(error_rate), error_rate.shape)
    print('Best combo: {}, error rate: {}'.format(best_combo,
                                                  error_rate[best_combo]))
