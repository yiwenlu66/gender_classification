import numpy as np
from sklearn.metrics import accuracy_score as accuracy
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from task1_classification.common import load_data

if __name__ == '__main__':
    Cs = np.logspace(-3, 1, 9)
    gammas = np.logspace(-3, 1, 9)

    # Linear SVM
    for (n_samples, n_feat, X_train, y_train, X_test, y_test) in load_data():
        if n_samples != 'all':
            # random shuffle
            np.random.seed(42)
            p = np.random.permutation(n_samples)
            X_train = X_train[p, :]
            y_train = y_train[p]
        svc = SVC(kernel='linear', random_state=42)
        clf = GridSearchCV(svc, {'C': Cs})
        clf.fit(X_train, y_train.ravel())
        print(clf.best_params_)
        print(1 - accuracy(y_test, clf.best_estimator_.predict(X_test)))

    # RBF SVM
    for (n_samples, n_feat, X_train, y_train, X_test, y_test) in load_data():
        if n_samples != 'all':
            # random shuffle
            p = np.random.permutation(n_samples)
            X_train = X_train[p, :]
            y_train = y_train[p]
        svc = SVC(kernel='rbf', random_state=42)
        clf = GridSearchCV(svc, {'C': Cs, 'gamma': gammas})
        clf.fit(X_train, y_train.ravel())
        print(clf.best_params_)
        print(1 - accuracy(y_test, clf.best_estimator_.predict(X_test)))
