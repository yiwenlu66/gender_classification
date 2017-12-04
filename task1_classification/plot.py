import numpy as np
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from task1_classification.common import load_data


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, dashed=False, label=None, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    contour = ax.contour(xx, yy, Z, **params)
    if label:
        contour.collections[0].set_label(label)
    if dashed:
        for c in contour.collections:
            c.set_dashes([(0, (2.0, 2.0))])


if __name__ == '__main__':
    _, _, X_train, y_train, X_test, y_test = load_data()[-1]
    _, _, X_train_small, y_train_small, _, _ = load_data()[1]
    y_train_small = y_train_small.ravel()

    # plot dots
    X_train_0, X_train_1 = X_train[y_train == 0], X_train[y_train == 1]
    X_train_small_0, X_train_small_1 = \
        X_train_small[y_train_small == 0], X_train_small[y_train_small == 1]
    X_test_0, X_test_1 = X_test[y_test == 0], X_test[y_test == 1]
    plt.xlim([np.min(X_train[:, 0]), np.max(X_train[:, 0])])
    plt.ylim([np.min(X_train[:, 1]), np.max(X_train[:, 1])])
    plt.plot(X_train_0[:, 0], X_train_0[:, 1], 'b.', ms=2, label='Train (0)')
    plt.plot(X_train_1[:, 0], X_train_1[:, 1], 'r.', ms=2, label='Train (1)')
    plt.plot(X_test_0[:, 0], X_test_0[:, 1], 'bx', ms=4, label='Test (0)')
    plt.plot(X_test_1[:, 0], X_test_1[:, 1], 'rx', ms=4, label='Test (1)')
    plt.plot(X_train_small_0[:, 0], X_train_small_0[:, 1],
             'bd', label='Train (small, 0)')
    plt.plot(X_train_small_1[:, 0], X_train_small_1[:, 1],
             'rd', label='Train (small, 1)')

    # FLD

    clf_fld = LinearDiscriminantAnalysis()
    clf_fld.fit(X_train, y_train.ravel())
    x = np.array([np.min(X_train[:, 0]), np.max(X_train[:, 0])])
    y = 1.0 / clf_fld.coef_[0, 1] * (
        - clf_fld.coef_[0, 0] * x - clf_fld.intercept_)
    plt.plot(x, y, 'g-', linewidth=2, antialiased=False, label='FLD')

    clf_fld = LinearDiscriminantAnalysis()
    clf_fld.fit(X_train_small, y_train_small.ravel())
    x = np.array([np.min(X_train[:, 0]), np.max(X_train[:, 0])])
    y = 1.0 / clf_fld.coef_[0, 1] * (
        - clf_fld.coef_[0, 0] * x - clf_fld.intercept_)
    plt.plot(x, y, 'g--', linewidth=2, antialiased=False)

    xx, yy = make_meshgrid(X_train[:, 0], X_train[:, 1], h=0.2)

    # Bayesian

    clf_bayes = GaussianNB()
    clf_bayes.fit(X_train, y_train.ravel())
    plot_contours(plt, clf_bayes, xx, yy, label='Naive Bayes', colors='c')

    clf_bayes = GaussianNB()
    clf_bayes.fit(X_train_small, y_train_small.ravel())
    plot_contours(plt, clf_bayes, xx, yy, dashed=True, colors='c')

    # Linear SVM

    clf_linsvm = SVC(C=10, kernel='linear')
    clf_linsvm.fit(X_train, y_train.ravel())
    plot_contours(plt, clf_linsvm, xx, yy, label='Linear SVM', colors='m')

    clf_linsvm = SVC(C=10, kernel='linear')
    clf_linsvm.fit(X_train_small, y_train_small.ravel())
    plot_contours(plt, clf_linsvm, xx, yy, dashed=True, colors='m')

    # RBF SVM

    clf_rbfsvm = SVC(C=10, gamma=.01, kernel='rbf')
    clf_rbfsvm.fit(X_train, y_train.ravel())
    plot_contours(plt, clf_rbfsvm, xx, yy, label='RBF SVM', colors='y')

    clf_rbfsvm = SVC(C=10, gamma=.01, kernel='rbf')
    clf_rbfsvm.fit(X_train_small, y_train_small.ravel())
    plot_contours(plt, clf_rbfsvm, xx, yy, dashed=True, colors='y')

    # MLP

    clf_mlp = MLPClassifier(
        hidden_layer_sizes=[350, 228, 210], random_state=42,
        activation='logistic',
        max_iter=10000,
    )
    clf_mlp.fit(X_train, y_train.ravel())
    plot_contours(plt, clf_mlp, xx, yy, label='MLP', colors='tab:purple')

    clf_mlp = MLPClassifier(
        hidden_layer_sizes=[350, 228, 210], random_state=42,
        activation='logistic',
        max_iter=10000,
    )
    clf_mlp.fit(X_train_small, y_train_small.ravel())
    plot_contours(plt, clf_mlp, xx, yy, dashed=True, colors='tab:purple')

    plt.legend()
    plt.show()
