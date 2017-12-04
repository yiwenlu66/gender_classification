import numpy as np
from sklearn.svm import SVC
from task2_feature.algorithms import *
from task2_feature.scores import *
import sys
import random
sys.path.append('sklearn-genetic')
from genetic_selection import GeneticSelectionCV
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt

def make_plot(X1, X2):
    plt.plot(X1[:, 0], X1[:, 1], 'rx')
    plt.plot(X2[:, 0], X2[:, 1], 'bx')
    plt.show()

if __name__ == '__main__':
    data = np.load('../data.npz')
    X_train, y_train, X_test, y_test = data['X_train'], data['y_train'],\
                                       data['X_test'], data['y_test']

    np.random.seed(42)
    random.seed(42)

    clf = SVC(C=10, kernel='linear')
    clf.fit(X_train, y_train)
    print('Linear SVM:')
    print('Train error: {}'.format(1 - clf.score(X_train, y_train)))
    print('Test error: {}'.format(1 - clf.score(X_test, y_test)))

    X1 = X_train[y_train == 0]
    X2 = X_train[y_train == 1]

    # Branch-and-bound
    print('')
    print('=== Branch-and-bound ===')

    for f in [euclidean_distance, divergence]:
        for d in [2, 3]:
            print('Method: {}'.format(f.__name__))
            features = branch_and_bound(X1, X2, d, f)
            print('Features selected: {}'.format(features))
            clf = SVC(C=10, kernel='linear')
            clf.fit(X_train[:, features], y_train)
            print('Train error: {}'.format(1 - clf.score(
                X_train[:, features], y_train)))
            print('Test error: {}'.format(1 - clf.score(
                X_test[:, features], y_test)))
            if d == 2:
                make_plot(X1[:, features], X2[:, features])


    # single best
    print('')
    print('=== Single best ===')
    for f in [euclidean_distance, divergence, t_test]:
        for d in [1, 2, 3]:
            print('Method: {}'.format(f.__name__))
            features = single_best(X1, X2, d, f)
            print('Features selected: {}'.format(features))
            clf = SVC(C=10, kernel='linear')
            clf.fit(X_train[:, features], y_train)
            print('Train error: {}'.format(1 - clf.score(
                X_train[:, features], y_train)))
            print('Test error: {}'.format(1 - clf.score(
                X_test[:, features], y_test)))
            if d == 2:
                make_plot(X1[:, features], X2[:, features])

    # SBS/SFS
    for g in [sbs, sfs]:
        print('')
        print('=== {} ==='.format(g.__name__.upper()))
        for f in [euclidean_distance, divergence]:
            for d in [2, 3]:
                print('Method: {}'.format(f.__name__))
                features = g(X1, X2, d, f)
                print('Features selected: {}'.format(features))
                clf = SVC(C=10, kernel='linear')
                clf.fit(X_train[:, features], y_train)
                print('Train error: {}'.format(1 - clf.score(
                    X_train[:, features], y_train)))
                print('Test error: {}'.format(1 - clf.score(
                    X_test[:, features], y_test)))
                if d == 2:
                    make_plot(X1[:, features], X2[:, features])

    # GA
    print('')
    print('=== Genetic Algorithm ===')
    clf = SVC(C=10, kernel='linear')
    selector = GeneticSelectionCV(clf,
                                  cv=5,
                                  verbose=1,
                                  scoring="accuracy",
                                  n_population=20,
                                  crossover_proba=0.5,
                                  mutation_proba=0.2,
                                  n_generations=10,
                                  crossover_independent_proba=0.5,
                                  mutation_independent_proba=0.05,
                                  tournament_size=3,
                                  caching=True,
                                  n_jobs=8)
    selector = selector.fit(X_train, y_train)
    features = [i for i in range(X_train.shape[1]) if selector.support_[i]]
    print('Features selected: {}'.format(features))
    clf = SVC(C=10, kernel='linear')
    clf.fit(X_train[:, features], y_train)
    print('Train error: {}'.format(1 - clf.score(
        X_train[:, features], y_train)))
    print('Test error: {}'.format(1 - clf.score(
        X_test[:, features], y_test)))

    # SVM-RFE
    print('')
    print('=== SVM-RFE ===')
    for d in [1, 2, 3]:
        clf = SVC(C=10, kernel='linear')
        selector = RFE(clf, d)
        selector = selector.fit(X_train, y_train)
        features = [i for i in range(X_train.shape[1]) if selector.support_[i]]
        print('Features selected: {}'.format(features))
        clf = SVC(C=10, kernel='linear')
        clf.fit(X_train[:, features], y_train)
        print('Train error: {}'.format(1 - clf.score(
            X_train[:, features], y_train)))
        print('Test error: {}'.format(1 - clf.score(
            X_test[:, features], y_test)))
        if d == 2:
            make_plot(X1[:, features], X2[:, features])

    # K-L
    print('')
    print('=== K-L ===')
    pipe = Pipeline([('scaling', StandardScaler()),
                     ('pca', PCA(n_components=2))])
    X_train = pipe.fit_transform(X_train)
    X_test = pipe.transform(X_test)
    clf = SVC(C=10, kernel='linear')
    clf.fit(X_train, y_train)
    print('Train error: {}'.format(1 - clf.score(X_train, y_train)))
    print('Test error: {}'.format(1 - clf.score(X_test, y_test)))
    make_plot(pipe.transform(X1), pipe.transform(X2))
