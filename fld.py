from common import load_data
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score as accuracy

if __name__ == '__main__':
    print('\t'.join(['Samples', 'Feat.', 'FLD']))
    for (n_samples, n_feat, X_train, y_train, X_test, y_test) in load_data():
        clf = LinearDiscriminantAnalysis()
        clf.fit(X_train, y_train.ravel())
        error = 1 - accuracy(y_test, clf.predict(X_test))
        print('\t\t'.join([str(n_samples), str(n_feat),
                           '{:.3f}'.format(error)]))
