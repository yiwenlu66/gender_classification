import numpy as np
from sklearn.metrics import accuracy_score as accuracy
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import check_X_y, check_array
from sklearn.utils.fixes import in1d
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted

from task1_classification.common import load_data


class GaussianNonNaiveBayes(GaussianNB):
    """
    Gaussian Non-Naive Bayes

    Parameters
    ----------
    priors : array-like, shape (n_classes,)
        Prior probabilities of the classes. If specified the priors are not
        adjusted according to the data.

    Attributes
    ----------
    class_prior_ : array, shape (n_classes,)
        probability of each class.

    class_count_ : array, shape (n_classes,)
        number of training samples observed in each class.

    theta_ : array, shape (n_classes, n_features)
        mean of each feature per class

    sigma_ : array, shape (n_classes, n_features, n_features)
        covariance matrix per class
    """
    def __init__(self, priors=None):
        GaussianNB.__init__(self, priors)

    def fit(self, X, y, sample_weight=None):
        """Fit Gaussian Bayes according to X, y

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target values.

        sample_weight: not supported

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y)
        return self._partial_fit(X, y, np.unique(y), _refit=True)

    @staticmethod
    def _update_mean_variance(n_past, mu, var, X, sample_weight=None):
        raise NotImplementedError

    @staticmethod
    def _update_mean_covariance(X):
        """Compute d-dimensional Gaussian mean and covariance matrix using MLE.

        Returns
        -------
        mu : array-like, shape (d,)
            Mean for the d-dimensional Gaussian.

        cov : array-like, shape (d, d)
            Covariance matrix for the d-dimensional Gaussian.
        """
        if X.shape[0] == 0:
            raise ValueError('Number of samples must be non-zeros.')

        mu = np.mean(X, axis=0)
        centered_X = X - mu.reshape((1, -1))
        cov = np.dot(centered_X.T, centered_X) / X.shape[0]

        return mu, cov

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        raise NotImplementedError

    def _partial_fit(self, X, y, classes=None, _refit=True, sample_weight=None):
        """Actual implementation of Gaussian Bayes fitting.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target values.

        classes : array-like, shape (n_classes,)
            List of all the classes that can possibly appear in the y vector.

        _refit: bool, must be True

        sample_weight : not supported

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y)

        # If the ratio of data variance between dimensions is too small, it
        # will cause numerical errors. To address this, we artificially
        # boost the variance by epsilon, a small fraction of the standard
        # deviation of the largest dimension.
        epsilon = 1e-9 * np.var(X, axis=0).max()

        if classes is None or not _refit:
            raise NotImplementedError('Partial fit is not supported.')

        self.classes_ = unique_labels(classes)

        n_features = X.shape[1]
        n_classes = len(self.classes_)
        self.theta_ = np.zeros((n_classes, n_features))
        self.sigma_ = np.zeros((n_classes, n_features, n_features))

        self.class_count_ = np.zeros(n_classes, dtype=np.float64)

        # Initialise the class prior
        n_classes = len(self.classes_)
        # Take into account the priors
        if self.priors is not None:
            priors = np.asarray(self.priors)
            # Check that the provide prior match the number of classes
            if len(priors) != n_classes:
                raise ValueError('Number of priors must match number of'
                                 ' classes.')
            # Check that the sum is 1
            if priors.sum() != 1.0:
                raise ValueError('The sum of the priors should be 1.')
            # Check that the prior are non-negative
            if (priors < 0).any():
                raise ValueError('Priors must be non-negative.')
            self.class_prior_ = priors
        else:
            # Initialize the priors to zeros for each class
            self.class_prior_ = np.zeros(len(self.classes_),
                                         dtype=np.float64)

        classes = self.classes_

        unique_y = np.unique(y)
        unique_y_in_classes = in1d(unique_y, classes)

        if not np.all(unique_y_in_classes):
            raise ValueError("The target label(s) %s in y do not exist in the "
                             "initial classes %s" %
                             (unique_y[~unique_y_in_classes], classes))

        for y_i in unique_y:
            i = classes.searchsorted(y_i)
            X_i = X[y == y_i, :]

            if sample_weight is not None:
                sw_i = sample_weight[y == y_i]
                N_i = sw_i.sum()
            else:
                sw_i = None
                N_i = X_i.shape[0]

            theta, sigma = self._update_mean_covariance(X_i)

            self.theta_[i, :] = theta
            self.sigma_[i, :, :] = sigma
            self.class_count_[i] += N_i

        self.sigma_[:, :, :] += epsilon

        # Update if only no priors is provided
        if self.priors is None:
            # Empirical prior, with sample_weight taken into account
            self.class_prior_ = self.class_count_ / self.class_count_.sum()

        return self

    def _joint_log_likelihood(self, X):
        check_is_fitted(self, "classes_")

        X = check_array(X)
        joint_log_likelihood = []
        for i in range(np.size(self.classes_)):
            jointi = np.log(self.class_prior_[i])
            n_ij = -0.5 * (X.shape[1] * np.log(2 * np.pi)
                           + np.log(np.linalg.det(self.sigma_[i, :, :])))
            X_centered = X - self.theta_[i, :].reshape((1, -1))
            n_ij -= 0.5 * (np.dot(X_centered,
                                  np.linalg.inv(self.sigma_[i, :, :])
                                  ) * X_centered).sum(1)
            joint_log_likelihood.append(jointi + n_ij)

        joint_log_likelihood = np.array(joint_log_likelihood).T
        return joint_log_likelihood


if __name__ == '__main__':
    print('\t'.join(['Samples', 'Feat.', 'Non-Naive', 'Naive']))
    for (n_samples, n_feat, X_train, y_train, X_test, y_test) in load_data():
        clf_non_naive = GaussianNonNaiveBayes()
        clf_naive = GaussianNB()
        scores = []
        for clf in (clf_non_naive, clf_naive):
            clf.fit(X_train, y_train.ravel())
            scores.append(accuracy(y_test, clf.predict(X_test)))
        print('\t\t'.join([str(n_samples), str(n_feat)] +
                        ['{:.3f}'.format(1 - score) for score in scores]))
