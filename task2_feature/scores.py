import numpy as np

def euclidean_distance(X1, X2):
    X = [X1, X2]
    m_s = [np.mean(X_i, 0) for X_i in X]
    m = 0.5 * (m_s[0] + m_s[1])
    J = 0
    for i in range(2):
        J += np.linalg.norm(m_s[i] - m) ** 2
        n = X[i].shape[0]
        for k in range(n):
            J += (1 / n) * np.linalg.norm(X[i][k, :] - m) ** 2
    return 0.5 * J


def divergence(X1, X2):
    X = [X1, X2]
    mu = [np.mean(X_i, 0) for X_i in X]
    Sigma = [1 / X_i.shape[0] * sum(
        [np.outer(X_i[j, :], X_i[j, :]) for j in range(X_i.shape[0])]
    ) for X_i in X]
    Sigma_inv = [np.linalg.inv(Sigma_i) for Sigma_i in Sigma]
    J = np.trace(np.matmul(Sigma_inv[0], Sigma[1]))
    J += np.trace(np.matmul(Sigma_inv[1], Sigma[0]))
    J -= 2
    J += np.matmul(np.matmul([mu[0] - mu[1]], sum(Sigma_inv)),
                   np.transpose([mu[0] - mu[1]]))
    return 0.5 * J.reshape((1,))[0]


def t_test(X1, X2):
    assert X1.shape[1] == 1
    assert X2.shape[1] == 1
    m = X1.shape[0]
    n = X2.shape[0]
    x_bar = np.mean(X1)
    y_bar = np.mean(X2)
    S_x = np.std(X1)
    S_y = np.std(X2)
    return abs(x_bar - y_bar) / np.sqrt((n - 1) * S_x ** 2 + (m - 1) * S_y ** 2)
