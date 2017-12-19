import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

if __name__ == '__main__':
    data = np.load('data.npz')
    X_train = data['X_train']
    pca = PCA()
    pca.fit(X_train)

    # plot variance
    plt.plot(range(1, 11), pca.explained_variance_, marker='o')
    plt.show()
