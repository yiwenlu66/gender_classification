import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

if __name__ == '__main__':
    data = np.load('data.npz')
    X_train = data['X_train']
    pca = PCA(n_components=3)
    X_train = pca.fit_transform(X_train)
    J = []

    for C in range(1, 7):
        kmeans = KMeans(n_clusters=C, random_state=42, n_jobs=-1)
        kmeans.fit(X_train)
        J.append(kmeans.inertia_)

    plt.plot(range(1, 7), J, marker='o')
    plt.show()

    pca = PCA(n_components=2)
    X_train = pca.fit_transform(X_train)
    kmeans = KMeans(n_clusters=2, random_state=42, n_jobs=-1)
    kmeans.fit(X_train)

    for i in range(len(kmeans.labels_)):
        plt.plot(X_train[i, 0], X_train[i, 1], 'x',
                 color='r' if kmeans.labels_[i] else 'b')
    plt.show()