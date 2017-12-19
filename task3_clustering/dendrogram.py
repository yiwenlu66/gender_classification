from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

if __name__ == '__main__':
    data = np.load('data.npz')
    X_train = data['X_train']
    for n_feat in [10, 2]:
        if n_feat == 2:
            pca = PCA(2)
            X_train = pca.fit_transform(X_train)
        for method in ['single', 'average', 'complete']:
            linkage = hierarchy.linkage(X_train, method)
            hierarchy.dendrogram(linkage)
            plt.title('{} features, {} linkage'.format(n_feat, method))
            plt.show()
