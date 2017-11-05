import pandas as pd
import numpy as np

def get_dataset(filename):
    df = pd.read_csv(filename, delim_whitespace=True, header=None)
    df[df.columns[-1]].replace({'F': 0, 'M': 1, 'f': 0, 'm': 1}, inplace=True)
    X = df[df.columns[:-1]].values
    y = df[df.columns[-1]].values
    return X, y

if __name__ == '__main__':
    X_train, y_train = get_dataset('dataset3.txt')
    X_test, y_test = get_dataset('dataset4.txt')
    np.savez('data.npz',
             X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
