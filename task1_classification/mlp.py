from sklearn.metrics import accuracy_score as accuracy
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from task1_classification.common import load_data

RANDOM_SEED = 42
MAX_NODES_PER_LAYER = 100
DEPTH_THRESH = 5

def get_error(hidden_layer_sizes, X_train, y_train, X_test, y_test):
    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation='logistic',
        random_state=RANDOM_SEED
    )
    clf.fit(X_train, y_train.ravel())
    error = 1 - accuracy(y_test, clf.predict(X_test))
    return error


def find_best_structure(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.3,
                                                      random_state=RANDOM_SEED)
    hidden_layer_sizes = [MAX_NODES_PER_LAYER]
    best_error = 1
    best_structure = hidden_layer_sizes[:]
    n_increase = 0
    # step 1: increase depth
    while True:
        error = get_error(hidden_layer_sizes, X_train, y_train, X_val, y_val)
        print('{}: {}'.format(hidden_layer_sizes, error))
        if error <= best_error or n_increase < DEPTH_THRESH:
            hidden_layer_sizes.append(MAX_NODES_PER_LAYER)
            if error < best_error:
                best_error = error
                best_structure = hidden_layer_sizes[:]
                n_increase = 0
            else:
                n_increase += 1
        else:
            hidden_layer_sizes = hidden_layer_sizes[:- n_increase - 1]
            n_increase = 0
            break
    # step 2: decrease width
    for idx_layer in range(len(hidden_layer_sizes)):
        while True:
            hidden_layer_sizes[idx_layer] -= 1
            error = get_error(
                hidden_layer_sizes, X_train, y_train, X_val, y_val)
            print('{}: {}'.format(hidden_layer_sizes, error))
            if error <= best_error:
                best_error = error
                best_structure = hidden_layer_sizes[:]
            else:
                if hidden_layer_sizes[idx_layer] == 2:
                    hidden_layer_sizes[idx_layer] = best_structure[idx_layer]
                    break

    print('{}: {}'.format(best_structure, best_error))
    return best_structure, best_error


if __name__ == '__main__':
    data = load_data()

    # try to find best structure for 2 features
    _, _, X_train, y_train, X_test, y_test = data[-1]
    best_structure_2, best_error_2 = find_best_structure(X_train, y_train)

    # try to find best structure for 10 features
    _, _, X_train, y_train, X_test, y_test = data[-2]
    best_structure_10, best_error_10 = find_best_structure(X_train, y_train)

    print(best_structure_2, best_structure_10)

    # test on all datasets
    for (n_samples, n_feat, X_train, y_train, X_test, y_test) in load_data():
        hidden_layer_sizes = best_structure_2 if n_feat == 2 else \
            best_structure_10
        error = get_error(hidden_layer_sizes, X_train, y_train, X_test, y_test)
        print('\t\t'.join([str(n_samples), str(n_feat),
                           '{:.3f}'.format(error)]))
