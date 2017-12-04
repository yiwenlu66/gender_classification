import numpy as np

def branch_and_bound(X1, X2, d, score_func):
    D = X1.shape[1]
    assert d < D
    assert X2.shape[1] == D

    def get_score(feature_set):
        return score_func(X1[:, feature_set], X2[:, feature_set])

    feature_set = list(range(d))
    search_path = list(range(d, D))
    state = 'backtrace'
    lower_bound = get_score(feature_set)
    best_features = feature_set[:]

    while True:
        if state == 'backtrace':
            if not search_path[-1]:
                break
            if len(search_path) == 1 or search_path[-1] > search_path[-2] + 1:
                search_path[-1] -= 1
                feature_set[feature_set.index(search_path[-1])] += 1
                state = 'forward'
                continue
            feature_set.append(search_path.pop())
            feature_set.sort()
        else:
            if len(search_path) == D - d:
                score = get_score(feature_set)
                if score > lower_bound:
                    lower_bound = score
                    best_features = feature_set[:]
                state = 'backtrace'
                continue
            if get_score(feature_set) <= lower_bound:
                state = 'backtrace'
                continue
            next_node = d + len(search_path)
            search_path.append(next_node)
            feature_set.remove(next_node)

    return best_features


def single_best(X1, X2, d, score_func):
    D = X1.shape[1]
    assert d < D
    assert X2.shape[1] == D

    scores = np.array([score_func(
        X1[:, i].reshape(-1, 1), X2[:, i].reshape(-1, 1)) for i in range(D)])
    return scores.argsort()[-d:]

def sfs(X1, X2, d, score_func):
    D = X1.shape[1]
    assert d < D
    assert X2.shape[1] == D

    def get_score(feature_set):
        return score_func(X1[:, feature_set], X2[:, feature_set])

    used_features = []
    unused_features = list(range(D))

    while len(used_features) < d:
        best_feature = None
        best_score = -1
        for feature in unused_features:
            score = get_score(used_features + [feature])
            if score > best_score:
                best_score = score
                best_feature = feature
        used_features.append(best_feature)
        unused_features.remove(best_feature)

    return used_features

def sbs(X1, X2, d, score_func):
    D = X1.shape[1]
    assert d < D
    assert X2.shape[1] == D

    def get_score(feature_set):
        return score_func(X1[:, feature_set], X2[:, feature_set])

    used_features = list(range(D))

    while len(used_features) > d:
        best_feature = None
        best_score = -1
        for feature in used_features:
            new_features = used_features[:]
            new_features.remove(feature)
            score = get_score(new_features)
            if score > best_score:
                best_score = score
                best_feature = feature
        used_features.remove(best_feature)

    return used_features
