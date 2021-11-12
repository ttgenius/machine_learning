import hashlib
import numpy as np

def pseudo_random(seed=0xDEADBEEF):
    """Generate an infinite stream of pseudo-random numbers"""
    state = (0xffffffff & seed)/0xffffffff
    while True:
        h = hashlib.sha256()
        h.update(bytes(str(state), encoding='utf8'))
        bits = int.from_bytes(h.digest()[-8:], 'big')
        state = bits >> 32
        r = (0xffffffff & bits)/0xffffffff
        yield r


def bootstrap(dataset, sample_size):
    r = pseudo_random()
    while True:
        new_sample = []
        for i in range(sample_size):
            new_sample.append(dataset[int(next(r)*len(dataset))])
        yield np.array(new_sample)


# dataset = np.array([[1, 0, 2, 3],
#                     [2, 3, 0, 0],
#                     [4, 1, 2, 0],
#                     [3, 2, 1, 0]])
# ds_gen = bootstrap(dataset, 3)
# print(next(ds_gen))
# print(next(ds_gen))


# out = []
# for d in data_points:
#     d_out = []
#     for c in classifiers:
#         d_out.append(c(d))
#     print(max(sorted(d_out), key=d_out.count))
#     out.append(d_out)
# print(out)


def voting_ensemble(classifiers):
    get_votes = lambda p: [c(p) for c in classifiers]
    return lambda p: max(sorted(get_votes(p)), key=get_votes(p).count)

# classifiers = [
#     lambda p: 1 if 1.0 * p[0] < p[1] else 0,
#     lambda p: 1 if 0.9 * p[0] < p[1] else 0,
#     lambda p: 1 if 0.8 * p[0] < p[1] else 0,
#     lambda p: 1 if 0.7 * p[0] < p[1] else 0,
#     lambda p: 1 if 0.5 * p[0] < p[1] else 0,
# ]
# data_points = [(0.2, 0.03), (0.1, 0.12),
#                (0.8, 0.63), (0.9, 0.82)]
# c = voting_ensemble(classifiers)
# for v in data_points:
#     print(c(v))



def bagging_model(learner, dataset, n_models, sample_size):
    boot_data = bootstrap(dataset,sample_size)
    models = [learner(next(boot_data)) for _ in range(n_models)]
    return voting_ensemble(models)




import sklearn.datasets
import sklearn.utils
import sklearn.tree


iris = sklearn.datasets.load_iris()
data, target = sklearn.utils.shuffle(iris.data, iris.target, random_state=1)
train_data, train_target = data[:-5, :], target[:-5]
test_data, test_target = data[-5:, :], target[-5:]
dataset = np.hstack((train_data, train_target.reshape((-1, 1))))

def tree_learner(dataset):
    features, target = dataset[:, :-1], dataset[:, -1]
    model = sklearn.tree.DecisionTreeClassifier(random_state=1).fit(features, target)
    return lambda v: model.predict(np.array([v]))[0]


bagged = bagging_model(tree_learner, dataset, 50, len(dataset)//2)
# Note that we get the first one wrong!
for (v, c) in zip(test_data, test_target):
    print(int(bagged(v)), c)

class weighted_bootstrap:
    def __init__(self, dataset, weights, sample_size):
        assert len(dataset) == len(weights)
        self.dataset = dataset
        self.weights = weights
        self.sample_size = sample_size
        self.random = pseudo_random()

    def __iter__(self):
        return self

    def __next__(self):
        new_samples = []
        total_weights = sum(self.weights)
        for _ in range(self.sample_size):
            p = next(self.random) * total_weights
            running_sum = 0
            for i in range(len(self.weights)):
                running_sum += self.weights[i]
                if running_sum > p:
                    new_samples.append(self.dataset[i])
                    break
        return np.array(new_samples)


# wbs = weighted_bootstrap([1, 2, 3, 4, 5], [1, 1, 1, 1, 1], 5)
# print(next(wbs))
# print(next(wbs))
# print()
# wbs.weights = [1, 1, 1000, 1, 1]
# print(next(wbs))
# print(next(wbs))
def compute_error(y, y_pred, w_i):
    '''
    Calculate the error rate of a weak classifier m. Arguments:
    y: actual target value
    y_pred: predicted value by weak classifier
    w_i: individual weights for each observation

    Note that all arrays should be the same length
    '''
    return (sum(w_i * (np.not_equal(y, y_pred)).astype(int))) / sum(w_i)


def compute_alpha(error):
    '''
    Calculate the weight of a weak classifier m in the majority vote of the final classifier. This is called
    alpha in chapter 10.1 of The Elements of Statistical Learning. Arguments:
    error: error rate from weak classifier m
    '''
    return np.log((1 - error) / error)


def update_weights(w_i, alpha, y, y_pred):
    '''
    Update individual weights w_i after a boosting iteration. Arguments:
    w_i: individual weights for each observation
    y: actual target value
    y_pred: predicted value by weak classifier
    alpha: weight of weak classifier used to estimate y_pred
    '''
    return w_i * np.exp(alpha * (np.not_equal(y, y_pred)).astype(int))

def adaboost(learner, dataset, n_models):
    features, target = dataset[:, :-1], dataset[:, -1]
    print("features", features)
    weights = [1/len(dataset) for _ in range(len(dataset))]
    print("l;en",len(weights))
    models = []
    for t in range(n_models):
        boot_data = weighted_bootstrap(dataset, weights, len(dataset)//2)
        d = next(boot_data)
        model = learner(d)
        models.append(d)
        predictions = np.zeros_like(features)
        w_err_sum = 0
        for row in range(features.shape[0]):
            pred = model(features[row])
            w_err = weights[row] * np.not_equal(target[row],pred).astype(int)
            w_err_sum += w_err
        err = w_err_sum/sum(weights)
        if err == 0 or err >=0.5:
            print("break")
            break
        alpha = np.log((1 - err) / err)
        for row in range(features.shape[0]):
            pred = model(features[row])
            weights[row] = weights[row] * np.exp(alpha * (np.not_equal(target[row], pred).astype(int)))
        print("err",err)
        print("alpha",alpha)
        print("pred",predictions)



import sklearn.datasets
import sklearn.utils
import sklearn.linear_model

digits = sklearn.datasets.load_digits()
data, target = sklearn.utils.shuffle(digits.data, digits.target, random_state=3)
train_data, train_target = data[:-5, :], target[:-5]
test_data, test_target = data[-5:, :], target[-5:]
dataset = np.hstack((train_data, train_target.reshape((-1, 1))))

def linear_learner(dataset):
    features, target = dataset[:, :-1], dataset[:, -1]
    model = sklearn.linear_model.SGDClassifier(random_state=1, max_iter=1000, tol=0.001).fit(features, target)
    return lambda v: model.predict(np.array([v]))[0]

print("Dataset",dataset.shape)
boosted = adaboost(linear_learner, dataset, 10)
for (v, c) in zip(test_data, test_target):
    print(int(boosted(v)), c)

import hashlib
import numpy as np


def pseudo_random(seed=0xDEADBEEF):
    """Generate an infinite stream of pseudo-random numbers"""
    state = (0xffffffff & seed) / 0xffffffff
    while True:
        h = hashlib.sha256()
        h.update(bytes(str(state), encoding='utf8'))
        bits = int.from_bytes(h.digest()[-8:], 'big')
        state = bits >> 32
        r = (0xffffffff & bits) / 0xffffffff
        yield r


class weighted_bootstrap:
    def __init__(self, dataset, weights, sample_size):
        assert len(dataset) == len(weights)
        self.dataset = dataset
        self.weights = weights
        self.sample_size = sample_size
        self.random = pseudo_random()

    def __iter__(self):
        return self

    def __next__(self):
        new_samples = []
        total_weights = sum(self.weights)
        for _ in range(self.sample_size):
            p = next(self.random) * total_weights
            running_sum = 0
            for i in range(len(self.weights)):
                running_sum += self.weights[i]
                if running_sum > p:
                    new_samples.append(self.dataset[i])
                    break
        return np.array(new_samples)


def voting_ensemble(classifiers):
    get_votes = lambda p: [c(p) for c in classifiers]
    return lambda p: max(sorted(get_votes(p)), key=get_votes(p).count)


def adaboost(learner, dataset, n_models):
    weights = [1 / len(dataset) for _ in range(len(dataset))]
    weighted_data = weighted_bootstrap(dataset, weights, len(dataset[0]))
    models = []
    alphas = []
    for t in range(n_models):
        model = learner(next(weighted_data))
        models.append(model)
        error = 0
        for i in range(len(dataset)):
            instance = dataset[i]
            feature = instance[:-1]
            target = instance[-1]
            if model(feature) != target:
                error += weights[i]
        if error == 0:
            alphas.append(-float('inf'))
            break
        elif error >= 0.5:
            alphas.append(np.log((1 - error) / error))
            break
        alphas.append(np.log((1 - error) / error))
        for i in range(len(dataset)):
            instance = dataset[i]
            feature = instance[:-1]
            target = instance[-1]
            if model(feature) == target:
                weights[i] *= (error / (1 - error))
        for i in range(len(weights)):
            weights[i] = weights[i] / sum(weights)

    return voting_ensemble(models)


import sklearn.datasets
import sklearn.utils
import sklearn.linear_model

digits = sklearn.datasets.load_digits()
data, target = sklearn.utils.shuffle(digits.data, digits.target, random_state=3)
train_data, train_target = data[:-5, :], target[:-5]
test_data, test_target = data[-5:, :], target[-5:]
dataset = np.hstack((train_data, train_target.reshape((-1, 1))))


def linear_learner(dataset):
    features, target = dataset[:, :-1], dataset[:, -1]
    model = sklearn.linear_model.SGDClassifier(random_state=1, max_iter=1000, tol=0.001).fit(features, target)
    return lambda v: model.predict(np.array([v]))[0]


boosted = adaboost(linear_learner, dataset, 10)
for (v, c) in zip(test_data, test_target):
    print(int(boosted(v)), c)


