from collections import namedtuple


class ConfusionMatrix(namedtuple("ConfusionMatrix",
                                 "true_positive false_negative "
                                 "false_positive true_negative")):

    def __str__(self):
        elements = [self.true_positive, self.false_negative,
                    self.false_positive, self.true_negative]
        return ("{:>{width}} " * 2 + "\n" + "{:>{width}} " * 2).format(
            *elements, width=max(len(str(e)) for e in elements))


def confusion_matrix(classifier, dataset):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for classes, label in dataset:
        prediction = classifier(classes)
        if prediction:
            if prediction == label:
                tp += 1
            else:
                fp += 1
        else:
            if prediction == label:
                tn += 1
            else:
                fn += 1
    return ConfusionMatrix(true_positive=tp, false_negative=fn, false_positive=fp, true_negative=tn)
#
# dataset = [
#     ((0.8, 0.2), 1),
#     ((0.4, 0.3), 1),
#     ((0.1, 0.35), 0),
# ]
# print(confusion_matrix(lambda x: 1, dataset))
# print()
# print(confusion_matrix(lambda x: 1 if x[0] + x[1] > 0.5 else 0, dataset))

#q2
from collections import namedtuple


def roc_non_dominated2(classifiers):
    # Example similar to the lecture notes
    non_dom = []
    if classifiers is not None and len(classifiers) > 0:
        rates_dict = dict()
        for label, cfm in classifiers:
            print("cfm",cfm)
            tpr = cfm.true_positive / (cfm.true_positive + cfm.false_negative)
            fpr = cfm.false_positive / (cfm.false_positive + cfm.true_negative)
            print(tpr, fpr)
            rates_dict[label] = (tpr, fpr)

        dom_dict = {label: [] for label in rates_dict.keys()}
        for a_label, a_rates in rates_dict.items():
            a_tpr, a_fpr = a_rates
            for b_label, b_rates in rates_dict.items():
                if a_label != b_label:
                    b_tpr, b_fpr = b_rates
                    if a_tpr > b_tpr and a_fpr < b_fpr:
                        dom_dict[b_label].append(a_label)  #b domined by a
        for label, domed_by in dom_dict.items():
            if domed_by == []:
                non_dom.append(label)
        print(non_dom)
    return non_dom

from collections import namedtuple

class ConfusionMatrix(namedtuple("ConfusionMatrix",
                                 "true_positive false_negative "
                                 "false_positive true_negative")):
    pass

def a_dom_b(a_cfm, b_cfm):
    a_tp, a_fn, a_fp, a_tn = a_cfm
    b_tp, b_fn, b_fp, b_tn = b_cfm

    a_tpr = a_tp / (a_tp + a_fn)
    a_fpr = a_fp / (a_fp + a_tn)

    b_tpr = b_tp / (b_tp + b_fn)
    b_fpr = b_fp / (b_fp + b_tn)

    return a_tpr > b_tpr and a_fpr < b_fpr


def is_domed(classifier, classifers):
    for c2 in classifers:
        if a_dom_b(c2[1], classifier[1]):
            return True
    return False


def roc_non_dominated(classifers):
    non_dom = []
    for c in classifers:
        if not is_domed(c, classifers):
            non_dom.append(c)
    return non_dom




# tp fp
# tn fn
# classifiers = [
#     ("Red", ConfusionMatrix(60, 40,
#                             20, 80)),
#     ("Green", ConfusionMatrix(40, 60,
#                               30, 70)),
#     ("Blue", ConfusionMatrix(80, 20,
#                              50, 50)),
# ]
# print(sorted(label for (label, _) in roc_non_dominated(classifiers)))
# #['Blue', 'Red']
#
# classifiers = []
# with open('roc._small.data.txt') as f:
#     for line in f.readlines():
#         label, tp, fn, fp, tn = line.strip().split(",")
#         classifiers.append((label,
#                             ConfusionMatrix(int(tp), int(fn),
#                                             int(fp), int(tn))))
# print(sorted(label for (label, _) in roc_non_dominated(classifiers)))
#


def initialize_centroids(points, k):
    """returns k centroids from the initial points"""
    centroids = points.copy()
    np.random.shuffle(centroids)
    return centroids[:k]

#q3
import numpy as np

def closest_centroid(points, centroids):
    """returns an array containing the index to the nearest centroid for each point"""
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)


def move_centroids(points, closest, centroids):
    """returns the new centroids assigned from the points closest to them"""
    new_centroids = np.empty_like(centroids)
    #cluster = [ _ for _ in range(len(centroids))]
    for k in range(len(centroids)):
        if len(points[closest==k]) > 0:
            #print("points[closet]",points[closest], "poinst[k]",points[k],"points[closet=k]",points[closest ==k])
            new_centroids[k] = points[closest == k].mean(axis=0)
            #cluster[k]=points[closest==k]
            #print("cluster[k]",cluster[k])
           # print("cluster",cluster)
        else:
            new_centroids[k] = centroids[k]
   # print("new_centriods",new_centroids)
    return new_centroids


def k_means(dataset, centroids):
    centroidss = np.asarray(centroids)
    while True:
        closest = closest_centroid(dataset, centroidss)
        new_centroids = move_centroids(dataset, closest, centroidss)
        if np.all(new_centroids == centroidss):
            break
        centroidss = new_centroids
    return new_centroids



import hashlib
import numpy as np
from scipy.spatial import distance as sdist

def pseudo_random(seed=0xdeadbeef):
    """generate an infinite stream of pseudo-random numbers"""
    state = (0xffffffff & seed)/0xffffffff
    while True:
        h = hashlib.sha256()
        h.update(bytes(str(state), encoding='utf8'))
        bits = int.from_bytes(h.digest()[-8:], 'big')
        state = bits >> 32
        r = (0xffffffff & bits)/0xffffffff
        yield r


def cluster_points(centroids, dataset):
    clusters = [_ for _ in range(len(centroids))]
    closest = closest_centroid(dataset, centroids)
    for k in range(len(centroids)):
        clusters[k] = dataset[closest == k]
    return clusters


def separation(clusters):
    min_intercluster_dists = []
    for i, c1 in enumerate(clusters):
        for j, c2 in enumerate(clusters):
            if i != j:
                min_intercluster_dists.append(sdist.cdist(c1, c2).min())
    return sum(min_intercluster_dists) / len(clusters)


def compactness(clusters):
    max_intracluster_dists = []
    for c in clusters:
        intracluster_dists = []
        for p1 in c:
            for p2 in c:
                intracluster_dists.append(np.linalg.norm(p1-p2))
        max_intracluster_dists.append(max(intracluster_dists))
    return sum(max_intracluster_dists) / len(clusters)


def goodness(clusters):
    return separation(clusters) / compactness(clusters)


def generate_random_vector(bounds, r):
    return np.array([(high - low) * next(r) + low for low, high in bounds])


def k_means_random_restart(dataset, k, restarts, seed=None):
    bounds = list(zip(np.min(dataset, axis=0), np.max(dataset, axis=0)))
    r = pseudo_random(seed=seed) if seed else pseudo_random()
    models = []
    for _ in range(restarts):
        random_centroids = tuple(generate_random_vector(bounds, r)
                                 for _ in range(k))
        new_centroids = k_means(dataset, random_centroids)
        clusters = cluster_points(new_centroids, dataset)
        if any(len(c) == 0 for c in clusters):
            continue
        models.append((goodness(clusters), new_centroids))
    return max(models, key=lambda x: x[0])[1]


#
# dataset = np.array([
#     [0.1, 0.1],
#     [0.2, 0.2],
#     [0.8, 0.8],
#     [0.9, 0.9]
# ])
# for c in k_means_random_restart(dataset, k=2, restarts=5):
#     print(c)
# c1 = k_means_random_restart(dataset, k=2, restarts=5)



import sklearn.datasets
import sklearn.utils

iris = sklearn.datasets.load_iris()
data, target = sklearn.utils.shuffle(iris.data, iris.target, random_state=0)
train_data, train_target = data[:-5, :], target[:-5]
test_data, test_target = data[-5:, :], target[-5:]

centroids = k_means_random_restart(train_data, k=3, restarts=10)
for c in centroids:
    print(c)

c1 = k_means_random_restart(dataset, k=2, restarts=5)


# We suggest you check which centroid each
# element in test_data is closest to, then see test_target.
# Note cluster 0 -> label 1
#      cluster 1 -> label 2
#      cluster 2 -> label 0