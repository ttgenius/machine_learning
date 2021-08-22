import math


class DTNode:
    def __init__(self, decision):
        self.decision = decision
        self.children = None

    def predict(self, feature_vector):
        if callable(self.decision):
            return self.children[self.decision(feature_vector)].predict(feature_vector)
        return self.decision

    def leaves(self):
        if len(self.children) == 0:
            return 1
        return sum([child.leaves() for child in self.children])

# The following (leaf) node will always predict True
# node = DTNode(True)
#
# # Prediction for the input (True, False):
# print(node.predict((True, False)))
#
# # Sine it's a leaf node, the input can be anything. It's simply ignored.
# print(node.predict(None))
#
# t = DTNode(True)
# f = DTNode(False)
# n = DTNode(lambda v: 0 if not v else 1)
# n.children = [t, f]
#
# print(n.predict(False))
# print(n.predict(True))
#
# tt = DTNode(False)
# tf = DTNode(True)
# ft = DTNode(True)
# ff = DTNode(False)
# t = DTNode(lambda v: 0 if v[1] else 1)
# f = DTNode(lambda v: 0 if v[1] else 1)
# t.children = [tt, tf]
# f.children = [ft, ff]
# n = DTNode(lambda v: 0 if v[0] else 1)
# n.children = [t, f]
#
# print(n.predict((True, True)))
# print(n.predict((True, False)))
# print(n.predict((False, True)))
# print(n.predict((False, False)))


def partition_by_feature_value(index, data):
    features = []
    d = {}
    for (v, c) in data:
        if d.get(v[index]) is None:
            d[v[index]] = [(v, c)]
            features.append(v[index])
        else:
            d[v[index]].append((v, c))
    separator = lambda f: features.index(f[index])
    return separator, list(d.values())

# dataset = [
#   ((True, True), False),
#   ((True, False), True),
#   ((False, True), True),
#   ((False, False), False),
# ]
# f, p = partition_by_feature_value(0, dataset)
# print("f",f)
# print("p",p)
# print(sorted(sorted(partition) for partition in p))
#
# partition_index = f((True, True))
# # Everything in the "True" partition for feature 0 is true
# print(all(x[0]==True for x,c in p[partition_index]))
# partition_index = f((False, True))
# # Everything in the "False" partition for feature 0 is false
# print(all(x[0]==False for x,c in p[partition_index]))
#
# # [[((False, False), False), ((False, True), True)],
# #  [((True, False), True), ((True, True), False)]]
# # True
# #
# from pprint import pprint
# dataset = [
#   (("a", "x", 2), False),
#   (("b", "x", 2), False),
#   (("a", "y", 5), True),
# ]
# f, p = partition_by_feature_value(1, dataset)
# pprint(sorted(sorted(partition) for partition in p))
# partition_index = f(("a", "y", 5))
# # everything in the "y" partition for feature 1 has a y
# print(all(x[1]=="y" for x, c in p[partition_index]))
#
# # [[(('a', 'x', 2), False), (('b', 'x', 2), False)], [(('a', 'y', 5), True)]]
# # True
# import math
#
def get_proportion(classification, data):
    total = 0
    for _, c in data:
        if c == classification:
            total += 1
    return total / len(data)


def get_classes(data):
    return set([d[-1] for d in data])


def misclassification(data):
    pks = []
    for k in get_classes(data):
        pks.append(get_proportion(k, data))
    return 1 - max(pks)


def gini(data):
    H = 0
    for k in get_classes(data):
        pk = get_proportion(k, data)
        H += pk * (1 - pk)
    return H


def entropy(data):
    H = 0
    for k in get_classes(data):
        pk = get_proportion(k, data)
        if pk != 0:
            H += pk * math.log(pk)
    return -H



# data = [
#     ((False, False), False),
#     ((False, True), True),
#     ((True, False), True),
#     ((True, True), False)
# ]
# print("{:.4f}".format(misclassification(data)))
# print("{:.4f}".format(gini(data)))
# print("{:.4f}".format(entropy(data)))


def get_impurity(criterion, k, data):
    separator, partition = partition_by_feature_value(k, data)
    if len(partition) == 1:
        return float('-inf')
    return sum([(len(p)/len(data)) * criterion(p) for p in partition])


def train_tree(data, criterion):
    classes = list(set([d[-1] for d in data]))
    if len(classes) == 1:
        return DTNode(data[0][1])
    elif len(data[0]) == 0:
        proportions = [get_proportion(k, data) for k in classes]
        most_common_label = classes[proportions.index(max(proportions))]
        return DTNode(most_common_label)
    else:
        features = data[0][0]
        impurities = [get_impurity(criterion, k, data) for k in range(len(features))]
        feature_index = impurities.index(max(impurities))
        separator, partition = partition_by_feature_value(feature_index, data)
        #print("partition", partition)
        node = DTNode(separator)
        node.children = [train_tree(p, criterion) for p in partition]
        return node


# dataset = [
#   ((True, True), False),
#   ((True, False), True),
#   ((False, True), True),
#   ((False, False), False)
# ]
# t = train_tree(dataset, misclassification)
# print(t.predict((True, False)))
# print(t.predict((False, False)))
#
# n = DTNode(True)
# print(n.leaves())
# t = DTNode(True)
# f = DTNode(False)
# n = DTNode(lambda v: 0 if not v else 1)
# n.children = [t, f]
# print(n.leaves())

# 6
bal_dataset = []
with open('bal.txt.txt', 'r') as f:
    for line in f.readlines():
        out, *features = line.strip().split(",")
        bal_dataset.append((tuple(features), out))

car_dataset = []
with open('car.txt.txt', 'r') as f:
    for line in f.readlines():
        *features, out = line.strip().split(",")
        car_dataset.append((tuple(features), out))
b = train_tree(bal_dataset, misclassification)
print((len(bal_dataset)/b.leaves()))
c = train_tree(car_dataset, misclassification)
print((len(car_dataset)/c.leaves()))


from sklearn import datasets
from sklearn.model_selection import train_test_split