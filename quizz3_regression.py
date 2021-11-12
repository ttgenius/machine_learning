import numpy as np
import math


def dot(x, y):
    return sum([xi * yi for xi, yi in zip(x, y)])


def linear_regression_1d(data):
    n = len(data)
    x, y = list(zip(*(data)))
    m = (n * dot(x, y) - sum(x) * sum(y)) / (n * dot(x, x) - sum(x) ** 2)
    c = (sum(y) - m * sum(x)) / n
    return m, c

# data = [(1, 4), (2, 7), (3, 10)]
# m, c = linear_regression_1d(data)
# print(m, c)
# print(4 * m + c)
# data = [(2, 2), (3, 8), (12, 62), (7, 32)]
# m, c = linear_regression_1d(data)
# print(m, c)
# print(100 * m + c)


def linear_regression(x, y):
    intercept = np.ones(x.shape[0]).reshape(-1, 1)
    x = np.concatenate((intercept, x), axis=1)
    theta = np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T, x)), x.T), y)
    return theta

# xs = np.arange(5).reshape((-1, 1))
# print("xs",xs)
# ys = np.arange(1, 11, 2)
# print(linear_regression(xs, ys))
#
# import numpy as np
#
# xs = np.array([[1, 2, 3, 4],
#                [6, 2, 9, 1]]).T
# ys = np.array([7, 5, 14, 8]).T
# print(linear_regression(xs, ys))


def linear_regression(xs, ys, basis_functions=None):
    intercept = np.ones(xs.shape[0]).reshape(-1, 1)
    if basis_functions is None:
        xs = np.concatenate((intercept, xs), axis=1)
    else:
        basis = intercept
        for f in basis_functions:
            basis_col = np.array([f(x) for x in xs]).reshape(-1,1)
            basis = np.concatenate((basis, basis_col), axis=1)
        xs = basis
    print("(np.matmul(xs.T, xs)",(np.matmul(xs.T, xs)))
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(xs.T, xs)), xs.T), ys)

# print("q3")
xs = np.arange(5).reshape((-1, 1))
ys = np.array([3, 6, 11, 18, 27])
# Can you see y as a function of x? [hint: it's quadratic.]
functions = [lambda x: x[0], lambda x: x[0] ** 2]
#functions = [lambda x: x[0], lambda x: 1]
print(linear_regression(xs, ys, functions))


# xs = np.arange(5).reshape((-1, 1))
# ys = np.array([-3.95, -3.9, 2.2, 20.4, 56.8])
# functions = [lambda x: x[0], lambda x: x[0]**2, lambda x: x[0]**3, lambda x: 2**x[0]]
# coefficients = linear_regression(xs, ys, functions).ravel()
# expected = np.array([-4.000, -1.000, 0.000, 1.000, 0.050])
# if np.allclose(coefficients, expected):
#     print("OK")
# else:
#     print("The foll")
#     print("{:.3f}".format(coefficients))



def linear_regression(xs, ys, basis_functions=None, penalty=0.):
    intercept = np.ones(xs.shape[0]).reshape(-1, 1)
    if basis_functions is None:
        xs = np.concatenate((intercept, xs), axis=1)
    else:
        basis = intercept
        for f in basis_functions:
            basis_col = np.array([f(x) for x in xs]).reshape(-1, 1)
            basis = np.concatenate((basis, basis_col), axis=1)
        xs = basis
    penalty_mat = np.identity(xs.shape[1]) * penalty
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(xs.T, xs) + penalty_mat), xs.T), ys)

# print("q4")
# xs = np.arange(5).reshape((-1, 1))
# ys = np.arange(1, 11, 2)
#
# print(linear_regression(xs, ys), end="\n\n")
#
# with np.printoptions(precision=5, suppress=True):
#     print(linear_regression(xs, ys, penalty=0.1))
#
# # we set the seed to some number so we can replicate the computation
# np.random.seed(0)
#
# xs = np.arange(-1, 1, 0.1).reshape(-1, 1)
# m, n = xs.shape
# # Some true function plus some noise:
# ys = (xs**2 - 3*xs + 2 + np.random.normal(0, 0.5, (m, 1))).ravel()
#
# functions = [lambda x: x[0], lambda x: x[0]**2, lambda x: x[0]**3, lambda x: x[0]**4,
#       lambda x: x[0]**5, lambda x: x[0]**6, lambda x: x[0]**7, lambda x: x[0]**8]
#
# for penalty in [0, 0.01, 0.1, 1, 10]:
#     with np.printoptions(precision=5, suppress=True):
#         print(linear_regression(xs, ys, basis_functions=functions, penalty=penalty)
#               .reshape((-1, 1)), end="\n\n")
#
# xs = np.array([[1, 2, 3, 4],
# [6, 2, 9, 1]]).T
# ys = np.array([7, 5, 14, 8]).T
# print(linear_regression(xs, ys))


def logistic_regression(xs, ys, alpha, num_iterations):
    intercept = np.ones(xs.shape[0]).reshape(-1, 1)
    xs = np.concatenate((intercept, xs), axis=1)

    sigmoid = lambda x: 1 / (1 + math.exp(-x))

    num_examples, features = xs.shape

    theta = np.zeros(features)
    # θj := θj + α(y (i) − hθ(x^(i)) xj^(i)
    for _ in range(num_iterations):
        for i in range(num_examples):
            xi = xs[i, :]
            yi = ys[i]
            theta = theta + alpha * (yi - sigmoid(np.matmul(theta.T, xi))) * xi

    return lambda x: sigmoid(np.matmul(theta.T, np.insert(x, 0, [1])))


# print("q5")
# xs = np.array([1, 2, 3, 101, 102, 103]).reshape((-1, 1))
# ys = np.array([0, 0, 0, 1, 1, 1])
# model = logistic_regression(xs, ys, 0.05, 10000)
# test_inputs = np.array([1.5, 4, 10, 20, 30, 40, 50, 60, 70, 80, 90, 101.8, 97]).reshape((-1, 1))
#
# for test_input in test_inputs:
#     print("{:.2f}".format(np.array(model(test_input)).item()))
#
# xs = np.array(
#     [0.50,0.75,1.00,1.25,1.50,
#      1.75,1.75,2.00,2.25,2.50,
#      2.75,3.00,3.25,3.50,4.00,
#      4.25,4.50,4.75,5.00,5.50]).reshape((-1, 1))
#
# ys = np.array([0,0,0,0,0,
#                0,1,0,1,0,
#                1,0,1,0,1,
#                1,1,1,1,1])
#
# model = logistic_regression(xs, ys, 0.02, 5000)
# sse = 0
# output = []
# expected = [0.02, 0.03, 0.07, 0.14, 0.25, 0.42, 0.60, 0.77, 0.87, 0.94, 0.97, 0.99]
# for i, x in enumerate(np.arange(0, 6, 0.5).reshape(-1,1)):
#     output.append(np.array(model(x)).item())
#     sse += (expected[i] - output[-1]) ** 2
#
# tolerance = 1e-3
# if sse / len(expected) < tolerance:
#     print("OK")
# else:
#     print("The error is too high.")
#     print("The expected output is: ", expected)
#     print("The output of the trained model is:", output)
#
#
# data = np.genfromtxt("data_banknote_authentication.txt", delimiter=',')
# np.random.seed(0)
# np.random.shuffle(data)
# data = data[:500, :]
#
# xs_train, xs_test = data[:-50, :-1], data[-50:, :-1]
# ys_train, ys_test = data[:-50, -1], data[-50:, -1]
# model = logistic_regression(xs_train, ys_train, 0.02, 1000)
# print(sum(abs(y - model(x)) for (x, y) in zip(xs_test, ys_test))/50 < 0.05)
