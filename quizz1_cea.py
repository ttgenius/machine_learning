import itertools
from copy import deepcopy

def input_space(domains):
    return list(itertools.product(*domains))

# #
# # domains1 = [
# # {9, 1}
# # ,{True, False}
# # ]
# #
# # domains = [
# # {0, 1, 2},
# # {True, False},
# # ]
# #
# # for element in sorted(input_space(domains)):
# #     print(element)
# #
# # print(len(input_space(domains)))
# #
# # #
# # import itertools
# #
# # b={"g","b","c"}
# # #d= input_space(domains)
# # x={'g',"l","b"}
# # l = [False, True]
# # ll = list(itertools.product(l, repeat=len(x)))
# # print(ll)
# # print(list(b))
#
#
#
#2
def all_possible_functions(X):
    l = [False, True]
    all_funcs = list(itertools.product(l, repeat=len(X)))
    out = []
    xs = list(X)
    for x in xs:
        for i, f in enumerate(all_funcs):
            def h(xx=x,func=f):
                return func[xs.index(xx)]
            out.append(h)
    return out[:len(out)//len(X)]
#
#
# # def all_possible_functions2(X):
# #     def h1(y):
# #         return True
# #     def h2(y):
# #         return False
# #     def h(x):
# #         y = x
# #         def h3(y):
# #             print("h3, x", x, "y", y)
# #             return True if x == y else False
# #         def h4(y):
# #             print("h4, x", x, "y", y)
# #             return True if x!= y else False
# #         return [h3, h4]
# #     hs = []
# #     for x in X:
# #         #print("hx",h(x))
# #         hs+=h(x)
# #     hs.append(h1)
# #     hs.append(h2)
# #     print("len(hs",len(hs))
# #     return hs
#
# X = {"green","purple"} # an input space with two elements
# F = all_possible_functions(X)
# # for h in F:
# #     print(h)
# #     for x in X:
# #         print("x",x,h(x))
# # Let's store the image of each function in F as a tuple
# images = set()
# for h in F:
#     images.add(tuple(h(x) for x in X))
#    # print(len(images))
#
# for image in sorted(images):
#     print(image)
#
#
# X2 = {1,2,3} # an input space with two elements
# F2 = all_possible_functions(X2)
# print("LENF2",len(F2))
#
#
#3
import itertools

def consistent(h, D):
    return all(h(x) == y for x, y in D)

def version_space(H, D):
    vs = set()
    for h in H:
        if consistent(h, D):
            vs.add(h)
    return vs
#
# X = {"green", "purple"} # an input space with two elements
# D = {("green", False)} # the training data is a subset of X * {True, False}
# F = all_possible_functions(X)
# H = F # H must be a subset of (or equal to) F
#
# VS = version_space(H, D)
#
# print(len(VS))
#
#
# for h in VS:
#     for x, y in D:
#         if h(x) != y:
#             print("You have a hypothesis in VS that does not agree with the D!")
#             break
#     else:
#         continue
#     break
# else:
#     print("OK")
#
#
#
# D = {
# ((False, True), False),
# ((True, True), True),
# }
# def h1(x): return True
# def h2(x): return False
# def h3(x): return x[0] and x[1]
# def h4(x): return x[0] or x[1]
# def h5(x): return x[0]
# def h6(x): return x[1]
# H = {h1, h2, h3, h4, h5, h6}
# VS = version_space(H, D)
# print(sorted(h.__name__ for h in VS))
#
#
# 4
def less_general_or_equal(ha, hb, X):
    a_set = set()
    b_set = set()
    for x in X:
        if ha(x):
            a_set.add(x)
        if hb(x):
            b_set.add(x)
    return a_set.issubset(b_set)
#
#
#
# X = list(range(1000))
#
# def h2(x): return x % 2 == 0
# def h3(x): return x % 3 == 0
# def h6(x): return x % 6 == 0
# H = [h2, h3, h6]
#
# for ha in H:
#     for hb in H:
#         print(ha.__name__, "<=", hb.__name__, "?", less_general_or_equal(ha, hb, X))
#
# 5
def decode(code):
    x1, y1, x2, y2 = code
    xmin = min(x1,x2)
    xmax = max(x1,x2)
    ymin = min(y1,y2)
    ymax = max(y1,y2)
    def h(p):
        px, py = p
        return (xmin <= px <= xmax) and (ymin <= py <= ymax)
    return h
#
# h = decode((-1, -1, 1, 1))
#
# for x in itertools.product(range(-2, 3), repeat=2):
#     print(x, h(x))
# import itertools
# h1 = decode((1, 4, 7, 9))
# h2 = decode((7, 9, 1, 4))
# h3 = decode((1, 9, 7, 4))
# h4 = decode((7, 4, 1, 9))
# for x in itertools.product(range(-2, 11), repeat=2):
#     if len({h(x) for h in [h1, h2, h3, h4]}) != 1:
#         print("Inconsistent prediction for", x)
#         break
#     else:
#         print("OK")

# If you wish, you can use the following template.
#
# Representation-dependent functions are defined outside of the main CEA
# function. This allows CEA to be representation-independent. In other words
# by defining the following functions appropriately, you can make CEA work with
# any representation.
def initial_S(domains):
    """Takes a list of domains and returns a set where each element is a
    code for the initial members of S."""

    return set([('0',)*len(domains)])


def initial_G(domains):
    """Takes a list of domains and returns a set where each element is a
    code for the initial members of G."""

    return set([("?",)*len(domains)])

#
def decode(code):
    """Takes a code and returns the corresponding hypothesis."""
    def h(x):
        hs = []
        for i in range(len(code)):
            if code[i] != '0' and (code[i] == '?' or code[i] == x[i]):
                hs.append(True)
            else:
                hs.append(False)
        return all(hs)
    return h


def match(code, x):
    """Takes a code and returns True if the corresponding hypothesis returns
    True (positive) for the given input."""
    return decode(code)(x)


def lge(code_a, code_b):
    """Takes two codes and returns True if code_a is less general or equal
    to code_b."""
    less_general = []
    for x, y in zip(code_a, code_b):
        if y == "?" :
            less_general.append(True)
        elif (x == y and y != "0"):
            less_general.append(True)
        elif (x == "0" and y != "0"):
            less_general.append(True)
        else:
           less_general.append(False)
    return all(less_general)


def generalize_S(x, G, S):
    """Generalize S"""
    S_prev = list(S)
    for s in S_prev:
        if not match(s, x):
            S.remove(s)
            Splus = minimal_generalisations(s, x)
            for h in Splus:
                if all([lge(h,g) for g in G]):
                    S.add(h)
            S.difference_update([h for h in S if any([lge(h, s1) for s1 in S if h != s1])])
    return S

def specialize_G(x, domains, G, S):
    """Specialize G"""
    G_prev = list(G)
    for g in G_prev:
        if match(g, x):
            G.remove(g)
            Gplus = minimal_specialisations(g, domains, x)

            for h in Gplus:
                if any([lge(s, h) == True for s in S]):
                    G.add(h)
            G.difference_update([h for h in G if any([lge(h, g1) for g1 in G if h != g1])])
    return G

def minimal_generalisations(code, x):
    """Takes a code (corresponding to a hypothesis) and returns the set of all
    codes that are the minimal generalisations of the given code with respect
    to the given input x."""
    h_new = []
    for i in range(len(code)):
        if match([x[i]],[code[i]]):
            h_new.append(code[i])
        else:
            if code[i] != '0':
                h_new.append('?')
            else:
                h_new.append(x[i])
    return [tuple(h_new)]


def minimal_specialisations(cc, domains, x):
    """Takes a code (corresponding to a hypothesis) and returns the set of all
    codes that are the minimal specialisions of the given code with respect
    to the given input x."""
    results = []
    for i in range(len(cc)):
        if cc[i] == "?":
            for val in domains[i]:
                if val != x[i]:
                    cc_new = list(deepcopy(cc))
                    cc_new[i] = val
                    results.append(tuple(cc_new))
        elif cc[i] != "0":
            cc_new = list(deepcopy(cc))
            cc_new[i] = "0"
            results.append(tuple(cc_new))
    return results


def cea_trace(domains, D):
    S_trace, G_trace = [], []
    S = initial_S(domains)
    G = initial_G(domains)
    # Append S and G (or thier copy) to corresponding trace list
    S_trace.append(deepcopy(S))
    G_trace.append(deepcopy(G))

    for x, y in D:
        if y:  # if positive
            G = {g for g in G if match(g, x)}
            S = generalize_S(x, G, S)

        else:  # if negative
            S = {s for s in S if not match(s, x)}
            G = specialize_G(x, domains, G, S)

        S_trace.append(deepcopy(S))
        G_trace.append(deepcopy(G))
    return S_trace, G_trace


def all_agree(S,G,x):
    ans = set()
    for s in S:
        ans.add(match(s,x))
    for g in G:
        ans.add(match(g,x))
    return len(ans) == 1








# domains = [
#     {'red', 'blue'}
# ]
#
# training_examples = [
#     (('red',), True)
# ]
#
# S_trace, G_trace = cea_trace(domains, training_examples)
# print(len(S_trace), len(G_trace))
# print(all(type(x) is set for x in S_trace + G_trace))
# S, G = S_trace[-1], G_trace[-1]
# print(len(S), len(G))
#
#
# domains = [
#     {'T', 'F'}
# ]
#
# training_examples = []  # no training examples
#
# S_trace, G_trace = cea_trace(domains, training_examples)
# print(len(S_trace), len(G_trace))
# S, G = S_trace[-1], G_trace[-1]
# print(len(S), len(G))
#
# domains = [
#     ('T', 'F'),
#     ('T', 'F'),
# ]
#
# training_examples = [
#     (('F', 'F'), True),
#     (('T', 'T'), False),
# ]
#
# S_trace, G_trace = cea_trace(domains, training_examples)
# print(len(S_trace), len(G_trace))
# S, G = S_trace[-1], G_trace[-1]
# print(len(S), len(G))
#
#
# # A case where the target function is not in H
#
# domains = [
#     {'red', 'green', 'blue'}
# ]
#
# training_examples = [
#     (('red',), True),
#     (('green',), True),
#     (('blue',), False),
# ]
#
# S_trace, G_trace = cea_trace(domains, training_examples)
# S, G = S_trace[-1], G_trace[-1]
# print(len(S)==len(G)==0)
#
#
# domains = [
#     {'sunny', 'cloudy', 'rainy'},
#     {'warm', 'cold'},
#     {'normal', 'high'},
#     {'strong', 'weak'},
#     {'warm', 'cool'},
#     {'same', 'change'},
# ]
#
# training_examples = [
#     (('sunny', 'warm', 'normal', 'strong', 'warm', 'same'), True),
#     (('sunny', 'warm', 'high', 'strong', 'warm', 'same'), True),
#     (('rainy', 'cold', 'high', 'strong', 'warm', 'change'), False),
#     (('sunny', 'warm', 'high', 'strong', 'cool', 'change'), True),
# ]
#
# S_trace, G_trace = cea_trace(domains, training_examples)
# print(len(S_trace) == len(G_trace) == 5)
# if len(S_trace) == len(G_trace) == 5:
#     print(len(S_trace[3]),len(G_trace[3]))
# else:
#     print("Incorrect number of snapshots in S_trace of G_trace.")

# domains = [
#     {'red', 'blue'},
# ]
#
# training_examples = [
#     (('red',), True),
# ]
#
# S_trace, G_trace = cea_trace(domains, training_examples)
# S, G = S_trace[-1], G_trace[-1]
# print(all_agree(S, G, ('red',)))
# print(all_agree(S, G, ('blue',)))
#
#
# domains = [
# {'T', 'F'},
# ]
# training_examples = []
# S_trace, G_trace = cea_trace(domains, training_examples)
# S, G = S_trace[-1], G_trace[-1]
# print(all_agree(S, G, ('T',)))
# print(all_agree(S, G, ('F',)))
#
#
#
#
# domains = [
# {'T', 'F'},
# {'T', 'F'},
# ]
# training_examples = [
# (('F', 'F'), True),
# (('T', 'T'), False),
# ]
# S_trace, G_trace = cea_trace(domains, training_examples)
# S, G = S_trace[-1], G_trace[-1]
# print(all_agree(S, G, ('F', 'F')))
# print(all_agree(S, G, ('T', 'T')))
# print(all_agree(S, G, ('F', 'T')))
# print(all_agree(S, G, ('T', 'F')))

# domains = [
#     {'red', 'blue'}
# ]
#
# training_examples = [
#     (('red',), True)
# ]
#
# S_trace, G_trace = cea_trace(domains, training_examples)
# print(len(S_trace), len(G_trace))
# print(all(type(x) is set for x in S_trace + G_trace))
# S, G = S_trace[-1], G_trace[-1]
# print(len(S), len(G))
#
# domains = [
#     {'T', 'F'}
# ]
#
# training_examples = []  # no training examples
#
# S_trace, G_trace = cea_trace(domains, training_examples)
# print(len(S_trace), len(G_trace))
# S, G = S_trace[-1], G_trace[-1]
# print(len(S), len(G))
#
# domains = [
#     ('T', 'F'),
#     ('T', 'F'),
# ]
#
# training_examples = [
#     (('F', 'F'), True),
#     (('T', 'T'), False),
# ]
#
# S_trace, G_trace = cea_trace(domains, training_examples)
# print(len(S_trace), len(G_trace))
# S, G = S_trace[-1], G_trace[-1]
# print(len(S), len(G))
#

# domains = [
#     {'sunny', 'cloudy', 'rainy'},
#     {'warm', 'cold'},
#     {'normal', 'high'},
#     {'strong', 'weak'},
#     {'warm', 'cool'},
#     {'same', 'change'},
# ]
#
# training_examples = [
#     (('sunny', 'warm', 'normal', 'strong', 'warm', 'same'), True),
#     (('sunny', 'warm', 'high', 'strong', 'warm', 'same'), True),
#     (('rainy', 'cold', 'high', 'strong', 'warm', 'change'), False),
#     (('sunny', 'warm', 'high', 'strong', 'cool', 'change'), True),
# ]
#
# S_trace, G_trace = cea_trace(domains, training_examples)
# print(len(S_trace) == len(G_trace) == 5)
#print(len(S),len(G))
# if len(S_trace) == len(G_trace) == 5:
#    print()


