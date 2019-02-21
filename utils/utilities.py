import pickle
import os
from math import factorial as f


def store_and_load(func):
    func_name = func.__name__
    file_name = "../tmp/" + func_name + ".pickle"
    if not os.path.isdir("../tmp"):
        os.mkdir("../tmp")

    def func_with_stored_results(*args, **kwargs):
        if os.path.isfile(file_name):
            with open(file_name, "rb") as f:
                print("Loading results from file: " + func_name)
                func_results = pickle.load(f)
        else:
            with open(file_name, "wb") as f:
                func_results = func(*args, **kwargs)
                print("Storing results: " + func_name)
                pickle.dump(func_results, f)
        return func_results

    return func_with_stored_results


def count_combinations(items, max_simul_draws):
    count = 0
    if max_simul_draws > len(items):
        max_simul_draws = len(items)
    from itertools import combinations
    for i in range(2, max_simul_draws + 1):
        for c in combinations(items, i):
            if len(set(c)) == len(c):
                count += 1
    return count


def n_over_k(n, k):
    return f(n) / (f(k) * f(n - k))
