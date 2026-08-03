from tarfile import data_filter
import numpy as np
from scipy.optimize import curve_fit
import json
from collections import defaultdict
import pathlib

def linear(logC, b, k):
    return b + k * logC

def predict_N(C, b, k):
    return 10 ** (b + k * np.log10(np.float32(C)))

def predict_D(C, b, k):
    N = predict_N(C, b, k)
    return C / (6.0 * N)

def main():
    path = pathlib.Path(__file__).resolve().parent.parent / 'data/isoflops_curves.json'
    data = json.load(open(path, 'r'))
    groups = defaultdict(list)
    for item in data:
        groups[item['compute_budget']].append(item)

    C_arr, N_arr = [], []
    for c in sorted(groups):
        best = min(groups[c], key=lambda x: x['final_loss'])
        C_arr.append(c)
        N_arr.append(best['parameters'])
    C_arr = np.array(C_arr)
    N_arr = np.array(N_arr)
    logC = np.log10(C_arr)
    logN = np.log10(N_arr)
    b, k = curve_fit(linear, logC, logN, p0 = [0.0, 0.5])[0]
    size1 = predict_N(10**23, b, k)
    size2 = predict_N(10**24, b, k)
    print(f"size1: {(size1/1024**3):.2f} GB, size2: {(size2/ 1024**3):.2f} GB")
    data_size1 = predict_D(10**23, b, k)
    data_size2 = predict_D(10**24, b, k)
    print(f"data_size1: {(data_size1/1024**3):.2f} GB, data_size2: {(data_size2/ 1024**3):.2f} GB")

if __name__ == "__main__":
    main()



