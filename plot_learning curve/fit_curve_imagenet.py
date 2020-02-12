import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.mixture import GaussianMixture
import scipy
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA as IPCA
from sklearn import preprocessing
import multiprocessing as mp
import ctypes as c
import time
import h5py
import argparse
import os
from scipy.stats import multivariate_normal
import yaml


def create_path(path):
    if os.path.isdir(path) is False:
        os.makedirs(path)


def get_delta(args):
    file = open("params.yaml", 'r')
    params = yaml.load(file, Loader=yaml.FullLoader)
    deltas = params[args.dtype]["dim{0}".format(args.dim)]
    deltas = np.linspace(deltas[0], deltas[1], 100)
    return deltas


# A global dictionary storing the variables passed from the initializer.
var_dict = {}


def update_pbar(*a):
    global pbar
    pbar.update()


def init_worker(X, X_shape):
    # Using a dictionary is not strictly necessary. You can also
    # use global variables.
    var_dict['X'] = X
    var_dict['X_shape'] = X_shape


def exp_run(delta_values, temp_a, run, test_points_idx, N_idx, M, dim):
    errors = np.frombuffer(var_dict['X'].get_obj()).reshape(var_dict['X_shape'])

    data = (temp_a / 2) * np.random.uniform(size=(M, dim))
    y = np.mean(np.linalg.norm(data, axis=1))

    for j, delta in enumerate(delta_values):
        errors[run, test_points_idx, N_idx, j] = np.minimum(1.0, y / delta)

    return 1


def main(args):
    dim = args.dim

    X = h5py.File('../data/imagenet/imagenet_train_feat.hdf5')
    X = X['features']

    log_N_values = np.linspace(-1, 14, 100)

    delta_values = get_delta(args)

    th_sampled_N = np.array([50,75,100,150,200,250,400,500,750,800,1000,1281.167])/1281.167

    exp_results_path = "../results/imagenet/resnet_imagenet_errors.mat"
    exp_results = scipy.io.loadmat(exp_results_path)
    exp_results = np.mean(exp_results['y'], axis=0)/100
    exp_results = exp_results.reshape(-1, 1)
    exp_results = exp_results[1:, :]

    save_path = "../results/imagenet/theoretical/features/imagenet_dim_{0}.mat".format(dim)

    M = 10**3

    k = M
    l = len(log_N_values)
    m = len(delta_values)

    errors = mp.Array(c.c_double, args.runs * k * l * m)

    pool = mp.Pool(25, initializer=init_worker,
                   initargs=(errors, (args.runs, k, l, m)))

    tqdm.write(".... Dimensionality reduction through PCA.......")
    pca = IPCA(n_components=dim)

    for i in tqdm(range(0, X.shape[0] // 10000)):
        pca.partial_fit(X[i*10000 : (i+1)*10000,:])

    X = pca.transform(X[0:50000, :])
    tqdm.write("Variance Ratio:{0}\n#components:{1}".format(
        pca.explained_variance_ratio_, pca.explained_variance_ratio_.shape))

    global pbar
    pbar = tqdm(total=args.runs * k * l)

    for run in range(args.runs):
        model = multivariate_normal(mean=np.mean(X, axis=0), cov=np.cov(X.T))
        test_data = model.rvs(size=(M)).reshape(-1, dim)

        for N_idx, log_N in enumerate(log_N_values):
            a = -(log_N + model.logpdf(test_data))
            a /= test_data.shape[1]
            a = np.exp(a)
            a = a.reshape(-1, 1)

            for test_points_idx in range(a.shape[0]):
                temp_a = a[test_points_idx, 0]
                pool.apply_async(exp_run,
                                 args=(delta_values, temp_a, run,
                                       test_points_idx, N_idx, 10**3, dim),
                                 callback=update_pbar)

    pool.close()
    pool.join()

    errors = np.frombuffer(errors.get_obj()).reshape(args.runs, k, l, m)
    errors = np.mean(errors, axis=1)

    x_values = np.exp(log_N_values - np.max(log_N_values))
    N_idx = []
    D = np.diag(1 / (exp_results.reshape(-1))**2)

    for th_N in th_sampled_N:
        idx = np.argmin(np.abs(x_values - th_N))
        N_idx.append(idx)

    mean_errors_over_runs = np.mean(errors, axis=0)
    exp_results = np.tile(exp_results, (1, mean_errors_over_runs.shape[1]))
    temp = np.abs(mean_errors_over_runs[N_idx, :] - exp_results)
    norm_value = np.sqrt(np.diag(np.matmul(np.matmul(np.transpose(temp), D), temp)))
    idx = np.argmin(norm_value)

    tqdm.write("Dim: {0}\nnorm: {1} ,idx: {2}, delta:{3}".
               format(dim, norm_value[idx], idx, delta_values[idx]))

    save_data = {
        'x': np.exp(log_N_values - np.max(log_N_values)),
        'err': errors,
        'delta': delta_values,
        'norm_idx': idx,
        'norm_value': np.min(norm_value),
        'mu': model.mean,
        'covar': model.cov,
    }

    scipy.io.savemat(save_path, save_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Theoretical Results')
    parser.add_argument('--runs', type=int, default=1, help='Runs')
    parser.add_argument('--dtype', type=str, default='imagenet')
    parser.add_argument('--model_width', type=int)
    parser.add_argument('--model_depth', type=int)
    args = parser.parse_args()

    for dim in [2,3]:
        args.dim = dim
        main(args)
