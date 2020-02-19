import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import scipy
from tqdm import tqdm
import multiprocessing as mp
import ctypes as c
import time
import h5py
import argparse
import os
import yaml
from scipy.stats import multivariate_normal
from sklearn import neighbors


def create_path(path):
    if os.path.isdir(path) is False:
        os.makedirs(path)


def get_delta(args):
    file = open("params.yaml", 'r')
    params = yaml.load(file, Loader=yaml.FullLoader)
    deltas = params[args.dtype][args.mtype]["dim{0}".format(args.dim)]
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

    if dim ==  1:
        data = (temp_a) * np.random.uniform(size=(M, dim))
    else:
        data = (temp_a / 2) * np.random.uniform(size=(M, dim))

    y = np.mean(np.linalg.norm(data, axis=1))

    for j, delta in enumerate(delta_values):
        errors[run, test_points_idx, N_idx, j] = np.minimum(1.0, y / delta)

    return 1


def main(args):
    dim = args.dim
    args.save_path = "results/"
    
    data = h5py.File(args.features_path, 'r')
    X_train = np.array(data["0.5/train/features"])
    Y_train = np.array(data["0.5/train/targets"]).reshape(-1)
    X_test = np.array(data["0.5/val/features"])    
    Y_test = np.array(data["0.5/val/targets"]).reshape(-1)
    X = np.concatenate((X_train, X_test), axis=0)

    log_N_values = np.linspace(-1, 11, 100)
    th_sampled_N = np.array([0.03125, 0.0625, 0.125, 0.25, 0.5, 0.75, 1.0])

    exp_results = scipy.io.loadmat(args.exp_results_path)
    exp_results = np.mean(exp_results['y'], axis=0)
    exp_results = exp_results.reshape(-1, 1)
    exp_results = exp_results[1:, :]

    delta_values = get_delta(args)

    M = 10**3

    k = M
    l = len(log_N_values)
    m = len(delta_values)

    errors = mp.Array(c.c_double, args.runs * k * l * m)

    pool = mp.Pool(25, initializer=init_worker,
                   initargs=(errors, (args.runs, k, l, m)))

    global pbar
    pbar = tqdm(total=args.runs * k * l)

    tqdm.write(".... Dimensionality reduction through PCA.......")

    # PCA to reduce data dimensions
    pca = PCA(n_components=dim, svd_solver='full')
    pca.fit(X)
    tqdm.write("Variance Ratio:{0}\n#components:{1}".format(
        pca.explained_variance_ratio_, pca.explained_variance_ratio_.shape))
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    for run in range(args.runs):
        #######################################################################
        #                             Gaussian Mixture Model
        #######################################################################
        train_model = GaussianMixture(n_components=1, covariance_type='diag',
                                      max_iter=100, random_state=run, verbose=0)
        train_model.fit(X_train)

        test_model = GaussianMixture(n_components=1, covariance_type='diag',
                                     max_iter=100, random_state=run, verbose=0)
        test_model.fit(X_test)

        test_data, _ = test_model.sample(M)

        for N_idx, log_N in enumerate(log_N_values):

            a = -(log_N + train_model.score_samples(test_data))
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

    # Matching the curve for various delta values and dimension. 
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
        'train_mu': train_model.means_,
        'train_covar': train_model.covariances_,
        'test_mu': test_model.means_,
        'test_covar': test_model.covariances_
    }

    scipy.io.savemat(save_path, save_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Theoretical Results')
    parser.add_argument('--runs', type=int, default=10, help='Runs')
    parser.add_argument('--exp_results_path', required=True, type=str)
    parser.add_argument('--features_path', required=True, type=str)
    parser.add_argument('--mtype', type=str, required=True)
    parser.add_argument('--dtype', type=str, required=True)

    args = parser.parse_args()

    for dim in [1,2,3,4,5]:
        args.dim = dim
        main(args)
