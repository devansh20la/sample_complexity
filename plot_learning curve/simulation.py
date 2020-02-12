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
from scipy import spatial


def create_path(path):
    if os.path.isdir(path) is False:
        os.makedirs(path)


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

    delta_values = np.linspace(1, 10, 10)
    N_values = np.floor(np.linspace(10**3, 10**5, 10))
    log_N_values = np.log(N_values)
    M = 10**3

    k = M
    l = len(log_N_values)
    m = len(delta_values)

    errors = mp.Array(c.c_double, args.runs * k * l * m)
    sim_loss = np.zeros((args.runs, l, m))
    model = multivariate_normal(mean=np.zeros((dim)), cov=10*np.eye((dim)))
    pool = mp.Pool(25, initializer=init_worker,
                   initargs=(errors, (args.runs, k, l, m)))
    global pbar
    pbar = tqdm(total=args.runs * k * l)

    for run in range(args.runs):
        test_data = model.rvs(size=(M)).reshape(-1, dim)

        for N_idx, log_N in enumerate(log_N_values):
            train_data = model.rvs(size=(int(np.exp(log_N)))).reshape(-1, dim)

            tree = spatial.KDTree(train_data)
            d, tr_idx = tree.query(test_data)

            for del_idx, delta in enumerate(delta_values):
                new_d = d / delta
                new_d[new_d >= 1] = 1
                sim_loss[run, N_idx, del_idx] = np.mean(new_d)

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
    errors = np.mean(np.mean(errors, axis=1), axis=0)
    sim_err = np.mean(sim_loss, axis=0)
    print(sim_err)

    save_data = {
        'x': N_values,
        'th_err': errors,
        'sim_err': sim_err,
        'delta': delta_values,
    }

    scipy.io.savemat("test_dim{0}".format(args.dim), save_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Theoretical Results')
    parser.add_argument('--runs', type=int, default=10, help='Runs')
    args = parser.parse_args()

    for dim in [1, 2, 3, 4, 5]:
        args.dim = dim
        main(args)
