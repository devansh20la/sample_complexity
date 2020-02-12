import numpy as np
from sklearn.decomposition import PCA
import scipy
from tqdm import tqdm
import h5py
import argparse
import os
from sklearn import neighbors


def create_path(path):
    if os.path.isdir(path) is False:
        os.makedirs(path)


def main(args):
    global save_path

    if 'mnist' in args.dtype:
        train_path = '{}/bottleneck_dim_{}_lenet_mnist_train_feat.hdf5'.format(args.data_path, args.dim)
        test_path = '{}/bottleneck_dim_{}_lenet_mnist_train_feat.hdf5'.format(args.data_path, args.dim)

        save_path = "../results/mnist/theoretical/knn/lenet/6_16/"
        create_path(save_path)
        save_path = os.path.join(save_path, "knn.mat")

        train_data = h5py.File(train_path, 'r')
        X_train = np.array(train_data["features"])
        Y_train = np.array(train_data["targets"]).reshape(-1)

        test_data = h5py.File(test_path, 'r')
        X_test = np.array(test_data["features"])
        Y_test = np.array(test_data["targets"]).reshape(-1)

    if 'cifar10' in args.dtype:
        train_path = '{}/bottleneck_dim_{}_{}_cifar10_train_feat.hdf5'.format(args.data_path, args.dim, args.mtype)
        test_path = '{}/bottleneck_dim_{}_{}_cifar10_val_feat.hdf5'.format(args.data_path, args.dim, args.mtype)

        save_path = "../results/cifar10/theoretical/knn/{}/".format(args.mtype)
        create_path(save_path)
        save_path = os.path.join(save_path, "knn.mat")

        train_data = h5py.File(train_path, 'r')
        X_train = np.array(train_data["features"])
        Y_train = np.array(train_data["targets"]).reshape(-1)

        test_data = h5py.File(test_path, 'r')
        X_test = np.array(test_data["features"])
        Y_test = np.array(test_data["targets"]).reshape(-1)

    if 'cifar100' in args.dtype:
        train_path = '{}/bottleneck_dim_{}_{}_cifar100_train_feat.hdf5'.format(args.data_path, args.dim, args.mtype)
        test_path = '{}/bottleneck_dim_{}_{}_cifar100_val_feat.hdf5'.format(args.data_path, args.dim, args.mtype)

        save_path = "../results/cifar100/theoretical/knn/{}/".format(args.mtype)
        create_path(save_path)
        save_path = os.path.join(save_path, "knn.mat")

        train_data = h5py.File(train_path, 'r')
        X_train = np.array(train_data["features"])
        Y_train = np.array(train_data["targets"]).reshape(-1)

        test_data = h5py.File(test_path, 'r')
        X_test = np.array(test_data["features"])
        Y_test = np.array(test_data["targets"]).reshape(-1)

    save_data = {
        "scores":[],
        "avg_correct_dist":[],
        "avg_incorrect_dist":[],
        "min_dist":[],
        "max_dist":[],
        "avg_dist":[],
    }

    clf = neighbors.KNeighborsClassifier(args.nn, weights="distance")
    clf.fit(X_train, Y_train)

    Y_pred = clf.predict(X_test)

    d, _ = clf.kneighbors(X_test[Y_pred == Y_test,:], return_distance = True)
    save_data["avg_correct_dist"].append(np.max(d))

    # d, _ = clf.kneighbors(X_test[Y_pred != Y_test,:], return_distance = True)
    save_data["avg_incorrect_dist"].append(0)

    d, _ = clf.kneighbors(X_test, return_distance = True)
    save_data["avg_dist"].append(np.mean(d))
    save_data["max_dist"].append(np.max(d))
    save_data["min_dist"].append(np.min(d))

    score = clf.score(X_test, Y_test)
    save_data["scores"].append(score)

    print("dim:{}, score:{}, avg_dist:{}, min_dist:{}, max_dist:{}, avg_correct_dist:{}, avg_incorrect_dist:{}".format(
          dim, 
          score,
          save_data["avg_dist"][-1],
          save_data["min_dist"][-1],
          save_data["max_dist"][-1],
          save_data["avg_correct_dist"][-1],
          save_data["avg_incorrect_dist"][-1]))

    return save_data

    

if __name__ == '__main__':
    global save_path
    parser = argparse.ArgumentParser(description='KNN')
    parser.add_argument('--dtype', type=str, default='cifar10')
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--mtype", type=str)
    parser.add_argument("--nn", type=int, help="nearest neighbor")

    args = parser.parse_args()
    save_data = {
        "scores":[],
        "avg_correct_dist":[],
        "avg_incorrect_dist":[],
        "min_dist":[],
        "max_dist":[],
        "avg_dist":[],
    }

    for dim in [2]:
        args.dim = dim
        data = main(args)

        for key, value in data.items():
            save_data[key] += value

    scipy.io.savemat(save_path, save_data)