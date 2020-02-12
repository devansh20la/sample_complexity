import numpy as np
from sklearn.decomposition import PCA
import scipy
from tqdm import tqdm
import h5py
import argparse
import os
from sklearn import neighbors
from sklearn.model_selection import train_test_split

def create_path(path):
    if os.path.isdir(path) is False:
        os.makedirs(path)


def main(args):

    if 'mnist' in args.dtype:
        path = "../data/mnist/bottleneck_mnist_lenet_feat.hdf5"
        data = h5py.File(path, 'r')

        save_path = "../results/mnist/theoretical/knn/lenet/6_16/"
        create_path(save_path)
        save_path = os.path.join(save_path, "knn.mat")

    if 'cifar10' in args.dtype:
        path = "../data/cifar/bottleneck_{}_{}_feat.hdf5".format(args.dtype, args.mtype)
        data = h5py.File(path, 'r')

        save_path = "../results/cifar10/theoretical/knn/{}/".format(args.mtype)
        create_path(save_path)
        save_path = os.path.join(save_path, "knn.mat")

    if 'cifar100' in args.dtype:
        path = "../data/cifar/bottleneck_{}_{}_feat.hdf5".format(args.dtype, args.mtype)
        data = h5py.File(path, 'r')

        save_path = "../results/cifar100/theoretical/knn/{}/".format(args.mtype)
        create_path(save_path)
        save_path = os.path.join(save_path, "knn.mat")

    if 'imagenet' in args.dtype:
        path = "../data/imagenet/bottleneck_{}_{}_feat.hdf5".format(args.dtype, args.mtype)
        data = h5py.File(path, 'r')

        save_path = "../results/imagenet/theoretical/knn/{}/".format(args.mtype)
        create_path(save_path)
        save_path = os.path.join(save_path, "knn.mat")

    save_data = {
        "scores":[],
    }
    
    for dim in [2]:
        args.dim = dim

        for args.data_size in ['0.03125','0.0625','0.125','0.25','0.5','0.75','1.0']:

            X_train = np.array(data[str(args.data_size)+"/train/features"])
            Y_train = np.array(data[str(args.data_size)+"/train/targets"]).reshape(-1)
            
            X_test = np.array(data[str(args.data_size)+"/val/features"])
            Y_test = np.array(data[str(args.data_size)+"/val/targets"]).reshape(-1)

            # pca = PCA(n_components=dim, svd_solver='full')
            # pca.fit(X_train)
            # tqdm.write("Variance Ratio:{0}\n#components:{1}".format(
            #     pca.explained_variance_ratio_, pca.explained_variance_ratio_.shape))
            # X_train_dim = pca.transform(X_train)
            # X_test_dim = pca.transform(X_test)

            X_train_dim = X_train
            X_test_dim = X_test

            errors = np.zeros((Y_test.shape[0],1))

            clf = neighbors.KNeighborsClassifier(args.nn, weights="uniform")
            clf.fit(X_train_dim, Y_train)

            Y_test_pred = clf.predict(X_test_dim)
            errors[Y_test != Y_test_pred] = 1

            Y_test_dist,_ = clf.kneighbors(X_test_dim, return_distance=True)
            errors[Y_test_dist>17.38] = 1

            errors[errors == 2] = 1
            print(errors)

            save_data["scores"].append(np.mean(errors))                
            print(save_data["scores"])
            scipy.io.savemat(save_path, save_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KNN')
    parser.add_argument('--dtype', type=str, default='cifar10')
    parser.add_argument("--data_size", type=int, help="nearest neighbor")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--mtype", type=str)
    parser.add_argument("--nn", type=int, help="nearest neighbor")

    args = parser.parse_args()

    main(args)