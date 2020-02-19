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

    data = h5py.File(args.data_path, 'r')

    save_path = "results/knn/{}_{}/".format(args.dtype, args.mtype)
    create_path(save_path)
    save_path = os.path.join(save_path, "knn.mat")

    save_data = {
        "scores":[],
    }
    
    for dim in [1,2,3,4,5,6]:
        args.dim = dim


        X_train = np.array(data["{}/train/features".format(args.dim)])
        Y_train = np.array(data["{}/train/targets".format(args.dim)]).reshape(-1)
        
        X_test = np.array(data["{}/val/features".format(args.dim)])
        Y_test = np.array(data["{}/val/targets".format(args.dim)]).reshape(-1)

        errors = np.zeros((Y_test.shape[0],1))

        clf = neighbors.KNeighborsClassifier(args.nn, weights="uniform")
        clf.fit(X_train, Y_train)

        error = clf.score(X_test,Y_test)

        save_data["scores"].append(error)                
        print(save_data["scores"])
    scipy.io.savemat(save_path, save_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KNN')
    parser.add_argument('--dtype', type=str, default='cifar10')
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--mtype", type=str)
    parser.add_argument("--nn", type=int, help="nearest neighbor")

    args = parser.parse_args()

    main(args)