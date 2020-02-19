import random
import time
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import spatial
from scipy.stats import multivariate_normal as mvn
import os
import argparse


def gauss_rand_numpy(dim, sigma):
    """Numpy gaussian random number generator."""
    mean = np.zeros(dim) # mu = 0
    cov = np.zeros((dim, dim))
    for i in range(dim):
        cov[i][i] = sigma ** 2
    x = np.random.multivariate_normal(mean, cov, 1).T
    return x

def get_traintest_dataHD(dim, sigma, num_tr_samples, num_te_samples):
    """Returns HD gaussian train and test datasets."""
    trds = []
    for i in range(num_tr_samples):
        sample = gauss_rand_numpy(dim=dim, sigma=sigma).flatten()
        trds.append(tuple(sample))

    teds = []
    for i in range(num_te_samples):
        sample = gauss_rand_numpy(dim=dim, sigma=sigma).flatten()
        teds.append(tuple(sample))

    return trds, teds

def est_loss(trData, teData, sigma, delta, power):
    """Computes E[\Phi] on testset"""
    M = len(teData)
    E_loss = 0 
    tree = spatial.KDTree(trData)
    for i in range(len(teData)):
        d, tr_idx = tree.query(teData[i])
        E_loss += min(1, d**power/delta)
    
    E_loss = E_loss / M

    return E_loss


#####################################


save_dir= './Results_hd/'

if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

dim_list = [2, 4]
sigma_list = [1]
N_list_temp = [ 10, 40, 160, 640, 1280]
Nx = 100
N_list = [i * int(Nx) for i in N_list_temp]
M_list = [1000]
delta_values_list = [1]
repeatx = 20
pw = 1

setting_num = 0
total_num_settings = len(dim_list)*len(sigma_list)*len(N_list)*len(M_list)*len(delta_values_list)*repeatx


for dim in dim_list:
    if dim==2:
        # We observe the correction for local variance vanish for dimensions>1 (see correctionHD.py).
        # But, in 2D simulation experiments the theory matches simulation for cube side length ~1.2a.
        mulby=0.6
    else:
        # For dim = 4 we dint need the correction.
        mulby=0.5 
    for sigma in sigma_list:
        mean = np.zeros(dim)
        cov = np.zeros((dim, dim))
        for i in range(dim):
            cov[i][i] = sigma ** 2
        distribution = mvn(mean=mean, cov=cov)
        for M in M_list:
            for delta_value in delta_values_list:
                save_results = []
                for N in N_list:
                    running_mean = []
                    running_mean_theory = []
                    for iter in range(repeatx):
                        setting_num += 1
                        if setting_num % repeatx == 0:
                            print("Setting: "+ str(setting_num)+'/'+str(total_num_settings))
                        trData, teData = get_traintest_dataHD(dim=dim, sigma=sigma, num_tr_samples=N, num_te_samples=M)
                        f = np.zeros((len(teData), 1))
                        loss_sum_theory = 0
                        for te_idx in range(len(teData)):
                            # compute a 
                            limit = 1/((N * distribution.pdf(np.array(teData[te_idx])))**(1/dim))
                            #iniformly sample test points in cube of side a
                            uni = np.random.uniform(low=0, high=float(limit * mulby), size=(100000, dim))
                            #distance
                            if pw == 1:
                                f[te_idx][0] = np.mean(np.sqrt(np.sum(uni**2, axis=1)))
                            if pw == 2:
                                f[te_idx][0] = np.mean(np.sum(uni**2, axis=1))
                            loss_sum_theory += min(1, f[te_idx][0]/delta_value)

                        rmean = est_loss(trData, teData, sigma, delta_value, power=pw)
                        running_mean.append(rmean)
                        rmean_theory = loss_sum_theory / M
                        running_mean_theory.append(rmean_theory)

                    exp_loss = np.mean(running_mean)
                    std_loss = np.std(running_mean)
                    exp_loss_theory = np.mean(running_mean_theory)
                    std_loss_theory = np.std(running_mean_theory)
                    newrow = [N, exp_loss_theory, std_loss_theory, exp_loss, std_loss]
                    save_results.append(newrow)
                
                save_results_numpy = np.array(save_results)
                np.save(save_dir+'/dim_'+str(dim)+'_delta_'+str(delta_value)+'_M_'+str(M)+'_Nx_'+str(Nx)+'_pw_'+str(pw)+'_rx_'+str(repeatx)+'_sigma_'+str(sigma)+'.npy', save_results_numpy)
