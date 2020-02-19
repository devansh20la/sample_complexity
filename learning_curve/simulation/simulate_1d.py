
import random
import time
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import erfi
from scipy.integrate import quad


def gauss_rand_numpy(mu, sigma):
    """Numpy gaussian random number generator."""
    sample = np.random.normal(mu, sigma, 1)
    return float(sample[0])

def get_traintest_data1D(num_samples, num_te_samples, mu=0.0, sigma=1):
    """Returns 1D gaussian train and test datasets."""
    trds = []
    for i in range(num_samples):
        sample = gauss_rand_numpy(mu, sigma)
        trds.append(sample)
    trds.sort()

    teds = []
    for i in range(num_te_samples):
        sample = gauss_rand_numpy(mu, sigma)
        teds.append(sample)
    teds.sort()

    return trds, teds

def find_xt(teData, Xj, te_idx, M):
    """Find all test points between Xi and Xj."""
    xt_temp = []
    if te_idx >= M - 1:
        return xt_temp, M
    else:    
        while te_idx <= M - 1 and teData[te_idx] <= Xj:
            xt_temp.append(teData[te_idx])
            te_idx += 1
    return xt_temp, te_idx

def est_loss(trData, teData, sigma, delta, power):
    """Experimental simulation of E[\Phi]. power = 1 or 2 for RMSE and MSE distance measures respectively."""
    M = len(teData)
    N = len(trData)
    
    te_idx = 0
    E_loss = 0 
    for i in range(len(teData)):
        if teData[i] < trData[0]:
            te_idx += 1
            E_loss += min(1, ((trData[0] - teData[i])**power)/ delta)
        else:
            break   
        
    for i in range(len(trData)-1):
        Xi = trData[i]
        Xj = trData[i+1]
        xt, te_idx = find_xt(teData, Xj, te_idx, M)
        Xi_Xj_mean = (Xi + Xj)/2
        for test_sample in xt:
            if test_sample < Xi_Xj_mean:
                abs_diff = test_sample - Xi
            else:
                abs_diff = Xj - test_sample
            E_loss += min(1, (abs_diff**power) / delta)
    
    while te_idx <= M - 1:
        if teData[te_idx] >= trData[N - 1]:
            E_loss += min(1, ((teData[te_idx] - trData[N - 1])**power) / delta)
        te_idx += 1
     
    E_loss = E_loss / M

    return E_loss


#####################################

save_dir= './Results_1d/'

if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

sigma_list = [100]
N_list_temp = [10, 40, 160, 640, 1280]
Nx = 1000
N_list = [i * int(Nx) for i in N_list_temp]
M_list = [10000]
delta_values_list = [1]
repeatx = 20
pw = 1

setting_num = 0
total_num_settings = len(sigma_list)*len(N_list)*len(M_list)*len(delta_values_list)*repeatx


for sigma in sigma_list:
    for M in M_list:
        for delta_value in delta_values_list:
            save_results = []
            for N in N_list:
                running_mean = []
                for iter in range(repeatx):
                    setting_num += 1
                    if setting_num % repeatx == 0:
                        print("Setting: "+ str(setting_num)+'/'+str(total_num_settings))
                    trData, teData = get_traintest_data1D(N, M, mu=0, sigma=sigma)
                    rmean = est_loss(trData, teData, sigma, delta_value, pw)
                    running_mean.append(rmean)

                exp_loss = np.mean(running_mean)
                std_loss = np.std(running_mean)
                if pw == 1:
                    if np.log((2 * N * delta_value) / math.sqrt(2 * math.pi * (sigma ** 2))) >= 0: 
                        x_star = sigma * math.sqrt(2) * math.sqrt(np.log((2 * N * delta_value) / math.sqrt(2 * math.pi * (sigma ** 2))))
                        comp_cum_prob = 1.0 - norm.cdf(x_star, loc = 0.0, scale = sigma)
                        exp_loss_theory = 2 * comp_cum_prob + (x_star / (N * delta_value))
                        std_loss_theory = 0

                        newrow = [N, exp_loss_theory, std_loss_theory, exp_loss, std_loss]
                        save_results.append(newrow)
                if pw == 2:
                    if np.log(math.sqrt(3 * N * N * delta_value) / math.sqrt(2 * math.pi * (sigma ** 2))) >= 0:
                        x_star = sigma * math.sqrt(2) * math.sqrt(np.log(math.sqrt(3 * N * N * delta_value) / math.sqrt(2 * math.pi * (sigma ** 2))))
                        comp_cum_prob = 1.0 - norm.cdf(x_star, loc = 0.0, scale = sigma)
                        exp_loss_theory = 2 * comp_cum_prob + (( sigma**2 * math.pi * 2) / ( 3 *N * N * delta_value)) * erfi(x_star/(math.sqrt(2)*sigma))
                        std_loss_theory = 0

                        newrow = [N, exp_loss_theory, std_loss_theory, exp_loss, std_loss]
                        save_results.append(newrow)

            save_results_numpy = np.array(save_results)
            np.save(save_dir+'/dim_1_delta_'+str(delta_value)+'_M_'+str(M)+'_Nx_'+str(Nx)+'_pw_'+str(pw)+'_rx_'+str(repeatx)+'_sigma_'+str(sigma)+'.npy', save_results_numpy) 
