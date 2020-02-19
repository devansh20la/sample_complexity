import numpy as np
import matplotlib.pyplot as plt


dim_list = [2, 4]
M_list = [1000]
Nx = 100
sigma_list = [1]
delta_values_list = [1]
pw = 1
repeatx = 20

if dim_list[0] == 1:
    saved_dir = './Results_1d/'
else:
    saved_dir = './Results_hd/'

for dim in dim_list:
    for sigma in sigma_list:
        for M in M_list:
            for delta_value in delta_values_list:
                saved_results = np.load(saved_dir+'dim_'+str(dim)+'_delta_'+str(delta_value)+'_M_'+str(M)+'_Nx_'+str(Nx)+'_pw_'+str(pw)+'_rx_'+str(repeatx)+'_sigma_'+str(sigma)+'.npy')
                save_fig_name = 'dim_'+str(dim)+'_delta_'+str(delta_value)+'_M_'+str(M)+'_Nx_'+str(int(Nx))+'_pw_'+str(pw)+'_rx_'+str(repeatx)+'_sigma_'+str(sigma)

                exp_loss_list = list(saved_results[:, 3])
                std_loss_list = list(saved_results[:, 4])
                exp_loss_theory_list = list(saved_results[:, 1])
                std_loss_theory_list = list(saved_results[:, 2])
                N_list = list(saved_results[:, 0])

                #plot
                plt.figure()
                ax = plt.subplot(111)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.get_xaxis().tick_bottom()
                ax.get_yaxis().tick_left()
                plt.xlabel('Training Set Size', fontsize=22)
                plt.ylabel('E['+r'$\Phi$'+']', fontsize=22)
                ax.ticklabel_format(style='sci',scilimits=(0,0),axis='x', useOffset=True, useMathText=True)
                ax.xaxis.offsetText.set_fontsize(fontsize=22)
                plt.xticks(fontsize=25)
                plt.yticks(fontsize=25)
                plt.fill_between(N_list, list(np.array(exp_loss_list) - np.array(std_loss_list)*2), list(np.array(exp_loss_list) + np.array(std_loss_list)*2), facecolor="blue", alpha=0.5)
                plt.plot(N_list, exp_loss_list, color="blue", lw=2, label='experiment')
                plt.fill_between(N_list, list(np.array(exp_loss_theory_list) - np.array(std_loss_theory_list)*2), list(np.array(exp_loss_theory_list) + np.array(std_loss_theory_list)*2), color="red", alpha=0.5)
                plt.plot(N_list, exp_loss_theory_list, color="red", lw=2, label='theory')
                plt.legend(loc='upper right', fontsize=22)
                ax.grid(which='minor', linestyle='-', linewidth='0.2', color='gray')
                plt.tight_layout()
                plt.savefig(saved_dir+save_fig_name+'.png')
                plt.xscale('log')
                plt.yscale('log')
                plt.savefig(saved_dir+save_fig_name+'_log.png')
                plt.close()
