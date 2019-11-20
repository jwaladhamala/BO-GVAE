import os.path as osp
import pickle
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
from skimage.filters import threshold_otsu

sys.path.append('/bayesopt/')

from bayes_opt import BayesianOptimization
import cardiacmodel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def hdbayesopt(vae, dpath, rpath, files, p_dim, acq_list=None, niter=None,
               inipts=None, z_mu_1=None, z_var_1=None, verbose=0):
    """Run High Dimensional Bayesion Optimization Experiments
    
    Args:
        vae: trained model
        dpath: input path fro experiments
        rpath: output path fore optimization results
        files: files of the experiments to run
        p_dim: number of meshfree node in given heart
        acq_list: acquisation functions to use
        niter: number of iterations for optimization
        inipts: number of initial input points
    """
    latent_dim = vae.latent_dim

    # set the total number of initial points
    # and iterations for optimization based on the size of latent_dim
    map_dim_niter = {2: 85, 3: 150, 5: 300}
    map_dim_inipts = {2: 5, 3: 10, 5: 20}
    if not niter:
        niter = map_dim_niter[latent_dim]
    if not inipts:
        niter = map_dim_inipts[latent_dim]
    if not acq_list:
        acq_list = ('ei')


    acq_list = tuple(acq_list)
    num_acq = len(acq_list)  # number of acquisation function
    num_exps = len(files)  # number of experiments

    parUnknownId = list(range(1, latent_dim + 1))  # if for each unknown
    bounds = [(-4, 4) for ij in parUnknownId]  # bounds on the optimization
    parUnknownId = [str(ij) for ij in parUnknownId]  # convert id to string

    # initialize variables to collect data
    timetaken = np.zeros((num_exps, num_acq))
    dicecoeff = np.zeros((num_exps, num_acq))
    rmse = np.zeros((num_exps, num_acq))
    fopt = np.zeros((num_exps, num_acq))  # optimum value of the function
    paramEstRes = np.zeros((num_exps, num_acq, p_dim))  # estimated parameter at original dim
    paramEstRes_z = np.zeros((num_exps, num_acq, latent_dim))  # estimated z
    paramGT = np.zeros((num_exps, num_acq, p_dim))  # ground truth

    # loop through each experiment for parameter estimation
    for i in range(num_exps):
        # read the experiment that is in matlab format
        fname = files[i] + '.mat'
        matFiles = scipy.io.loadmat(dpath + '/' + fname, squeeze_me=True, struct_as_record=False)
        parTrue = matFiles['parTrue']
        obs = matFiles['obs']
        simu = matFiles['simu']
        corMfree = matFiles['corMfree']

        # instance of cardaic model
        cardiac_model = cardiacmodel.CardiacModel(simu, obs, parTrue, corMfree, maskidx_12lead=0,
                                                 device=device)
        thresh_gt = 0.18  # threshold_otsu(cardiac_model.parTrue)
        idx_gt = np.where(cardiac_model.true_par >= thresh_gt)[0]
        # t,tmp,bsp=cardiac_model.simulate_ecg(parTrue)

        # for each exp loop through each acq func
        for j in range(num_acq):
            gp_surr = BayesianOptimization(cardiac_model.compute_objfunc,
                                           dict(zip(parUnknownId, bounds)), vae, verbose=verbose)
            acq_func = acq_list[j]
            tstart = time.time()
            if (acq_func == 'ei'):
                gp_surr.maximize(init_points=inipts, n_iter=niter, acq='ei', xi=0.0001)
            elif (acq_func == 'ei_prior'):
                gp_surr.maximize(init_points=inipts, n_iter=niter, acq='ei_prior', xi=1.0)
            elif (acq_func == 'ei_post_agg'):
                gp_surr.maximize(init_points=inipts, n_iter=niter, acq='ei_post_agg',
                                 z_m=z_mu_1, z_v=z_var_1, xi=1.0)
            else:
                # TODO bring implementation of other acquisation function in thsi code
                print('incorrect acq func')
            tend = time.time()
            # optimum found 
            xmax = gp_surr.res['max']['max_params']
            z_mu = torch.from_numpy(np.array([xmax]*vae.batch_size)).float()
            z_mu = z_mu.to(device)
            with torch.no_grad():
                x_mean = vae.decode(z_mu)
                x_mean = (x_mean[0]).cpu().numpy()
            #            if use_cpd:
            #                x_mean = x_mean[correspondance]

            # compute dice coefficient and rmse
            thresh_c = threshold_otsu(x_mean)
            idx_c = np.where(x_mean >= thresh_c)[0]
            rmse_temp = np.sqrt(((cardiac_model.true_par - x_mean) ** 2).mean())
            dicecoeff_temp = 2 * len(np.intersect1d(idx_gt, idx_c)) / (len(idx_gt) + len(idx_c))

            rmse[i, j] = rmse_temp
            dicecoeff[i, j] = dicecoeff_temp
            paramEstRes[i, j, :] = x_mean
            paramEstRes_z[i, j, :] = xmax
            fopt[i, j] = gp_surr.res['max']['max_val']
            paramGT[i, j, :] = cardiac_model.true_par
            timetaken[i, j] = tend - tstart

            # plot of the true and estimated paraemters
            fname_save = osp.join(rpath, files[i] + '_' + acq_func + '.png')
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 2, 1, projection='3d')
            cardiac_model.plot_gt(ax1)
            ax2 = fig.add_subplot(1, 2, 2, projection='3d')
            cardiac_model.plot_param(x_mean, ax=ax2)
            fig.savefig(fname_save, dpi=300, bbox_inches='tight', transparent=True)
            print('exp #{:03d} with {}, dc:{:.4f}, rmse:{:.4f}, time:{:.4f}'.format(i + 1, acq_func,
                                                                                    dicecoeff_temp, rmse_temp,
                                                                                    timetaken[i, j] / 60))
            # print('exp # {}' + str(i+1)+ ': ' + acq_func + ': '
            #      + str(dicecoeff_temp) + ', ' str(rmse_temp) + ', ' + str(tend-tstart))
            del gp_surr
            del xmax
            del z_mu
            del x_mean
            del thresh_c
            del idx_c
            del rmse_temp
            del dicecoeff_temp

            # save optimization result in pickle format
            fname_save = osp.join(rpath, files[i] + '_' + str(latent_dim) + '_d.pkl')
            with open(fname_save, 'wb') as output:
                pickle.dump(fname, output, pickle.HIGHEST_PROTOCOL)
                pickle.dump(cardiac_model, output, pickle.HIGHEST_PROTOCOL)
                pickle.dump(paramEstRes, output, pickle.HIGHEST_PROTOCOL)
                pickle.dump(paramEstRes_z, output, pickle.HIGHEST_PROTOCOL)
                pickle.dump(rmse, output, pickle.HIGHEST_PROTOCOL)
                pickle.dump(dicecoeff, output, pickle.HIGHEST_PROTOCOL)
                pickle.dump(fopt, output, pickle.HIGHEST_PROTOCOL)
                pickle.dump(paramGT, output, pickle.HIGHEST_PROTOCOL)
                pickle.dump(timetaken, output, pickle.HIGHEST_PROTOCOL)

        del cardiac_model
        del idx_gt
