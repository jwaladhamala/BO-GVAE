#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 16:15:00 2017

@author: jd1336

Cardiac Electrophysiological Model: Aliev Panfilov
"""

import matplotlib.cm as cm
import numpy as np
import torch
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint



class CardiacModel(object):
    """Simulation of Aliev-panfilov Model
    Attrs:
        k: model parameter
        e: model parameter
        lb: lower bound on parameter a
        ub: upper bound on parameter a
    Args:
        simu: simulations parameters (initial activation, MK)
        obs: measured EKG signals
        true_par: ground truth tissue properties/parameter a
        cor_mfree: meshfree nodes
        H: transfer matrix
    """
    k = 8
    e = 0.01
    lb = 0
    ub = 0.5

    def __init__(self, simu, obs, true_par, cor_mfree, maskidx_12lead=0,
                 device='cpu'):

        # Measurement data (t, EKG)
        self.dataf = obs.bsp
        self.datat = obs.time

        # Simulation parameters (initial excitation, M*K)
        self.y0 = np.require(simu.X0, requirements=['C'])
        self.ts = np.require(simu.s, requirements=['C'])

        # Forward matrix
        self.H = np.require(simu.H, requirements=['C'])

        # Other parameters
        self.dim = simu.H.shape[1]  # number of meshfree nodes
        self.cor_mfree = cor_mfree  # 3D coordinates of meshfree nodes
        self.true_par = true_par  # ground truth parameters
        self.maskidx_12lead = maskidx_12lead  # if 12 lead ECG is used
        self.num_leads = obs.bsp.shape[0]  # number of leads
        self.device = device

    @staticmethod
    def fp(y, t, k, e, s, dim, par):
        """the Aliev-panfilov model
        """
        u = y[0:dim]
        v = y[dim:dim * 2]
        dudt = s.dot(u) + k * u * (1 - u) * (u - par) - u * v
        dvdt = -e * (k * u * (u - par - 1) + v)
        dydt = np.r_[dudt, dvdt]
        return dydt

    def simulate_ecg(self, fullpar, vae=None):
        """Simulates EKG using AP model
        
        Args:
            fullpar: model parameters
            vae: trained VAE model
        """
        if vae and len(fullpar) == vae.latent_dim:
            fullpar = self.mapparam(fullpar, vae)  # mean from generative model

        # calculate transmural action potential (TMP)
        sol = odeint(self.fp, self.y0, self.datat,
                     args=(self.k, self.e, self.ts, self.dim, fullpar))
        tmp = sol[:, 0:self.dim].transpose()

        # compute the ECG measurement
        bsp = self.H.dot(tmp)

        # if 12-lead ECG is used extract those from full 120-lead
        if (self.num_leads == 12):
            bsp = self.get_12leads(bsp)
        return self.datat, tmp, bsp

    def get_12leads(self, bsp):
        """Extract 12-lead EKG from 120-lead EKG
        """
        bsp120 = bsp[self.maskidx_12lead - 1, :]
        RA = (bsp120[15 - 3 - 1, :] + bsp120[16 - 3 - 1, :]) / 2
        LA = (bsp120[64 - 3 - 1, :] + bsp120[65 - 3 - 1, :]) / 2
        LL = bsp120[70 - 3 - 1, :]
        I = LA - RA
        II = LL - RA
        III = LL - LA
        aVR = -(I + II) / 2;
        aVL = (I - III) / 2;
        aVF = (II + III) / 2;
        V1 = bsp120[25 - 3 - 1, :]
        V2 = bsp120[39 - 3 - 1, :]
        V3 = (bsp120[46 - 3 - 1, :] + bsp120[53 - 3 - 1, :] + bsp120[47 - 3 - 1, :]
              + bsp120[54 - 3 - 1, :]) / 4
        V4 = bsp120[61 - 3 - 1, :]
        V5 = (bsp120[68 - 3 - 1, :] + 2 * bsp120[72 - 3 - 1, :]) / 3
        V6 = bsp120[76 - 3 - 1, :]
        bsp12 = np.asarray([I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6])
        return bsp12

    def compute_objfunc(self, fullpar, vae=None):
        """ Compute the objective function (negative sum of squared error)
        between the simulated bsp and measured bsp
        
        Args:
            fullpar: model parameter
            vae: trained VAE model
        """
        if vae and len(fullpar) == vae.latent_dim:
            fullpar = self.map_param(fullpar, vae)  # mean of the generative model

        # calculate transmural action potential (TMP)
        sol = odeint(self.fp, self.y0, self.datat, args=(self.k, self.e, self.ts, self.dim, fullpar))
        tmp = sol[:, 0:self.dim].transpose()

        # compute the simulated ECG
        bsp = self.H.dot(tmp)

        # if 12-lead ECG is measured then extract 12-lead from simulated 120-lead
        if self.num_leads == 12:
            bsp = self.get_12leads(bsp)

        # compute negative of sum of squared error
        diff = ((self.dataf - bsp) ** 2)
        nsse = - diff.sum()
        return nsse

    def map_param(self, latent_code, vae):
        """ Maps parameters from z-space to original parameter space. 
        Use the expectation of the generative model on z.
        
        Args:
            latent_code: latent_code (z space) 
        """
        z_mu = torch.from_numpy(np.array([latent_code] * vae.batch_size)).float()
        z_mu = z_mu.to(self.device)
        with torch.no_grad():
            x_mean = vae.decode(z_mu)  # expectation of the generative model
            fullpar = (x_mean[0]).cpu().numpy()
        #        if self.use_cpd:
        #            fullpar = fullpar[self.correspond]
        return fullpar

    def plot_param(self, fullpar, vae=None, ax=None):
        """Plot the paramters/tissue properties on the cardiac mesh
        """
        if vae and len(fullpar) == vae.latent_dim:
            fullpar = self.mapparam(fullpar, vae)

        if not ax:
            fig = pyplot.figure()
            ax = Axes3D(fig)
        ax.scatter(self.cor_mfree[:, 0], self.cor_mfree[:, 1], self.cor_mfree[:, 2],
                   s=20, c=fullpar, vmin=self.lb, vmax=self.ub, cmap=cm.get_cmap('jet'))
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        # pyplot.show()

    def plot_gt(self, ax=None):
        """Plot the ground truth parameter in cardiac mesh
        """
        fullpar = self.true_par
        if not ax:
            fig = pyplot.figure()
            ax = Axes3D(fig)
        ax.scatter(self.cor_mfree[:, 0], self.cor_mfree[:, 1], self.cor_mfree[:, 2],
                   s=20, c=fullpar, vmin=self.lb, vmax=self.ub, cmap=cm.get_cmap('jet'))
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        # pyplot.show()
