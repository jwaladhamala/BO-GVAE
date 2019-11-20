import pickle
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

import net
import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_losses(train_a, test_a, model_dir, num_epochs):
    """Plot epoch against train loss and test loss 
    """
    # plot of the train/validation error against num_epochs
    fig, ax1 = plt.subplots(figsize=(6, 5))
    ax1.set_xticks(np.arange(0 + 1, num_epochs + 1, step=10))
    ax1.set_xlabel('epochs')
    ax1.plot(train_a, color='green', ls='-', label='train accuracy')
    ax1.plot(test_a, color='red', ls='-', label='test accuracy')
    h1, l1 = ax1.get_legend_handles_labels()
    ax1.legend(h1, l1, fontsize='14', frameon=False)
    ax1.grid(linestyle='--')
    plt.tight_layout()
    fig.savefig(model_dir + '/accplot.png', dpi=300, bbox_inches='tight', transparent=True)
    # plt.show()


def plot_reconstructions(model, x_loader, model_dir, corMfree):
    """Plot inputs and reconstructions
    """
    batch_size = model.batch_size
    model.eval()
    with torch.no_grad():
        data = next(iter(x_loader))
        data = next(iter(x_loader))
        data = data.to(device)
        recon_data, _, _ = model(data)

    N = 4
    inds = np.random.permutation(batch_size)
    x_sample = data.x.view(batch_size, -1)
    x_sample = np.squeeze(x_sample.detach().cpu().numpy())
    x_reconstruct = recon_data.view(batch_size, -1)
    x_reconstruct = np.squeeze(x_reconstruct.detach().cpu().numpy())  # encode then decode

    # plot the figure
    fig = plt.figure(figsize=(10, 5))
    for i in range(N):
        ax1 = fig.add_subplot(2, N, i + 1, projection='3d')
        p1 = ax1.scatter(corMfree[:, 0], corMfree[:, 1], corMfree[:, 2], c=x_sample[inds[i]],
                         vmin=0.15, vmax=0.51, cmap=plt.cm.get_cmap('jet'))
        fig.colorbar(p1)
        ax2 = fig.add_subplot(2, N, N + (i + 1), projection='3d')
        p2 = ax2.scatter(corMfree[:, 0], corMfree[:, 1], corMfree[:, 2], vmin=0.15, vmax=0.51,
                         c=x_reconstruct[inds[i]], cmap=plt.cm.get_cmap('jet'))
        fig.colorbar(p2)
    plt.tight_layout()
    fig.savefig(model_dir + '/recons.png', dpi=300, bbox_inches='tight', transparent=True)


def plot_zmean(model, x_loader, model_dir, corMfree):
    """Plot the latent codes
    """
    batch_size = model.batch_size
    latent_dim = model.latent_dim
    n = len(x_loader.dataset)
    n = (n - n % batch_size)
    num_meshfree = len(corMfree)
    z_mu = np.empty((n, latent_dim))
    label = np.empty((n))
    all_recons = np.empty((n, num_meshfree))
    all_inps = np.empty((n, num_meshfree))
    model.eval()
    i = 0
    with torch.no_grad():
        for data in (x_loader):
            data = data.to(device)
            recon_data, mu, _ = model(data)
            mu = mu.view(batch_size, latent_dim)
            mu = mu.detach().cpu().numpy()
            y = data.y.detach().cpu().numpy()
            y = data.y.detach().cpu().numpy()
            z_mu[i * batch_size:(i + 1) * batch_size, :] = mu
            label[i * batch_size:(i + 1) * batch_size] = y
            all_recons[i * batch_size:(i + 1) * batch_size, :] = recon_data.detach().cpu().numpy()
            all_inps[i * batch_size:(i + 1) * batch_size, :] = data.x.view(batch_size, -1).detach().cpu().numpy()
            i += 1

    fig = plt.figure(figsize=(8, 6))
    plt.scatter(z_mu[:, 0], z_mu[:, 1], c=np.squeeze(label),  # specify the labels
                cmap=plt.cm.get_cmap('jet'))
    plt.colorbar()
    plt.grid()
    fig.savefig(model_dir + '/ld.png', dpi=300, bbox_inches='tight', transparent=True)
    # plt.show()

    dc = utils.calc_dc(all_inps, all_recons)
    msse, mmae = utils.calc_msse(all_inps, all_recons)
    dch, dcs = utils.calc_dc_fixedthres(all_inps, all_recons)
    print(' dc: {:05.5f}, sse: {:05.5f}, dch: {:05.5f}, dcs: {:05.5f},mmae: {:05.5f}'
          .format(dc, msse, dch, dcs, mmae))



def train(epoch, model, optimizer, train_loader, batch_size, model_dir):
    """Train a model and compute train loss
    """
    model.train()
    train_loss = 0
    n = 0  # len(train_dataset)
    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        recon_batch, mu, logvar = model(data)
        loss = net.loss_function(recon_batch, data.x.view(batch_size, -1), mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        n += 1
        utils.inline_print(f'Running epoch {epoch}, batch {n}, Average loss for epoch: {str(train_loss / (n * batch_size))}')

    torch.save(model.state_dict(), model_dir + '/m_latest')
    if epoch % 10 == 0:
        torch.save(model.state_dict(), model_dir + '/m_' + str(epoch))
        # calculate and save q(z)
        calc_marginal_qz(model_dir, model, train_loader)
    return (train_loss / (n * batch_size))


def test(model, test_loader, batch_size):
    """Evaluated a trained model by computing validation loss
    """
    model.eval()
    test_loss = 0
    n = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += net.loss_function(recon_batch, data.x.view(batch_size, -1), mu, logvar).item()
            n += 1
    return (test_loss / (n * batch_size))


def train_vae(model, optimizer, train_loader, test_loader,
              model_dir, num_epochs, batch_size, corMfree):
    """
    """
    train_a = []
    test_a = []
    for epoch in range(1, num_epochs + 1):
        ts = time.time()
        train_acc = train(epoch, model, optimizer, train_loader, batch_size, model_dir)
        test_acc = test(model, test_loader, batch_size)
        te = time.time()
        train_a.append(train_acc)
        test_a.append(test_acc)
        with open(model_dir + '/errors.pickle', 'wb') as f:
            pickle.dump(te - ts, f, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(epoch, f, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(train_a, f, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(test_a, f, protocol=pickle.HIGHEST_PROTOCOL)
        print('Epoch: {:03d}, Time: {:.4f}, Train: {:.4f}, Test: {:.4f}'.format(epoch, (te - ts) / 60, train_acc,
                                                                                test_acc))
        # TODO: Also write to file print('Epoch: {:03d}, Time: {:.4f}, Train: {:.4f}, Test: {:.4f}'.format(epoch, (te-ts)/60, train_acc, test_acc))

    plot_losses(train_a, test_a, model_dir, num_epochs)
    plot_reconstructions(model, test_loader, model_dir, corMfree)
    plot_zmean(model, test_loader, model_dir, corMfree)

def get_network_paramcount(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    return num_params

    
def eval_vae(model, train_loader, test_loader, model_dir, 
             batch_size, corMfree):
    """
    """
    # 
    num_params = get_network_paramcount(model)
    print('The number of network prameters: {}\n'.format(num_params))
    plot_reconstructions(model, test_loader, model_dir, corMfree)
    plot_zmean(model, test_loader, model_dir, corMfree)
    
# TODO: test; this code was added from MICCAI 2018 code and has not been tested extensively
def calc_marginal_qz(model_dir, model, x_loader):
    """
    :param x_loader: data loader
    :param model:trained vae model

    :return q(z) approximated as a single Gaussian distribution
    """
    batch_size = model.batch_size
    latent_dim = model.latent_dim
    n = len(x_loader.dataset)
    n = (n - n % batch_size)
    mu_all = np.empty((n, latent_dim))
    logvar_all = np.empty((n, latent_dim))
    i = 0
    model.eval()
    with torch.no_grad():
        for data in x_loader:
            data = data.to(device)
            _, mu, logvar = model(data)
            mu = mu.view(batch_size, latent_dim)
            mu = mu.detach().cpu().numpy()
            logvar = logvar.view(batch_size, latent_dim)
            logvar = logvar.detach().cpu().numpy()
            mu_all[i * batch_size:(i + 1) * batch_size, :] = mu
            logvar_all[i * batch_size:(i + 1) * batch_size, :] = logvar
            i+=1


    var_all = np.exp(logvar_all)
    print(var_all)
    Exxt = 0
    for i in range(n):
        mu_i = mu_all[i, :]
        var_i = np.diag(var_all[i, :])
        Exxt = Exxt + (var_i + mu_i.reshape((-1, 1)) * mu_i)
    mu_hat = 1 / n * np.sum(mu_all, 0)
    var_hat = 1 / n * Exxt - mu_hat.reshape((-1, 1)) * mu_hat
    invvar_hat = np.linalg.inv(var_hat)

    print(mu_hat)
    print(invvar_hat)

    with open(model_dir + "/z_posterior.pkl", 'wb') as f:
        pickle.dump(mu_hat, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(invvar_hat, output, pickle.HIGHEST_PROTOCOL)


