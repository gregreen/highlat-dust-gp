from __future__ import print_function, division

import math
import torch
import gpytorch

import numpy as np
import scipy.integrate
import h5py
import healpy as hp
import os
import json

from matplotlib import pyplot as plt

from rq_kernel import RationalQuadraticKernel
from sph_harmonic_mean import SphericalHarmonicMean


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)

        # Mean
        #self.mean_module = gpytorch.means.ConstantMean()
        a_lm_prior = gpytorch.priors.NormalPrior(0.0, 0.25)
        self.mean_module = SphericalHarmonicMean(prior=a_lm_prior)

        # Kernel
        alpha_prior = gpytorch.priors.NormalPrior(4.5, 2.0)
        ell_prior = gpytorch.priors.NormalPrior(0., np.radians(0.5))
        base_kernel = RationalQuadraticKernel(
            power_law_prior=alpha_prior,
            lengthscale_prior=ell_prior
        )

        # Scale kernel to get overall variance
        kernel = gpytorch.kernels.ScaleKernel(base_kernel)
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def set_parameters(self, alpha=None, ell=None, A=None, a_lm=None):
        if alpha is not None:
            self.covar_module.base_kernel.power_law = alpha
        if ell is not None:
            self.covar_module.base_kernel.lengthscale = ell
        if A is not None:
            self.covar_module.outputscale = A
        if a_lm is not None:
            self.mean_module.a_lm = a_lm


def load_sources(fname):
    chisq, metadata, pctiles = [], [], []

    with h5py.File(fname, 'r') as f:
        pixels = list(f['chisq'].keys())

        for pix in pixels:
            #print('Loading pixel {} ...'.format(pix))

            l0,b0 = hp.pixelfunc.pix2ang(32, int(pix),
                                         nest=True, lonlat=True)
            #print('(l,b) = ({:.2f}, {:.2f})'.format(l0, b0))

            chisq.append(f['chisq'][pix][:])
            metadata.append(f['metadata'][pix][:])
            pctiles.append(f['percentiles'][pix][:])

            #print('n_stars = {:d}'.format(len(chisq[-1])))

    chisq = np.concatenate(chisq)
    metadata = np.concatenate(metadata)
    pctiles = np.concatenate(pctiles)

    return {'chisq':chisq, 'metadata':metadata, 'pctiles':pctiles}


def filter_sources(d, chisq_max=1., sigma_E_max=0.1,
                      d_min=1000., n_sig_d=2., E_min=0.):
    dm_min = 5. * (np.log10(d_min) - 1.)
    print('Minimum DM = {:.3f}'.format(dm_min))

    sigma_E = 0.5 * (d['pctiles']['E'][:,2] - d['pctiles']['E'][:,0])
    sigma_dm = 0.5 * (d['pctiles']['dm'][:,2] - d['pctiles']['dm'][:,0])
    E_med = d['pctiles']['E'][:,1]

    idx_d = (d['pctiles']['dm'][:,1] - n_sig_d*sigma_dm > dm_min)
    idx_sig_E = (sigma_E < sigma_E_max)
    idx_chisq = (d['chisq'] < chisq_max)
    idx = idx_d & idx_sig_E & idx_chisq
    
    def filt_msg(name, i):
        n = np.count_nonzero(i)
        f = 100. * n / i.size
        print(r'  {:d} ({:.1f}%) pass {:s} cut.'.format(n, f, name))

    print('{:d} sources:'.format(idx.size))
    filt_msg('min. dist.', idx_d)
    filt_msg('max. sigma_E', idx_sig_E)
    filt_msg('chi^2/passband', idx_chisq)

    if E_min is not None:
        idx_E_min = (E_med > E_min)
        filt_msg('min. E', idx_E_min)
        idx &= idx_E_min

    filt_msg('the combined', idx)

    d_filt = {k: d[k][idx] for k in d}

    return d_filt


def extract_train_test_data(d, n_train, n_test, log=True):
    l = np.radians(d['metadata']['l'])
    b = np.radians(d['metadata']['b'])

    coords = np.stack((
        np.cos(l) * np.cos(b),
        np.sin(l) * np.cos(b),
        np.sin(b)
    ), axis=1)

    y = d['pctiles']['E'][:,1]
    sy = 0.5 * (d['pctiles']['E'][:,2] - d['pctiles']['E'][:,0])
    print(sy)
    print(np.percentile(sy, [0., 1., 5., 10., 16., 50., 84., 90., 95., 99., 100.]))
    var_y = sy**2 + 0.02**2
    print(var_y)

    if log:
        var_y /= y**2
        y = np.log(y)

    idx = np.arange(y.size)
    np.random.shuffle(idx)

    idx_train = idx[:n_train]
    idx_test = idx[n_train:n_train+n_test]

    train_x = torch.from_numpy(coords[idx_train].astype('f4'))
    train_y = torch.from_numpy(y[idx_train].astype('f4'))
    train_var_y = torch.from_numpy(var_y[idx_train].astype('f4'))

    test_x = torch.from_numpy(coords[idx_test].astype('f4'))
    test_y = torch.from_numpy(y[idx_test].astype('f4'))
    test_var_y = torch.from_numpy(var_y[idx_test].astype('f4'))

    ret = {
        'train': {'x': train_x, 'y': train_y, 'var_y': train_var_y},
        'test': {'x': test_x, 'y': test_y, 'var_y': test_var_y},
        'all': {'x': coords, 'y': y, 'var_y': var_y}
    }

    return ret


def get_model(data):
    #likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
        noise=data['train']['var_y'],
        learn_additional_noise=False
    )
    #likelihood.noise = 0.10

    model = ExactGPModel(
        data['train']['x'],
        data['train']['y'],
        likelihood
    )
    return model, likelihood


def get_progress_fn(model):
    kernel = model.covar_module
    if isinstance(kernel, gpytorch.kernels.ScaleKernel):
        kernel = kernel.base_kernel

    if isinstance(kernel, RationalQuadraticKernel):
        fmt = (
            'it {: >3d} of {:d}:'
            '  loss={: >+6.3f}'
            '  mean={: >+6.3f}'
            '  A={: >5.3f}'
            '  ell={: >5.3f}'
            '  alpha={: >5.3f}'
            '  noise={: >5.3f}'
        )

        def progress(i, n_iter, loss):
            a_lm = model.mean_module.a_lm.tolist()
            print(
                'a_lm = ['
                + ', '.join(['{: >+6.3f}'.format(a) for a in a_lm])
                + ']'
            )

            txt = fmt.format(
                i+1, n_iter, loss.item(),
                a_lm[0] + a_lm[2] + 2.*a_lm[6] + 2.*a_lm[12],
                #model.mean_module.constant.item(),
                model.covar_module.outputscale.item(),
                np.degrees(model.covar_module.base_kernel.lengthscale.item()),
                model.covar_module.base_kernel.power_law.item(),
                model.likelihood.noise.tolist()[0]
            )
            return txt
        
        return progress
    elif isinstance(kernel, gpytorch.kernels.RBFKernel):
        fmt = (
            'it {: >3d} of {:d}:'
            '  loss={: >+6.3f}'
            '  mean={: >+6.3f}'
            '  A={: >5.3f}'
            '  ell={: >5.3f}'
            '  noise={: >5.3f}'
        )

        def progress(i, n_iter, loss):
            print(model.mean_module.a_lm.tolist())
            txt = fmt.format(
                i+1, n_iter, loss.item(),
                model.mean_module.a_lm.tolist()[0],
                model.covar_module.outputscale.item(),
                model.covar_module.base_kernel.lengthscale.item(),
                model.likelihood.noise.item()
            )
            return txt

        return progress
    else:
        fmt = (
            'it {: >3d} of {:d}:'
            '  loss={: >+6.3f}'
            '  mean={: >+6.3f}'
            '  noise={: >5.3f}'
        )

        def progress(i, n_iter, loss):
            txt = fmt.format(
                i+1, n_iter, loss.item(),
                model.mean_module.constant.item(),
                model.likelihood.noise.item()
            )
            return txt

        return progress


def train_model(n_iter, data, model, likelihood):
    model.train()
    likelihood.train()

    print(model)

    optimizer = torch.optim.Adam([
        {'params': model.parameters()}
    ], lr=0.05)
    
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    print('Training:')
    progress = get_progress_fn(model)
    
    for i in range(n_iter):
        optimizer.zero_grad()
        output = model(data['train']['x'])
        loss = -mll(output, data['train']['y'])
        loss.backward()

        print(progress(i, n_iter, loss))
        
        optimizer.step()


def predict(coords, model, likelihood, sample=False):
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        m = model(coords)
        pred = likelihood(m)
        if sample:
            y = pred.sample().numpy()
        else:
            y = pred.mean.numpy()
    return y


def test_variance(data, model, likelihood, return_predictions=False):
    test_y_pred = predict(data['test']['x'], model, likelihood)
    delta_y = test_y_pred - data['test']['y'].numpy()
    var_delta_y = np.var(delta_y)

    if return_predictions:
        return var_delta_y, test_y_pred
    else:
        return var_delta_y


def proj_c2xy(c):
    xy = np.empty((c.shape[0], 2), dtype=c.dtype)
    xy[:,0] = c[:,0]
    xy[:,1] = c[:,1]
    return xy


def proj_xy2c(xy):
    c = np.empty((xy.shape[0], 3), dtype=xy.dtype)
    c[:,0] = xy[:,0]
    c[:,1] = xy[:,1]
    c[:,2] = np.sqrt(1. - xy[:,0]**2 - xy[:,1]**2)
    return c


def proj_stereo_c2xy(c):
    xy = np.empty((c.shape[0], 2), dtype=c.dtype)
    norm = 1. / (1. + c[:,2])
    xy[:,0] = c[:,0] * norm
    xy[:,1] = c[:,1] * norm
    return xy


def proj_stereo_xy2c(xy):
    c = np.empty((xy.shape[0], 3), dtype=xy.dtype)
    r2 = xy[:,0]**2 + xy[:,1]**2
    norm = 1. / (r2 + 1.)
    c[:,0] = 2 * xy[:,0] * norm
    c[:,1] = 2 * xy[:,1] * norm
    c[:,2] = (1. - r2) * norm
    return c


def proj_c2lonlat(c, center_lon=False, scale_lon=False):
    lon = np.degrees(np.arctan2(c[:,1], c[:,0]))
    lat = np.degrees(np.arcsin(c[:,2]))
    
    if center_lon:
        lon0 = 0.5 * (np.max(lon) + np.min(lon))
        lon -= lon0
        idx = (lon > 180.)
        if np.any(idx):
            lon[idx] -= 360.

    if scale_lon:
        lon *= np.cos(np.radians(lon))

    xy = np.empty((lon.size, 2), dtype=lon.dtype)
    xy[:,0] = lon
    xy[:,1] = lat

    return xy


def proj_lonlat2c(xy):
    lon = np.radians(xy[:,0])
    lat = np.radians(xy[:,1])
    c = np.empty((lat.size,3), dtype=lat.dtype)
    c[:,0] = np.cos(lon) * np.cos(lat)
    c[:,1] = np.sin(lon) * np.cos(lat)
    c[:,2] = np.sin(lat)
    return c


def get_coord_grid(data, proj, inv_proj, plot_size):
    # Project coordinates
    c_proj_data = proj(data['all']['x'])

    # Determine projection extent
    c_proj_min = np.min(c_proj_data, axis=0)
    c_proj_max = np.max(c_proj_data, axis=0)

    extent = [0 for i in range(4)]

    for i in range(2):
        w = c_proj_max[i] - c_proj_min[i]
        extent[2*i] = c_proj_min[i] - 0.1*w
        extent[2*i+1] = c_proj_max[i] + 0.1*w

    # Create grid of projected coordinates
    grid_x = np.linspace(extent[0], extent[1], plot_size, dtype='f4')
    grid_y = np.linspace(extent[2], extent[3], plot_size, dtype='f4')
    grid_proj = np.stack(np.meshgrid(grid_x, grid_y), axis=-1)
    grid_proj.shape = (plot_size*plot_size, 2)
    c_grid = torch.from_numpy(inv_proj(grid_proj))

    return extent, c_grid


def plot_results(fname, data, model, likelihood,
                 proj, inv_proj, plot_size=100):
    # Get grid of coordinates to plot
    extent, c_grid = get_coord_grid(data, proj, inv_proj, plot_size)

    # Project coordinates of training and test data
    c_proj_data = proj(data['all']['x'])
    c_proj_train = proj(data['train']['x'].numpy())
    c_proj_test = proj(data['test']['x'].numpy())

    # Evaluate GP mean
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        grid_pred = likelihood(model(c_grid))
        grid_pred_mean = grid_pred.mean.view(plot_size, plot_size).numpy()
        test_pred = likelihood(model(data['test']['x']))
        test_pred_mean = test_pred.mean.numpy()

    # Create figure
    fig = plt.figure(figsize=(18, 12), dpi=200)

    vmin, vmax = np.percentile(data['all']['y'], [2., 98.])
    gridsize = 25

    # Smooth GP prediction
    ax = fig.add_subplot(2,3,1)
    ax.set_title('Smooth Prediction')
    im = ax.imshow(
        grid_pred_mean,
        origin='lower',
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        aspect='auto'
    )
    fig.colorbar(im, ax=ax)
    ax.scatter(
        c_proj_train[:,0],
        c_proj_train[:,1],
        alpha=0.5,
        c='w',
        edgecolors='none',
        s=3
    )

    # True values
    ax = fig.add_subplot(2,3,2)
    ax.set_title('All data')
    im = ax.hexbin(
        c_proj_data[:,0], c_proj_data[:,1],
        C=data['all']['y'], reduce_C_function=np.median,
        gridsize=gridsize,
        vmin=vmin, vmax=vmax
    )
    fig.colorbar(im, ax=ax)

    # Kernel as function of distance
    #if isinstance(model.covar_module.base_kernel, RationalQuadraticKernel):
    #    ax = fig.add_subplot(2,3,3)
    #    ax.set_title('Kernel')
    #    r = np.radians(np.linspace(0., 10., 1000))
    #    A = model.covar_module.outputscale.item()
    #    ell = model.covar_module.base_kernel.lengthscale.item()
    #    alpha = model.covar_module.base_kernel.power_law.item()
    #    K_r = rational_quadratic(r, ell, alpha)
    #    label = r'$\ell = {:.3f}^{{\circ}}, \ \alpha = {:.3f}$'.format(
    #        np.degrees(ell),
    #        alpha
    #    )
    #    ax.semilogy(np.degrees(r), K_r, label=label)
    #    ax.set_xlabel(r'$r \ \left( \mathrm{deg} \right)$')
    #    ax.set_ylabel(r'$K \left( r \right) / K \left( 0 \right)$')
    #    ax.legend(loc='upper right')
    #    ax.grid(True)
    
    # Prediction at test points
    ax = fig.add_subplot(2,3,4)
    ax.set_title('Prediction at test points')
    im = ax.hexbin(
        c_proj_test[:,0], c_proj_test[:,1],
        C=test_pred_mean, reduce_C_function=np.mean,
        gridsize=gridsize,
        vmin=vmin, vmax=vmax
    )
    ax.set_xlim(extent[:2])
    ax.set_ylim(extent[2:])
    fig.colorbar(im, ax=ax)

    # True values at test points
    ax = fig.add_subplot(2,3,5)
    ax.set_title('True values at test points')
    im = ax.hexbin(
        c_proj_test[:,0], c_proj_test[:,1],
        C=data['test']['y'], reduce_C_function=np.mean,
        gridsize=gridsize,
        vmin=vmin, vmax=vmax
    )
    ax.set_xlim(extent[:2])
    ax.set_ylim(extent[2:])
    fig.colorbar(im, ax=ax)

    # Residuals at test points
    ax = fig.add_subplot(2,3,6)
    ax.set_title('Residuals at test points')
    resid = test_pred_mean - data['test']['y'].numpy()
    vmax = np.percentile(np.abs(resid), 95.)
    im = ax.hexbin(
        c_proj_test[:,0], c_proj_test[:,1],
        C=resid, reduce_C_function=np.mean,
        gridsize=gridsize,
        vmin=-vmax, vmax=vmax, cmap='coolwarm_r'
    )
    ax.set_xlim(extent[:2])
    ax.set_ylim(extent[2:])
    fig.colorbar(im, ax=ax)

    fig.savefig(fname, dpi=200)
    plt.close(fig)


def save_model(model, fname):
    import gpytorch_serializer
    encoder = gpytorch_serializer.get_encoder(ndarray_mode='readable')

    d = model.state_dict()
    txt = json.dumps(d, indent=2, sort_keys=True, cls=encoder)

    with open(fname, 'w') as f:
        f.write(txt)


def load_model(model, fname):
    import gpytorch_serializer
    decoder = gpytorch_serializer.MultiJSONDecoder

    with open(fname, 'r') as f:
        txt = f.read()

    print('model state:')
    print(txt)

    d = json.loads(txt, cls=decoder)
    model.load_state_dict(d)

    #for param in model.named_parameters():
    #    print(param)
    #    print(type(param))


def rational_quadratic(r, l, a):
    return (1. + r**2/(2.*a*l**2.))**(-a)


def plot_kernel_grid(fname, data, model, likelihood,
                     proj, inv_proj,
                     plot_size=100,
                     plot_training=True):
    # Get grid of coordinates to plot
    extent, c_grid = get_coord_grid(data, proj, inv_proj, plot_size)

    # Project coordinates of training data
    c_proj_train = proj(data['train']['x'].numpy())

    # Create figures
    figs = {s: plt.figure(figsize=(16,16), dpi=200) for s in [True,False]}

    vmin, vmax = np.percentile(data['all']['y'], [2., 98.])
    a_lm = np.zeros(16)
    a_lm[0] = np.median(data['all']['y'])

    for col,alpha in enumerate([0.1, 1., 3., 10.]):
        for row,ell in enumerate([0.1, 1., 3., 10.]):
            print('Plotting kernel for (alpha, ell) = ({:.1f}, {:.1f})'.format(
                alpha, ell
            ))

            # Update kernel parameters
            model.set_parameters(
                alpha=alpha,
                ell=np.radians(ell),
                a_lm=a_lm,
                A=0.005
            )
            model.eval()
            likelihood.eval()

            # Evaluate GP at grid points
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                grid_pred = likelihood(model(c_grid))

            for sample in [True, False]:
                # Either sample or take mean of distribution
                if sample:
                    grid_pred_value = grid_pred.sample()
                else:
                    grid_pred_value = grid_pred.mean

                grid_pred_value = grid_pred_value.view(
                    plot_size, plot_size
                ).numpy()

                fig = figs[sample]

                # Smooth GP prediction
                ax = fig.add_subplot(4,4,4*row+col+1, aspect='equal')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                #ax.axis('off')

                if row == 0:
                    ax.set_title(r'$\alpha = {}$'.format(alpha))
                if col == 0:
                    ax.set_ylabel(r'$\ell = {}^{{\circ}}$'.format(ell))

                im = ax.imshow(
                    grid_pred_value,
                    origin='lower',
                    extent=extent,
                    vmin=vmin,
                    vmax=vmax,
                    aspect='auto'
                )

                if plot_training:
                    ax.scatter(
                        c_proj_train[:,0],
                        c_proj_train[:,1],
                        alpha=0.5,
                        c='w',
                        edgecolors='none',
                        s=2
                    )

    for sample in [True, False]:
        fig = figs[sample]
        fig.subplots_adjust(
            left=0.06, right=0.98,
            bottom=0.02, top=0.94,
            hspace=0.02, wspace=0.02
        )
        suffix = 'sample' if sample else 'mean'
        fig.savefig(fname.format(suffix), dpi=200)
        plt.close(fig)


def main():
    # Load data
    fname = 'stellar_params_highlat_thinned.h5'

    data = load_sources(fname)
    data = filter_sources(data, E_min=-0.30, sigma_E_max=0.10, d_min=500.)
    data = extract_train_test_data(data, 2000, 2000, log=False)

    # Set up model
    model, likelihood = get_model(data)
    fname = 'model.json'
    
    # Plot grid of predictions with different kernel parameters
    plot_kernel_grid(
        'kernel_grid_{}.png',
        data, model, likelihood,
        proj_stereo_c2xy, proj_stereo_xy2c,
        plot_size=150,
    )

    # Load the model
    #print('Loading model from {:s}'.format(fname))
    #load_model(model, fname)

    # Train the model
    train_model(250, data, model, likelihood)

    # Save the model
    print('Saving model to {:s}'.format(fname))
    save_model(model, fname)

    # Test the model
    test_y_var = test_variance(data, model, likelihood)
    raw_y_var = np.var(data['test']['y'].numpy())
    print('Test variance: {:.3f} (raw)'.format(raw_y_var))
    print('               {:.3f} (pred)'.format(test_y_var))

    # Plot the results
    fname = 'results1.png'
    plot_results(
        fname, data, model, likelihood,
        proj_stereo_c2xy, proj_stereo_xy2c,
        #proj_c2xy, proj_xy2c,
        #proj_c2lonlat, proj_lonlat2c,
        plot_size=300
    )

    return 0


if __name__ == '__main__':
    main()
