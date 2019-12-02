#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generates figures depicting pipeline distributions + parameter impact
"""

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.utils.extmath import randomized_svd

from vibecheck.plotting import savefig

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.sans-serif'] = ['Myriad Pro']
plt.rcParams['font.size'] = 20.0

DATA_DIR = Path('./data/derivatives').resolve()
FIG_DIR = Path('./figures').resolve()
GENENORM_ORDER = ['None', 'srs', 'zscore']
NORMSTRU_ORDER = [False, True]
DIST_PARAMS = (
    ('dist_spearmanr', 'gene_norm', GENENORM_ORDER, dict(
        xticks=(-0.6, -0.4, -0.2, 0), yticks=(0, 2.5, 5),
        xticklabels=(-0.6, None, None, 0),
        yticklabels=(0, None, 5),
        xlim=(-0.6, 0), ylim=[None, 5.0],
        xlabel='correlation (r)', ylabel=''
    )),
    ('silhouette', 'gene_norm', GENENORM_ORDER, dict(
        xticks=(-0.9, -0.7, -0.5, -0.3, -0.1), yticks=(0, 3.5, 7),
        xticklabels=(-0.9, None, None, None, -0.1),
        yticklabels=(0, None, 7),
        xlim=(-0.9, -0.1), ylim=[None, 7.0],
        xlabel='silhouette score', ylabel=''
    )),
    ('t1t2_spearmanr', 'gene_norm', GENENORM_ORDER, dict(
        xticks=(-0.2, 0.1, 0.4, 0.7, 1.0), yticks=(0, 4, 8, 12, 16),
        xticklabels=(-0.2, None, None, None, 1.0),
        yticklabels=(0, None, None, None, 16),
        xlim=(-0.2, 1.0), ylim=[None, 16],
        xlabel='correlation (rho)', ylabel=''
    ))
)
PARAMETERS = dict(
    ibf_threshold='ibf threshold',
    probe_selection='probe selection',
    donor_probes='donor probe selection',
    lr_mirror='mirroring samples',
    missing='missing data',
    tolerance='sample-to-region distance',
    sample_norm='sample normalization',
    gene_norm='gene normalization',
    norm_matched='normalize matched',
    norm_structures='normalize structures',
    region_agg='sample-to-region method',
    agg_metric='sample-to-region metric',
    corrected_mni='updated mni',
    reannotated='updated annotations',
    atlas_name='atlas type'
)
PALETTE = (np.row_stack([(35, 95, 192),
                         (205, 78, 78),
                         (123, 30, 153)]) / 255).tolist()


def make_hexplot(x, y, parameter, fname=None, vmin=None, vmax=None, cmaps=None,
                 order=None, **kwargs):
    if vmin is not None or vmax is not None:
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = None
    fig, ax = plt.subplots(1, 1)
    if order is None:
        order = parameter.astype('category').cat.categories
    if cmaps is None:
        cmaps = ['Blues', 'Reds', 'Purples']
    cbar = None
    for n, code in enumerate(order):
        mask = np.asarray(parameter == code)
        opts = {'norm': norm} if norm is not None else {'bins': 'log'}
        opts.update(kwargs)
        coll = ax.hexbin(x[mask], y[mask], cmap=cmaps[n],
                         alpha=1.0, mincnt=1, linewidth=0.2, **opts)
        if cbar is None:
            cbar = fig.colorbar(coll, shrink=0.83, label='density')
            cbar.ax.collections[0].set_cmap('gray_r')
            if norm is None:
                norm = mcolors.LogNorm(vmin=cbar.vmin, vmax=cbar.vmax)
    ax.set(xticks=(-0.4, 0, 0.4, 0.8),
           xticklabels=(-0.4, None, None, 0.8),
           xlim=(-0.4, 0.9), xlabel='pc1 scores',
           yticks=(-0.3, 0, 0.3, 0.6),
           yticklabels=(-0.3, None, None, 0.6),
           ylim=(-0.4, 0.7), ylabel='pc2 scores')
    sns.despine(ax=ax, trim=True)
    fig.tight_layout()
    if fname is not None:
        if not fname.endswith('.svg'):
            fname += '.svg'
        savefig(fig, FIG_DIR / fname, dpi=1000)
    else:
        return fig


def make_displots(pipelines):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    rgb = np.array([[154, 27, 91]]) / 255
    ylim = ((0, 10), (0, 5), (0, 20))
    for n, params in enumerate(DIST_PARAMS):
        est, kwargs = params[0], params[-1].copy()
        del kwargs['yticks'], kwargs['yticklabels'], kwargs['ylim']
        ax = sns.kdeplot(x=est, data=pipelines, color=rgb, fill=True,
                         ax=axes[n])
        ax.set(**kwargs, ylim=ylim[n], yticks=np.linspace(*ylim[n], 6))
        sns.despine(ax=ax, offset=5, trim=True)

    savefig(fig, FIG_DIR / 'full_distributions.svg')


def make_colorbar_plot(cmap):
    fig, ax = plt.subplots(1, 1, figsize=(9, 1.5))
    a = np.array([[0, 1]])
    img = ax.imshow(a, cmap=cmap)
    ax.set_visible(False)
    cax = fig.add_axes([0.1, 0.2, 0.8, 0.6])
    cbar = fig.colorbar(img, orientation="horizontal", cax=cax)
    cbar.set_ticks([])
    savefig(fig, FIG_DIR / f'colorbar_{cmap}.svg')


def make_impact_plot(ranks):
    n_colors = len(ranks)
    cmap = sns.color_palette('rocket_r', n_colors=n_colors)
    fig, ax = plt.subplots(1, 1)
    ax = sns.heatmap(ranks, square=True, cmap=cmap,
                     yticklabels=[PARAMETERS.get(p) for p in ranks.index],
                     xticklabels=['CGE', 'GCE', 'RGE'], ax=ax)
    ax.set(xlabel='', ylabel='')
    ax.tick_params(width=0)
    colorbar = ax.collections[0].colorbar
    r = colorbar.vmax - colorbar.vmin
    colorbar.set_ticks([
        colorbar.vmin + 0.5 * r / (n_colors) + r * i / (n_colors)
        for i in range(0, n_colors, 7)
    ])
    colorbar.set_ticklabels((1, 8, 15))
    colorbar.set_label('impact rank')
    colorbar.ax.tick_params(width=0)
    colorbar.ax.invert_yaxis()
    savefig(fig, FIG_DIR / 'impact.svg')


def pca(data):
    data = np.asarray(data)
    data -= data.mean(axis=0, keepdims=True)
    u, s, v = randomized_svd(data, 2, random_state=1234)
    v = v.T
    scores = data @ v
    return scores


if __name__ == "__main__":
    pipelines = pd.read_csv(DATA_DIR / 'pipelines.csv.gz') \
                  .replace({np.nan: "None"})
    impact = pd.read_csv(DATA_DIR / 'impact.csv', index_col=0)
    twilight = np.asarray(sns.color_palette('twilight', n_colors=9))

    # make distributional plots showing impact of pipelines
    make_displots(pipelines)

    # now split distributions based on selected parameter
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for n, (key, hue, hue_order, params) in enumerate(DIST_PARAMS):
        ax = sns.kdeplot(x=key, hue=hue, data=pipelines, fill=True,
                         palette=dict(zip(hue_order, twilight[[3, 1, 7]])),
                         hue_order=hue_order, ax=axes[n], legend=False)
        ax.set(**params)
        sns.despine(ax=ax, trim=True, offset=5)
    savefig(fig, FIG_DIR / 'distplot_genenorm.svg')

    # make plot showing relative parameter importance
    order = impact.rank().mean(axis=1).sort_values(ascending=False).index
    ranks = impact.loc[order].rank(ascending=False)
    make_impact_plot(ranks)

    # make plot of PCs of pipelines
    scores = pca(pipelines[['dist_spearmanr', 'silhouette', 't1t2_spearmanr']])
    gn = [sns.light_palette(f, as_cmap=True) for f in twilight[[3, 1, 7]]]
    make_hexplot(*scores.T, pipelines['gene_norm'], 'hex_genenorm',
                 vmin=1, vmax=5000, rasterized=True, cmaps=gn,
                 order=GENENORM_ORDER)
    ns = [sns.light_palette(f, as_cmap=True) for f in twilight[[2, 6]]]
    make_hexplot(*scores.T, pipelines['norm_structures'], 'hex_normstruct',
                 vmin=1, vmax=5000, rasterized=True, cmaps=ns,
                 order=NORMSTRU_ORDER)

    make_colorbar_plot('BuPu')
