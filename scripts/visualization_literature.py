#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generates figures examining reproduced pipelines
"""

from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from surfer import Brain

import abagen
from vibecheck.plotting import savefig

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.sans-serif'] = ['Myriad Pro']
plt.rcParams['font.size'] = 20.0

GENE = 'SST'
FIG_DIR = Path('./figures').resolve()
DATA_DIR = Path('./data/derivatives/literature').resolve()
LIT_COLORS = dict(zip([
    'hawrylycz2015', 'french2015', 'whitakervertes2016', 'krienen2016',
    'anderson2018', 'burt2018', 'romerogarcia2018', 'anderson2020', 'liu2020'
], sns.color_palette('twilight', n_colors=9)))
RENAME = dict(
    gene_norm='gene normalization',
    lr_mirror='mirroring samples',
    norm_matched='normalize matched',
    agg_metric='sample-to-region metric',
    norm_structures='normalize structures',
    tolerance='sample-to-region distance',
    region_agg='sample-to-region method',
    probe_selection='probe selection',
    surface='atlas type',
    sample_norm='sample normalization',
    ibf_threshold='ibf threshold',
    reannotated='updated annotations',
    corrected_mni='updated mni',
    missing='missing data',
    donor_probes='donor probe selection',
    native='individualized or group',
    sim_threshold='similarity threshold'
)
PARAMETERS = pd.DataFrame.from_dict({
    "hawrylycz2015": {
        "surface": True,
        "native": True,
        "reannotated": False,
        "corrected_mni": False,
        "tolerance": 0,
        "sample_norm": None,
        "gene_norm": "zscore",
        "missing": "centroids",
        "probe_selection": "diff_stability",
        "donor_probes": "aggregate",
        "region_agg": "donors",
        "agg_metric": "mean",
        "lr_mirror": None,
        "ibf_threshold": 0,
        "norm_matched": True,
        "norm_structures": False,
        "sim_threshold": 0
    },
    "french2015": {
        "surface": False,
        "native": False,
        "reannotated": False,
        "corrected_mni": False,
        "tolerance": 1,
        "sample_norm": None,
        "gene_norm": None,
        "missing": None,
        "probe_selection": "average",
        "donor_probes": "aggregate",
        "region_agg": "donors",
        "agg_metric": "median",
        "lr_mirror": None,
        "ibf_threshold": 0,
        "norm_matched": True,
        "norm_structures": False,
        "sim_threshold": 0
    },
    "whitakervertes2016": {
        "surface": False,
        "native": False,
        "reannotated": False,
        "corrected_mni": False,
        "tolerance": 0,
        "sample_norm": None,
        "gene_norm": "zscore",
        "missing": "centroids",
        "probe_selection": "average",
        "donor_probes": "aggregate",
        "region_agg": "donors",
        "agg_metric": "mean",
        "lr_mirror": None,
        "ibf_threshold": 0,
        "norm_matched": True,
        "norm_structures": False,
        "sim_threshold": 0
    },
    "krienen2016": {
        "surface": False,
        "native": False,
        "reannotated": False,
        "corrected_mni": False,
        "tolerance": 0,
        "sample_norm": None,
        "gene_norm": "center",
        "missing": None,
        "probe_selection": "average",
        "donor_probes": "aggregate",
        "region_agg": "donors",
        "agg_metric": "mean",
        "lr_mirror": None,
        "ibf_threshold": 0,
        "norm_matched": True,
        "norm_structures": False,
        "sim_threshold": 0
    },
    "anderson2018": {
        "surface": False,
        "native": False,
        "reannotated": False,
        "corrected_mni": False,
        "tolerance": 0,
        "sample_norm": None,
        "gene_norm": "center",
        "missing": None,
        "probe_selection": "corr_intensity",
        "donor_probes": "aggregate",
        "region_agg": "donors",
        "agg_metric": "mean",
        "lr_mirror": None,
        "ibf_threshold": 0,
        "norm_matched": True,
        "norm_structures": True,
        "sim_threshold": 0
    },
    "burt2018": {
        "surface": True,
        "native": True,
        "reannotated": False,
        "corrected_mni": False,
        "tolerance": 2,
        "sample_norm": "zscore",
        "gene_norm": "zscore",
        "missing": "interpolate",
        "probe_selection": "corr_variance",
        "donor_probes": "independent",
        "region_agg": "donors",
        "agg_metric": "mean",
        "lr_mirror": None,
        "ibf_threshold": 0,
        "norm_matched": True,
        "norm_structures": False,
        "sim_threshold": 5
    },
    "romerogarcia2018": {
        "surface": True,
        "native": True,
        "reannotated": True,
        "corrected_mni": False,
        "tolerance": 0,
        "sample_norm": None,
        "gene_norm": "zscore",
        "missing": "interpolate",
        "probe_selection": "max_intensity",
        "donor_probes": "aggregate",
        "region_agg": "samples",
        "agg_metric": "median",
        "lr_mirror": "rightleft",
        "ibf_threshold": 0,
        "norm_matched": True,
        "norm_structures": False,
        "sim_threshold": 0
    },
    "anderson2020": {
        "surface": True,
        "native": True,
        "reannotated": False,
        "corrected_mni": False,
        "tolerance": -4,
        "sample_norm": "zscore",
        "gene_norm": "zscore",
        "missing": None,
        "probe_selection": "max_intensity",
        "donor_probes": "aggregate",
        "region_agg": "donors",
        "agg_metric": "mean",
        "lr_mirror": None,
        "ibf_threshold": 0.2,
        "norm_matched": True,
        "norm_structures": False,
        "sim_threshold": 0
    },
    "liu2020": {
        "surface": False,
        "native": False,
        "reannotated": True,
        "corrected_mni": False,
        "tolerance": 5,
        "sample_norm": "zscore",
        "gene_norm": "zscore",
        "missing": None,
        "probe_selection": "average",
        "donor_probes": "aggregate",
        "region_agg": "donors",
        "agg_metric": "mean",
        "lr_mirror": None,
        "ibf_threshold": 0.5,
        "norm_matched": True,
        "norm_structures": False,
        "sim_threshold": 0
    }
}, orient='index')


def param_choices(parameters):
    """
    Generates heatmap showing variable parameter choices across manuscripts

    Parameters
    ----------
    parameters : pd.DataFrame
        DataFrame where columns are parameters and rows are manuscripts

    """
    # cast to categorical + then get codes
    for col in parameters.columns:
        parameters[col] = parameters[col].astype('category').cat.codes
    parameters = parameters[list(RENAME.keys())]
    parameters = parameters.rename(RENAME, axis=1).T

    # heatmap
    n_colors = np.unique(parameters).size
    cmap = sns.color_palette('rocket', n_colors)
    ax = sns.heatmap(parameters, square=True, cbar=True, cmap=cmap,
                     yticklabels=parameters.index)
    ax.tick_params(width=0)
    cbar = ax.collections[0].colorbar
    r = cbar.vmax - cbar.vmin
    cbar.set_ticks([
        cbar.vmin + r / n_colors * (0.5 + i)
        for i in range(n_colors)
    ])
    cbar.set_ticklabels(range(n_colors))
    cbar.ax.tick_params(width=0)
    cbar.set_label('parameter choice')

    savefig(ax.figure, FIG_DIR / 'literature_parameters.svg')


def parc_brainmaps(gene):
    """
    Generates brainplots of `gene` expression values for each manuscript

    Parameters
    ----------
    gene : str
        Gene for which brainplots should be generated
    """

    fig_dir = FIG_DIR / 'brainmaps'
    fig_dir.mkdir(parents=True, exist_ok=True)

    aparc = nib.freesurfer.read_annot(
        '/opt/freesurfer/subjects/fsaverage/label/lh.aparc.annot'
    )[0]
    aparc[aparc == -1] = 0
    aparc = abagen.images._relabel(aparc)
    ids = np.unique(aparc)
    for study, params in PARAMETERS.iterrows():
        data = pd.read_hdf(DATA_DIR / f'{study}.h5')
        data.loc[0] = -100
        data = np.asarray(data.loc[ids, gene])
        # mean, std = data[1:].mean(), data[1:].std(ddof=1)
        # vmin, vmax = mean - (2 * std), mean + (2 * std)
        brain = Brain('fsaverage', 'lh', 'inflated', background='white',
                      size=1000, offscreen=True)
        brain.add_data(data[aparc], thresh=-99,
                       colorbar=False, colormap='BuPu',
                       min=data[1:].min(), max=data[1:].max())
        brain.save_image(fig_dir / f'{gene.lower()}_{study}.png')
        brain.close()


def make_lit_barplot(literature):
    """
    Makes barplot of statistical estimates in `literature`

    Parameters
    ----------
    literature : pd.Dataframe
        Dataframe with statistical estimates from different manuscripts
    """
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    ests = ['dist_spearmanr', 'silhouette', 't1t2_spearmanr']
    xlim = [(-0.50, 0), (-0.8, 0), (0, 1.00)]
    literature['filename'] = literature['filename'].apply(
        lambda x: x[:-3] if x.endswith('.h5') else x
    )
    for n, (ax, param) in enumerate(zip(axes, ests)):
        ax = sns.barplot(x=param, y='filename', data=literature, ax=ax,
                         palette=LIT_COLORS, order=list(LIT_COLORS.keys()))
        ax.set(xlim=xlim[n], xticks=np.linspace(*xlim[n], 3),
               xticklabels=[xlim[n][0], None, xlim[n][1]])
        ax.tick_params('y', width=0)
        if n > 0:
            ax.set(yticks=[], ylabel=None)
        sns.despine(ax=ax)

    savefig(fig, FIG_DIR / 'literature_estimates.svg')


def make_gene_heatmap(gene):
    """
    Generate heatmap of correlations for `gene` expression between manuscripts

    Parameters
    ----------
    gene : str
        Gene for which correlations should be computed
    """
    fig_dir = FIG_DIR / 'brainmaps'
    fig_dir.mkdir(parents=True, exist_ok=True)

    data = []
    for study in PARAMETERS.index:
        data.append(np.asarray(pd.read_hdf(DATA_DIR / f'{study}.h5')[gene]))
    data = np.row_stack(data)
    mask = np.logical_not(np.isnan(data).any(axis=0))
    heatmap = np.corrcoef(data[:, mask])

    fig, ax = plt.subplots(1, 1)
    sns.heatmap(heatmap, ax=ax, vmin=0.75, vmax=1.00,
                xticklabels=PARAMETERS.index, yticklabels=PARAMETERS.index)

    savefig(fig, FIG_DIR / f'literature_{gene}_corrmap.svg')


if __name__ == "__main__":
    literature = pd.read_csv(DATA_DIR / 'analysis.csv')
    make_lit_barplot(literature)
    param_choices(PARAMETERS.copy())
    parc_brainmaps(GENE)
    make_gene_heatmap(GENE)
