# -*- coding: utf-8 -*-
"""
Functions for analyzing abagen expression data
"""

import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import distance_matrix
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import silhouette_score

import abagen
from . import abautils

REPO_DIR = Path(os.path.abspath(__file__)).resolve().parent.parent.parent
DATA_DIR = REPO_DIR / 'data' / 'raw'


def get_unique_genes(reannotated=False):
    """
    Returns unique genes from the AHBA with valid Entrez IDs

    Parameters
    ----------
    reannotated : bool, optional
        Whether to return reannotated genes. Default: False

    Returns
    -------
    genes : pd.Series
        Gene symbols in the AHBA
    """

    probes = abagen.utils.first_entry(abagen.fetch_microarray(), 'probes')
    genes = abagen.io.read_probes(probes)
    if reannotated:
        genes = abagen.probes_.reannotate_probes(genes)
    genes = genes.dropna(subset=['entrez_id']) \
                 .get('gene_symbol') \
                 .unique()

    return genes


def correlate_distance(data, atlas):
    """
    Calculates distance-dependence of gene expression `data` from `atlas`

    Parameters
    ----------
    data : (R, G) pandas.DataFrame
        Regional gene expression data for `R` regions and `G` genes, as
        obtained by :func:`abagen.get_expression_data`
    atlas : niimg-like, optional
        Filepath or pre-loaded image of atlas used in generating `data`. Only
        used if `coords` is not specified. If not specified uses the output
        of :func:`get_dk_nosubcortex`. Default: None

    Returns
    -------
    pearson : float
        Pearson correlation of correlated gene expression w/regional distance
    spearman : float
        Spearman correlation of correlated gene expression w/regional distance
    """

    genes = abautils.get_unique_genes()

    if isinstance(data, (str, os.PathLike)):
        data = pd.read_hdf(data, index_col=0)

    data = data[np.intersect1d(genes, data.columns)]
    coords = np.row_stack(list(atlas.centroids.values()))

    triu = np.triu_indices(len(coords), k=1)
    data = np.corrcoef(data)[triu]
    dist = distance_matrix(coords, coords)[triu]

    # drop NaNs before correlation
    keep = np.logical_not(np.logical_or(np.isnan(data), np.isnan(dist)))
    data, dist = data[keep], dist[keep]
    pearson, spearman = pearsonr(data, dist)[0], spearmanr(data, dist)[0]

    return pearson, spearman


def parcellate_t1t2(data_dir, roi_dir):
    """
    Parcellates group-averaged T1w/T2w map from Human Connectome Project

    Parameters
    ----------
    data_dir : pathlike, optional
        Data directory where T1w/T2w .func.gii images are stored
    roi_dir : str or os.PathLike

    Returns
    -------
    data : numpy.ndarray
        Average t1w/t2w ratio in each parcel of provided `atlas`
    """

    data_dir, roi_dir = Path(data_dir), Path(roi_dir)
    atlas = 'atlas-desikankilliany_space-fsLR_den-32k_hemi-{hemi}_parc.label'
    t1t2path = 'S1200.{hemi}.MyelinMap_BC_MSMAll.32k_fs_LR.func.gii'
    out = []
    for hemi in ('L', 'R'):
        atlas = nib.load(roi_dir / atlas.format(hemi=hemi)).agg_data()
        t1t2 = nib.load(data_dir / t1t2path.format(hemi=hemi)).agg_data()
        data = ndimage.mean(t1t2, labels=atlas, index=np.unique(atlas)[1:])
        out.append(data)

    out = np.hstack(out).squeeze()

    return out


def correlate_t1t2(data, t1t2):
    """
    Parameters
    ----------
    data : (R, G) pandas.DataFrame
        Regional gene expression data for `R` regions and `G` genes, as
        obtained by :func:`abagen.get_expression_data`
    t1t2 : (R,) numpy.ndarray
        Parcellated t1w/t2w ratio data

    Returns
    -------
    pearson : float
        Pearson correlation of PC1 of the gene expression `data` with `t1t2`
    spearman : float
        Spearman correlation of PC1 of the gene expression `data` with `t1t2`
    """

    genes = abautils.get_unique_genes()

    if isinstance(data, (str, os.PathLike)):
        data = pd.read_hdf(data, index_col=0)

    data = data[np.intersect1d(genes, data.columns)]

    # drop NaNs before SVD
    data = np.asarray(data)
    keep = np.logical_not(np.any(np.isnan(data), axis=1))
    data, t1t2 = data[keep], t1t2[keep]

    # get principal eigengene from data
    data -= data.mean(axis=0, keepdims=True)
    pc1 = np.linalg.svd(data, full_matrices=False)[0][:, 0]

    # run correlations
    pearson, spearman = pearsonr(pc1, t1t2)[0], spearmanr(pc1, t1t2)[0]

    # the sign doesn't matter, let's just make these all positive
    pearson *= np.sign(pearson)
    spearman *= np.sign(spearman)

    return pearson, spearman


def load_oldham2008_modules(all_modules=True):
    """
    Returns gene modules from [1]_

    Parameters
    ----------
    all_modules : bool, optional
        Whether to retain all modules from the original data instead of
        dropping those with limited overlap (i.e., <=2 genes per module) to the
        AHBA (when considering the highest IBF threshold). Default: True

    Returns
    -------
    modules : dict
        Dictionary where keys are module (colors) and values are lists of gene
        acronyms

    References
    ----------
    .. [1] Oldham, M. C., Konopka, G., Iwamoto, K., Langfelder, P., Kato, T.,
       Horvath, S., & Geschwind, D. H. (2008). Functional organization of the
       transcriptome in human brain. Nature Neuroscience, 11(11), 1271.
    """

    # these are modules for which we only have a limited overlap (i.e., <=2
    # genes per module) with AHBA gene list when considering the highest
    # intensity-based filtering threshold
    drop_mod = [
        'darkolivegreen', 'palegreen', 'powderblue', 'purple', 'salmon', 'tan',
        'tomato'
    ]

    # load data from supplementary table, keeping only acronym + module columns
    data = pd.read_excel(DATA_DIR / 'gene_modules' / 'oldham2008.xls',
                         usecols=['Gene', 'Module'])

    # get rid of genes that weren't assigned to a module (NaN / grey)
    # also remove the weird "---" genes
    data = data.dropna(subset=['Module']) \
               .query('Module != "grey" & Gene != "---"')

    # some genes are really lists (delimited by ' /// '), so we need to split
    # those lists and then explode the resulting dataframe
    data['Gene'] = data['Gene'].apply(lambda x: x.split(' /// '))
    data = data.explode('Gene').reset_index(drop=True)

    # finally, convert the dataframe into a dictionary where modules are keys
    # and gene lists are the corresponding values
    modules = data.groupby('Module')['Gene'].apply(list).to_dict()

    if not all_modules:
        for m in drop_mod:
            del modules[m]

    return modules


def gene_silhouette(data, modules=None, all_modules=True):
    """
    Calculates gene module modularity from expression `data` and gene `modules`

    Parameters
    ----------
    data : (R, G) pandas.DataFrame
        Regional gene expression data for `R` regions and `G` genes, as
        obtained by :func:`abagen.get_expression_data`
    modules : dict, optional
        Dictionary where keys are modules and values are lists of genes
        belonging to each module. If not specified will use modules returned by
        :func:`load_oldham2008_modules`

    Returns
    -------
    modularity : float
        Modularity statistic
    """

    genes = abautils.get_unique_genes()

    if isinstance(data, (str, os.PathLike)):
        data = pd.read_hdf(data, index_col=0)

    data = data[np.intersect1d(genes, data.columns)]

    # maybe we didn't get modules. use the oldham ones
    if modules is None:
        modules = load_oldham2008_modules(all_modules)

    data = data.dropna(axis=0, how='all')

    # convert modules into community vector
    comms = np.sum([n * (np.isin(data.columns, genes))
                    for n, genes in enumerate(modules.values(), 1)], axis=0)

    keep = comms != 0

    return silhouette_score(data.T[keep], comms[keep], random_state=1234)
