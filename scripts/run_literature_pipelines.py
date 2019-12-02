#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reproduces (or attempts to reproduce) processing pipelines for nine previously
published articles using the AHBA
"""

from pathlib import Path

import numpy as np
import pandas as pd

import abagen
from vibecheck import abautils, analysis

RAW_DIR = Path('./data/raw').resolve()
DATA_DIR = Path('./data/derivatives/literature').resolve()
SURFACE = {
    'anderson2018': False,
    'anderson2020': True,
    'burt2018': True,
    'french2015': False,
    'hawrylycz2015': True,
    'krienen2016': False,
    'liu2020': False,
    'romerogarcia2018': True,
    'whitakervertes2016': False,
}


def main():
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    # Hawrylycz et al., 2015, Nature Neuroscience
    atlas = abagen.fetch_desikan_killiany(surface=True, native=True)
    expression = abagen.get_expression_data(atlas['image'],
                                            reannotated=False,
                                            corrected_mni=False,
                                            tolerance=0,
                                            sample_norm=None,
                                            gene_norm='zscore',
                                            missing='centroids',
                                            probe_selection='diff_stability',
                                            donor_probes='aggregate',
                                            region_agg='donors',
                                            agg_metric='mean',
                                            lr_mirror=None,
                                            ibf_threshold=0,
                                            norm_matched=True,
                                            norm_structures=False,
                                            sim_threshold=None)
    expression.to_hdf(DATA_DIR / 'hawrylycz2015.h5', 'data',
                      complib='zlib', complevel=9)

    # French et al., 2015, Front Neurosci
    atlas, atlas_info = abautils.get_dknosub(surface=False, return_info=True)
    expression = abagen.get_expression_data(atlas, atlas_info,
                                            reannotated=False,
                                            corrected_mni=False,
                                            tolerance=1,
                                            sample_norm=None,
                                            gene_norm=None,
                                            missing=None,
                                            probe_selection='average',
                                            donor_probes='aggregate',
                                            region_agg='donors',
                                            agg_metric='median',
                                            lr_mirror=None,
                                            ibf_threshold=0,
                                            norm_matched=True,
                                            norm_structures=False,
                                            sim_threshold=None)
    expression.to_hdf(DATA_DIR / 'french2015.h5', 'data',
                      complib='zlib', complevel=9)

    # Whitaker, Vertes et al., 2016, PNAS
    tree = abautils.get_dknosub(surface=False)
    ids, coords = list(tree.centroids.keys()), list(tree.centroids.values())
    tree = abagen.AtlasTree(np.asarray(ids), np.asarray(coords))
    expression = abagen.get_expression_data(tree,
                                            reannotated=False,
                                            corrected_mni=False,
                                            tolerance=0,
                                            sample_norm=None,
                                            gene_norm='zscore',
                                            missing='centroids',
                                            probe_selection='average',
                                            donor_probes='aggregate',
                                            region_agg='donors',
                                            agg_metric='mean',
                                            lr_mirror=None,
                                            ibf_threshold=0,
                                            norm_matched=True,
                                            norm_structures=False,
                                            sim_threshold=None)
    expression.to_hdf(DATA_DIR / 'whitakervertes2016.h5', 'data',
                      complib='zlib', complevel=9)

    # Krienen et al.,2016, PNAS
    atlas, atlas_info = abautils.get_dknosub(surface=False, return_info=True)
    expression = abagen.get_expression_data(atlas, atlas_info,
                                            reannotated=False,
                                            corrected_mni=False,
                                            tolerance=0,
                                            sample_norm=None,
                                            gene_norm='center',
                                            missing=None,
                                            probe_selection='average',
                                            donor_probes='aggregate',
                                            region_agg='donors',
                                            agg_metric='mean',
                                            lr_mirror=None,
                                            ibf_threshold=0,
                                            norm_matched=True,
                                            norm_structures=False,
                                            sim_threshold=None)
    expression.to_hdf(DATA_DIR / 'krienen2016.h5', 'data',
                      complib='zlib', complevel=9)

    # Anderson et al., 2018, Nature Communications
    atlas, atlas_info = abautils.get_dknosub(surface=False, return_info=True)
    expression = abagen.get_expression_data(atlas, atlas_info,
                                            reannotated=False,
                                            corrected_mni=False,
                                            tolerance=0,
                                            sample_norm=None,
                                            gene_norm='center',
                                            missing=None,
                                            probe_selection='corr_intensity',
                                            donor_probes='aggregate',
                                            region_agg='donors',
                                            agg_metric='mean',
                                            lr_mirror=None,
                                            ibf_threshold=0,
                                            norm_matched=True,
                                            norm_structures=True,
                                            sim_threshold=None)
    expression.to_hdf(DATA_DIR / 'anderson2018.h5', 'data',
                      complib='zlib', complevel=9)

    # Burt et al., 2018, Nature Neuroscience
    atlas = abagen.fetch_desikan_killiany(surface=True, native=True)
    expression = abagen.get_expression_data(atlas['image'], atlas['info'],
                                            reannotated=False,
                                            corrected_mni=False,
                                            tolerance=2,
                                            sample_norm='zscore',
                                            gene_norm='zscore',
                                            missing='interpolate',
                                            probe_selection='corr_variance',
                                            donor_probes='independent',
                                            region_agg='donors',
                                            agg_metric='mean',
                                            lr_mirror=None,
                                            ibf_threshold=0,
                                            norm_matched=True,
                                            norm_structures=False,
                                            sim_threshold=5)
    expression.to_hdf(DATA_DIR / 'burt2018.h5', 'data',
                      complib='zlib', complevel=9)

    # Romero-Garcia et al., 2018, NeuroImage
    atlas = abagen.fetch_desikan_killiany(surface=True, native=True)
    expression = abagen.get_expression_data(atlas['image'], atlas['info'],
                                            reannotated=True,
                                            corrected_mni=False,
                                            tolerance=0,
                                            sample_norm=None,
                                            gene_norm='zscore',
                                            missing='interpolate',
                                            probe_selection='max_intensity',
                                            donor_probes='aggregate',
                                            region_agg='samples',
                                            agg_metric='median',
                                            lr_mirror='rightleft',
                                            ibf_threshold=0,
                                            norm_matched=True,
                                            norm_structures=False,
                                            sim_threshold=None)
    expression.to_hdf(DATA_DIR / 'romerogarcia2018.h5', 'data',
                      complib='zlib', complevel=9)

    # Anderson et al., 2020, PNAS
    atlas = abagen.fetch_desikan_killiany(surface=True, native=True)
    expression = abagen.get_expression_data(atlas['image'], atlas['info'],
                                            reannotated=False,
                                            corrected_mni=False,
                                            tolerance=-4,
                                            sample_norm='zscore',
                                            gene_norm='zscore',
                                            missing=None,
                                            probe_selection='max_intensity',
                                            donor_probes='aggregate',
                                            region_agg='donors',
                                            agg_metric='mean',
                                            lr_mirror=None,
                                            ibf_threshold=0.2,
                                            norm_matched=True,
                                            norm_structures=False,
                                            sim_threshold=None)
    expression.to_hdf(DATA_DIR / 'anderson2020.h5', 'data',
                      complib='zlib', complevel=9)

    # Liu et al., 2020, NeuroImage
    atlas, atlas_info = abautils.get_dknosub(surface=False, return_info=True)
    expression = abagen.get_expression_data(atlas, atlas_info,
                                            reannotated=True,
                                            corrected_mni=False,
                                            tolerance=5,
                                            sample_norm='zscore',
                                            gene_norm='zscore',
                                            missing=None,
                                            probe_selection='average',
                                            donor_probes='aggregate',
                                            region_agg='donors',
                                            agg_metric='mean',
                                            lr_mirror=None,
                                            ibf_threshold=0.5,
                                            norm_matched=True,
                                            norm_structures=False,
                                            sim_threshold=None)
    expression.to_hdf(DATA_DIR / 'liu2020.h5', 'data',
                      complib='zlib', complevel=9)

    out_fname = DATA_DIR / 'analysis.csv'
    volatlas = abautils.get_dknosub(surface=False)
    surfatlas = abautils.get_dknosub(surface=True)
    t1t2 = analysis.parcellate_t1t2(RAW_DIR / 'hcp',
                                    RAW_DIR / 'dk' / 'fslr32k')
    modules = analysis.load_oldham2008_modules(all_modules=False)

    # run the three analyses for expression data generated from each of the
    # manuscript pipelines
    data = []
    for study, surf in SURFACE.items():
        expression = pd.read_hdf(DATA_DIR / f'{study}.h5')
        atlas = volatlas if not surf else surfatlas
        distp, dists = analysis.correlate_distance(expression, atlas)
        t1t2p, t1t2s = analysis.correlate_t1t2(expression, t1t2)
        silhouette = analysis.gene_silhouette(expression, modules)
        data.append(dict(
            filename=f'{study}.h5',
            dist_pearsonr=distp, dist_spearmanr=dists,
            t1t2_pearsonr=t1t2p, t1t2_spearmanr=t1t2s,
            silhouette=silhouette
        ))
        pd.DataFrame(data).to_csv(out_fname, index=False)


if __name__ == "__main__":
    main()
