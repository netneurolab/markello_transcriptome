#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generates CSVs containing all combinations of processing parameters to test
"""

import itertools
from pathlib import Path
import uuid

import pandas as pd

DATA_DIR = Path('./data/derivatives').resolve()

# generate giant list of lists of dict with different parameter combinations
# that we're going to run `abagen` with
OPTS = [
    [{key: val} for val in vals] for key, vals in [
        ('ibf_threshold', [0, 0.25, 0.5]),
        ('probe_selection', ['average', 'max_intensity', 'corr_variance',
                             'corr_intensity', 'diff_stability', 'rnaseq']),
        ('donor_probes', ['aggregate', 'independent', 'common']),
        ('lr_mirror', [None, 'bidirectional', 'leftright']),
        ('missing', [None, 'centroids']),
        ('tolerance', [0, 1, 2]),
        ('sample_norm', ['srs', 'zscore', None]),
        ('gene_norm', ['srs', 'zscore', None]),
        ('norm_matched', [True, False]),
        ('norm_structures', [True, False]),
        ('region_agg', ['donors', 'samples']),
        ('agg_metric', ['mean', 'median']),
        ('corrected_mni', [True, False]),
        ('reannotated', [True, False]),
    ]
]
# `probe_selection` methods for which `donor_probes` MUST be 'aggregate'
AGG_METHODS = ['average', 'diff_stability', 'rnaseq']


def gen_params():
    """ Generates parameters.csv files for running abagen pipelines
    """

    fnames = set()
    for atlas in ('dk', 'dksurf'):
        data = []
        for args in itertools.product(*OPTS):
            # make a dictionary with the given parameter combination
            kwargs = {k: v for d in args for k, v in d.items()}
            kwargs['atlas_name'] = atlas
            # this combination would raise an error
            if (kwargs['donor_probes'] != 'aggregate'
                    and kwargs['probe_selection'] in AGG_METHODS):
                continue
            # get a unique filename
            while True:
                fname = f'{str(uuid.uuid4())}.h5'
                if fname not in fnames:
                    fnames.add(fname)
                    break
            kwargs['filename'] = fname
            data.append(kwargs)
        df = pd.DataFrame(data)
        df.to_csv(DATA_DIR / atlas / 'parameters.csv', index=False)


if __name__ == '__main__':
    gen_params()
