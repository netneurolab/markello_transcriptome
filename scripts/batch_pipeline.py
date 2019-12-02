#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Runs abagen processing pipeline using specific parameters
"""

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd

import abagen
from vibecheck import abautils

DATA_DIR = Path('./data/derivatives').resolve()


def gen_expression(opts, kwargs, fname):
    """
    Generates expression data provided parameters and saves to `fname`

    Parameters
    ----------
    opts : dict
        Options that are consistent across pipelines
    kwargs : dict
        Options that are variable across pipelines
    fname : str or os.PathLike
        Filepath to where generated expression data should be saved. Data are
        saved as HDF5 file
    """
    if fname.exists():
        return

    t = time.strftime('%Y-%m-%d:%H:%M:%S')
    print(f'{t} {fname.name}: {kwargs}', flush=True)

    fkwargs = {k: v for k, v in kwargs.items() if k != 'atlas_name'}
    expression = abagen.get_expression_data(**opts, **fkwargs)
    expression.to_hdf(fname, 'data', complib='zlib', complevel=9)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_jobs', default=50, type=int,
                        help='How many jobs to run. Default: 50')
    parser.add_argument('--atlas', choices=['dk', 'dksurf'], default='dksurf',
                        help='Which atlas to use')
    parser.add_argument('start', type=int, help='Which job to start at')

    return vars(parser.parse_args())


def main():
    args = get_parser()
    out_dir = DATA_DIR / args['atlas']
    out_dir.mkdir(parents=True, exist_ok=True)
    sl = slice(args['start'], args['start'] + args['n_jobs'])
    print(f'START: {sl.start}, STOP: {sl.stop}', flush=True)
    print(f'OUTPUT_DIRECTORY: {out_dir}')

    atlas, info = abautils.get_dknosub(surface=args['atlas'] == 'dksurf',
                                       return_info=True)
    opts = dict(atlas=atlas, atlas_info=info)
    params = pd.read_csv(out_dir / 'parameters.csv').replace({np.nan: None})
    params = params.iloc[sl]

    for n, job in params.iterrows():
        kwargs = job.to_dict()
        fname = out_dir / kwargs.pop('filename')
        gen_expression(opts, kwargs, fname)


if __name__ == '__main__':
    main()
