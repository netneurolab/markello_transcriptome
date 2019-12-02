#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Runs three analyses on outputs of abagen pipelines
"""

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd

from vibecheck import abautils, analysis

RAW_DIR = Path('./data/raw').resolve()
DATA_DIR = Path('./data/derivatives').resolve()


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
    out_dir = DATA_DIR / args["atlas"]
    out_dir.mkdir(parents=True, exist_ok=True)
    sl = slice(args['start'], args['start'] + args['n_jobs'])
    print(f'START: {sl.start}, STOP: {sl.stop}', flush=True)
    print(f'OUTPUT_DIRECTORY: {out_dir}')

    params = pd.read_csv(out_dir / 'parameters.csv').replace({np.nan: None})
    params = params.iloc[sl]

    # prepare information we'll need for the analyses
    atlas = abautils.get_dknosub(surface=args['atlas'] == 'dksurf')
    t1t2 = analysis.parcellate_t1t2(RAW_DIR / 'hcp',
                                    RAW_DIR / 'dk' / 'fslr32k')
    modules = analysis.load_oldham2008_modules(all_modules=False)

    data, fnames = [], set()
    out_fname = out_dir / f'analysis_{args["start"]}.csv'
    if out_fname.exists():
        fn = pd.read_hdf(out_fname)
        data = fn.to_dict('records')
        fnames = set(fn['filename'])
    for n, job in params.iterrows():
        kwargs = job.to_dict()
        fname = out_dir / kwargs.pop('filename')
        if fname.name in fnames:
            continue

        t = time.strftime('%Y-%m-%d:%H:%M:%S')
        print(f'{t} {fname.name}', flush=True)

        distp, dists = analysis.correlate_distance(fname, atlas)
        t1t2p, t1t2s = analysis.correlate_t1t2(fname, t1t2)
        silhouette = analysis.gene_silhouette(fname, modules)
        data.append(dict(
            filename=fname.name, atlas_name=args['atlas'],
            dist_pearsonr=distp, dist_spearmanr=dists,
            t1t2_pearsonr=t1t2p, t1t2_spearmanr=t1t2s,
            silhouette=silhouette
        ))
        pd.DataFrame(data).to_csv(out_fname, index=False)


if __name__ == '__main__':
    main()
