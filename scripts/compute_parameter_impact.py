#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Computes average impact of each processing parameter for every analysis
"""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

REPO_DIR = Path('.').resolve()
DATA = pd.read_csv(REPO_DIR / 'data' / 'derivatives' / 'pipelines.csv.gz') \
         .replace({np.nan: "None"})
PARAMETERS = DATA.columns[:15]


def _apply(df, metric, param):
    """
    For use in `get_parameter_impact()`

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with information about processing impacts
    metric : str
        Column in `df` to quantify parameter impact
    param : str
        Parameter in `df` that is being assessed

    Returns
    -------
    impact : float
    """

    # grab metric from dataframe
    t = np.asarray(df[metric])

    # for categorical parameters (e.g., `probe_selection`), compute differences
    # between all combinations of parameters
    if np.asarray(df[param]).dtype == np.dtype('object'):
        diffs = [np.diff(f) for f in itertools.combinations(t, 2)]
    # for boolean parameters OR interval parameters only compute differences
    # between adjacent parameters (i.e., for interval range [0, 1, 2] compute
    # 2-1 and 1-0 but not 2-0)
    else:
        diffs = np.diff(t)

    # this is possible in some cases with e.g., `donor_probes` where some
    # processing options were skipped
    if len(diffs) == 0:
        return np.nan

    return np.mean(diffs)


def get_parameter_impact(data, parameters, metric='spearmanr'):
    """
    Returns impact of modifying each of `parameters` on `metric` in `data`

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe with at least columns `parameters` and `metric`
    parameters : list of str
        Columns in `data` specifying parameters that were modified
    metric : str
        Metric in `data` to quantify impact of parameter modification

    Returns
    -------
    impact : pandas.DataFrame
        Where the column is `metric`, the index is `parameters`, and the values
        are the average impact of each parameter on the given `metric`, sorted
        in descending order
    """

    parameters = set(parameters)
    assert len(parameters - set(data.columns)) == 0
    assert metric in data.columns

    impact = dict()
    for param in parameters:
        gb = data.groupby(list(parameters - set([param])))
        out = np.asarray(gb.apply(_apply, metric, param))
        # calculate the average impact score and take the absolute value;
        # we don't really care which direction it moved, just the magnitude
        impact[param] = np.abs(np.mean(out[~np.isnan(out)]))

    impact = pd.DataFrame(impact, index=[metric]).T
    return impact.sort_values(metric, ascending=False)


if __name__ == "__main__":
    impact = pd.concat([
        get_parameter_impact(DATA, PARAMETERS, metric) for metric in
        ('dist_spearmanr', 'silhouette', 't1t2_spearmanr')
    ], axis=1)
    impact.index.name = 'parameter'
    impact.to_csv(REPO_DIR / 'data' / 'derivatives' / 'impact.csv', index=True)
