# -*- coding: utf-8 -*-
"""
Functions for plotting
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt


def pathify(path):
    """
    Convenience function for coercing a potential pathlike to a Path object

    Parameter
    ---------
    path : str or os.PathLike
        Path to be checked for coercion to pathlib.Path object

    Returns
    -------
    path : pathlib.Path
    """

    if isinstance(path, (str, os.PathLike)):
        path = Path(path)
    return path


def savefig(fig, fname, **kwargs):
    """
    Saves `fig` to `fname`, creating parent directories if necessary

    Parameters
    ----------
    fig : matplotlib.pyplot.Figure
        Figure object to be saved
    fname : str or os.PathLike
        Filepath to where `fig` should be saved
    """

    opts = {'bbox_inches': 'tight', 'transparent': True, 'dpi': 500}
    opts.update(**kwargs)
    fname = pathify(fname)
    if not fname.parent.exists():
        fname.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fname, **opts)
    plt.close(fig=fig)
