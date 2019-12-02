# -*- coding: utf-8 -*-
"""
Utility functions for working with abagen
"""

import nibabel as nib
import numpy as np
import pandas as pd

import abagen
from abagen import images
abagen.allen.LGR.setLevel(40)


def get_dknosub(surface=False, return_info=False):
    """
    Gets the Desikan-Killiany atlas from `abagen` but with subcortex removed

    Parameters
    ----------
    surface : bool, optional
        Whether to return surface-based atlas. Default: False
    return_info : bool, optional
        Whether to return a DataFrame with info about atlas. Default: False

    Returns
    -------
    atlas : abagen.AtlasTree
        Cortex-only DK atlas
    atlas_info : pandas.DataFrame
        Info on `atlas`; only returned if `return_info=True`. Also accessible
        via the `atlas.atlas_info` property.
    """

    # load DK atlas + auxiliary info
    atlas = abagen.fetch_desikan_killiany(surface=surface)
    if surface:
        atlas, atlas_info = atlas['image'], atlas['info']
    else:
        atlas, atlas_info = \
            nib.load(atlas['image']), pd.read_csv(atlas['info'])

        # get subcortical IDs and zero them out in the atlas data
        subcortex = atlas_info.query('structure != "cortex"')['id']
        atlas_data = np.asarray(atlas.dataobj)
        atlas_data[np.isin(atlas_data, subcortex)] = 0

        # recreate atlas w/o subcortex and drop from info dataframe
        atlas = atlas.__class__(atlas_data, atlas.affine, header=atlas.header)
        atlas_info = atlas_info.query('structure == "cortex"') \
                               .reset_index(drop=True)

    atlas = images.check_atlas(atlas, atlas_info)

    if return_info:
        return atlas, atlas_info

    return atlas
