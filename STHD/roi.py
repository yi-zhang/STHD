"""Function for region of interest analysis"""

import pandas as pd


def barcode_convert(sthdata, nspot=4, switch_index=False):
    """Convert barcodes from 2um bin to 8um bin, add to .adata.obs

    Args:
    ----
    sthdata: Data object containing 'adata'.
    nspot: number of spots to bin together.
    switch_index: whether to set index to converted barcodes.

    Returns:
    -------
    Modified 'sthdata' in place by updating the 8um barcodes as index.
    add the old 2um barcodes into the sthdata.adata.obs['barcodes_2um']

    """
    # process row and column num for 8um bins
    # sthdata = sthd_data.copy()
    array_row_8um = (
        pd.Series(sthdata.adata.obs.index)
        .str.split("-")
        .str.get(0)
        .str.split("_")
        .str.get(2)
        .astype(int)
        .apply(lambda x: str(x // nspot).zfill(5))
    )
    array_col_8um = (
        pd.Series(sthdata.adata.obs.index)
        .str.split("-")
        .str.get(0)
        .str.split("_")
        .str.get(3)
        .astype(int)
        .apply(lambda x: str(x // nspot).zfill(5))
    )
    # Update the barcode index in 'sthdata' to 8um bins
    sthdata.adata.obs["barcodes_2um"] = sthdata.adata.obs.index
    sthdata.adata.obs["barcodes_8um"] = (
        "s_008um_" + array_row_8um + "_" + array_col_8um + "-1"
    )
    if switch_index:
        sthdata.adata.obs.set_index("barcodes_8um")

    return sthdata
