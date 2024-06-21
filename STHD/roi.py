import pandas as pd


def barcode_convert(sthdata, nspot=4, switch_index=False):
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
