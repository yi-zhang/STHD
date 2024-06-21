import anndata
import pandas as pd
import scanpy as sc


def cluster_raw_spots(sthdata, resolution=0.5):
    adata_spot = sthdata.adata.copy()
    sc.pp.normalize_total(adata_spot, target_sum=10e4)  # seurat style
    sc.pp.log1p(adata_spot)
    sc.pp.highly_variable_genes(adata_spot, n_top_genes=4000)
    sc.tl.pca(adata_spot)
    sc.pp.neighbors(adata_spot)
    sc.tl.leiden(adata_spot, resolution=resolution)
    return adata_spot


def get_bin_adata(sthdata, nspot=4):
    adata = sthdata.adata.copy()
    # whether each row index is multiplicate of 4 (plus min), and aggregate based on the two number as index
    # nspot= 4 # 8um=4*2um

    ## get the left top bin start, row and col.
    array_row_min, array_col_min = (
        adata.obs["array_row"].min().astype(int),
        adata.obs["array_col"].min().astype(int),
    )
    adata.obs["array_row_step4"] = (
        nspot * (((adata.obs["array_row"] - array_row_min) / nspot).astype(int))
        + array_row_min
    )
    adata.obs["array_col_step4"] = (
        nspot * (((adata.obs["array_col"] - array_col_min) / nspot).astype(int))
        + array_col_min
    )

    # create adata df
    tmp = adata.to_df().copy()
    tmp = tmp.merge(
        adata.obs[["array_row_step4", "array_col_step4"]],
        how="left",
        left_index=True,
        right_index=True,
    )
    bindf = tmp.groupby(["array_row_step4", "array_col_step4"]).sum()

    # put in original barcodes and create obs
    binobs = bindf.index.to_frame()
    barcodes_included = tmp.groupby(
        ["array_row_step4", "array_col_step4"], observed=True
    ).apply(lambda x: x.index.tolist())
    binobs["barcodes_included"] = barcodes_included

    binobs["spot_index"] = (
        binobs["array_row_step4"].astype(str)
        + "_"
        + binobs["array_col_step4"].astype(str)
    )
    binobs = binobs.set_index("spot_index")

    bindf.index = binobs.index
    # create var, recycle
    binvar = adata.var.copy()

    # create binadata
    binadata = anndata.AnnData(bindf, var=binvar, obs=binobs)
    # create spatial, take mean for each bin
    spatial = pd.DataFrame(
        adata.obsm["spatial"], index=adata.obs.index, columns=["x", "y"]
    )
    for i in binobs.index:
        barcodes = binobs.loc[i]["barcodes_included"]
        a = spatial.loc[barcodes]
        xc, yc = a.mean().values
        binobs.loc[i, "x"] = xc
        binobs.loc[i, "y"] = yc

    binadata.obsm["spatial"] = binobs[["x", "y"]].values

    # put back image
    binadata.uns = adata.uns.copy()

    # get counts
    binadata.obs["n_counts"] = binadata.X.sum(axis=1)
    print(binadata)
    return binadata


def get_sthd_guided_bin_adata_v1(
    sthdata,
    pred_col="STHD_pred_ct",
    nspot=4,
    remove_ambiguous_spot=True,
    ambiguous_spot_celltypes=["ambiguous"],
    min_nspot_to_aggregate=2,
):
    adata = sthdata.adata.copy()
    array_row_min, array_col_min = (
        adata.obs["array_row"].min().astype(int),
        adata.obs["array_col"].min().astype(int),
    )
    adata.obs["array_row_step4"] = (
        nspot * (((adata.obs["array_row"] - array_row_min) / nspot).astype(int))
        + array_row_min
    )
    adata.obs["array_col_step4"] = (
        nspot * (((adata.obs["array_col"] - array_col_min) / nspot).astype(int))
        + array_col_min
    )

    ########## create adata df ##########
    tmp = adata.to_df()
    tmp = tmp.merge(
        adata.obs[["array_row_step4", "array_col_step4", pred_col]],
        how="left",
        left_index=True,
        right_index=True,
    )
    if remove_ambiguous_spot:  # Optional: remove ambiguous
        tmp = tmp.loc[~tmp[pred_col].isin(ambiguous_spot_celltypes)]  # ambiguous
    bindf = tmp.groupby(
        ["array_row_step4", "array_col_step4", pred_col], observed=True
    ).sum()

    ##########  put in original barcodes and create obs ##########
    binobs = bindf.index.to_frame()
    barcodes_included = tmp.groupby(
        ["array_row_step4", "array_col_step4", pred_col], observed=True
    ).apply(lambda x: x.index.tolist())
    binobs["barcodes_included"] = barcodes_included

    binobs["spot_index"] = (
        binobs["array_row_step4"].astype(str)
        + "_"
        + binobs["array_col_step4"].astype(str)
        + "_"
        + binobs[pred_col].astype(str)
    )
    binobs = binobs.set_index("spot_index")

    bindf.index = binobs.index
    ##########  create var, recycle ##########
    binvar = adata.var.copy()

    ###########  create binadata ##########
    binadata = anndata.AnnData(bindf, var=binvar, obs=binobs)
    ###########  create spatial, take mean for each bin ##########
    spatial = pd.DataFrame(
        adata.obsm["spatial"], index=adata.obs.index, columns=["x", "y"]
    )
    for i in binobs.index:
        barcodes = binobs.loc[i]["barcodes_included"]

        a = spatial.loc[barcodes]
        xc, yc = a.mean().values
        binobs.loc[i, "x"] = xc
        binobs.loc[i, "y"] = yc

    binadata.obsm["spatial"] = binobs[["x", "y"]].values

    ###########  put back image ##########
    binadata.uns = adata.uns.copy()

    ###########  get counts ##########
    binadata.obs["n_counts"] = binadata.X.sum(axis=1)
    binadata.obs["bin_n_spot"] = binadata.obs["barcodes_included"].apply(
        lambda t: len(t)
    )
    # filter minimum spot number
    binadata = binadata[binadata.obs["bin_n_spot"] >= min_nspot_to_aggregate]

    # change list to comma
    binadata.obs["barcodes_included"] = binadata.obs["barcodes_included"].str.join(
        sep=","
    )

    print(binadata)
    return binadata


def remove_celltype_sthd_guided_bin_adata(
    binadata,
    pred_col="STHD_pred_ct",
    remove_ambiguous_spot=True,
    ambiguous_spot_celltypes=["ambiguous"],
):
    """Remove aggregated spots with ambiguous celll types from bindata"""
    if remove_ambiguous_spot:  # Optional: remove ambiguous
        mask = binadata.obs.loc[
            ~binadata.obs[pred_col].isin(ambiguous_spot_celltypes)
        ]  # ambiguous
        binadata = binadata[mask]
    return binadata


def cluster_bin_data(binadata, resolution=1):
    if ("counts") not in binadata.layers.keys():
        print('[Log] Copying raw counts to adata.layers["counts"]')
        binadata.layers["counts"] = binadata.X.copy()
    print("[Log] Normalizing")
    sc.pp.normalize_total(binadata, target_sum=10e4)
    sc.pp.log1p(binadata)
    sc.pp.highly_variable_genes(binadata, n_top_genes=4000)
    sc.tl.pca(binadata)
    sc.pp.neighbors(binadata)
    print("[Log] umap for visualization")
    sc.tl.umap(binadata)
    print("[Log] leiden clustering")
    sc.tl.leiden(binadata, resolution=resolution)
    return binadata
