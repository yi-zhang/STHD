import argparse
import os
import sys
from collections import defaultdict

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm
from STHD import train


def get_raw_bins(adata, nspot):
    array_row_min, array_col_min = (
        adata.obs["array_row"].min(),
        adata.obs["array_col"].min(),
    )
    bin_row = (
        ((adata.obs["array_row"].values - array_row_min) // nspot).astype(int) * nspot
        + array_row_min
    ).astype(int)
    bin_col = (
        ((adata.obs["array_col"].values - array_col_min) // nspot).astype(int) * nspot
        + array_col_min
    ).astype(int)

    bins = defaultdict(
        list
    )  # []defaultdict must guarantee order of bins. e.g. python>3.7
    for i in tqdm(range(len(bin_row))):
        bins[f"{bin_row[i]}||{bin_col[i]}"].append(i)
    return bins


def get_bins(adata, nspot):
    array_row_min, array_col_min = (
        adata.obs["array_row"].min(),
        adata.obs["array_col"].min(),
    )
    bin_row = (
        ((adata.obs["array_row"].values - array_row_min) // nspot).astype(int) * nspot
        + array_row_min
    ).astype(int)
    bin_col = (
        ((adata.obs["array_col"].values - array_col_min) // nspot).astype(int) * nspot
        + array_col_min
    ).astype(int)
    bin_ct = adata.obs["STHD_pred_ct"].values

    bins = defaultdict(list)
    for i in tqdm(range(len(bin_row))):
        bins[f"{bin_row[i]}||{bin_col[i]}||{bin_ct[i]}"].append(i)
    return bins


def bin_X(adata, bins):
    """Use data in adata.X."""
    bin_value = []
    for bid in tqdm(bins, total=len(bins)):
        cur_ids = bins[bid]
        cur_value = np.squeeze(np.asarray(adata.X[cur_ids].mean(axis=0)))
        bin_value.append(cur_value)
    return np.array(bin_value)


def bin_obsm(adata, bins, obsm_key_list=["spatial"]):
    obsm_res = dict()
    print("[Log] binning obsm...")
    for obsm_key in obsm_key_list:
        bin_value = []
        for bid in tqdm(bins, total=len(bins)):
            cur_ids = bins[bid]
            cur_value = adata.obsm[obsm_key][cur_ids].mean(axis=0)
            bin_value.append(cur_value)
        obsm_res[obsm_key] = np.array(bin_value)
    return obsm_res


def bin_raw_obs(adata, bins):
    val_bin_row = []
    val_bin_col = []
    val_barcodes_included = []
    val_n_counts = []
    val_bin_n_spot = []

    barcodes = np.array(list(adata.obs.index))

    bin_id_list = list(bins.keys())
    print("[Log] binning obs...")
    for bid in tqdm(bin_id_list, total=len(bin_id_list)):
        cur_bin_row, cur_bin_col = bid.split("||")
        cur_ids = bins[bid]

        cur_bin_n_spot = len(cur_ids)
        cur_barcodes_included = list(barcodes[cur_ids])
        cur_n_counts = adata.X[cur_ids].sum()

        val_bin_row.append(cur_bin_row)
        val_bin_col.append(cur_bin_col)
        val_barcodes_included.append(cur_barcodes_included)
        val_n_counts.append(cur_n_counts)
        val_bin_n_spot.append(cur_bin_n_spot)
    return pd.DataFrame(
        {
            "bin_row": val_bin_row,
            "bin_col": val_bin_col,
            "barcodes_included": val_barcodes_included,
            "n_counts": val_n_counts,
            "bin_n_spot": val_bin_n_spot,
        },
        index=bin_id_list,
    )


def bin_obs(adata, bins):
    val_bin_row = []
    val_bin_col = []
    val_STHD_pred_ct = []
    val_barcodes_included = []
    val_n_counts = []
    val_bin_n_spot = []

    barcodes = np.array(list(adata.obs.index))

    bin_id_list = list(bins.keys())
    print("[Log] binning obs...")
    for bid in tqdm(bin_id_list, total=len(bin_id_list)):
        cur_bin_row, cur_bin_col, cur_STHD_pred_ct = bid.split("||")
        cur_ids = bins[bid]

        cur_bin_n_spot = len(cur_ids)
        cur_barcodes_included = list(barcodes[cur_ids])
        cur_n_counts = adata.X[cur_ids].sum()

        val_bin_row.append(cur_bin_row)
        val_bin_col.append(cur_bin_col)
        val_STHD_pred_ct.append(cur_STHD_pred_ct)
        val_barcodes_included.append(cur_barcodes_included)
        val_n_counts.append(cur_n_counts)
        val_bin_n_spot.append(cur_bin_n_spot)
    return pd.DataFrame(
        {
            "bin_row": val_bin_row,
            "bin_col": val_bin_col,
            "STHD_pred_ct": val_STHD_pred_ct,
            "barcodes_included": val_barcodes_included,
            "n_counts": val_n_counts,
            "bin_n_spot": val_bin_n_spot,
        },
        index=bin_id_list,
    )


def bin_obs_v2(adata, bins, extra_obs_col=""):
    """Temp for : extra obs cols, has to be numerical"""
    val_bin_row = []
    val_bin_col = []
    val_STHD_pred_ct = []
    val_barcodes_included = []
    val_n_counts = []
    val_bin_n_spot = []
    val_extra_obs_col = []

    barcodes = np.array(list(adata.obs.index))

    bin_id_list = list(bins.keys())
    print("[Log] binning obs...")
    for bid in tqdm(bin_id_list, total=len(bin_id_list)):
        cur_bin_row, cur_bin_col, cur_STHD_pred_ct = bid.split("||")
        cur_ids = bins[bid]

        cur_bin_n_spot = len(cur_ids)
        cur_barcodes_included = list(barcodes[cur_ids])
        cur_n_counts = adata.X[cur_ids].sum()

        cur_col_value = adata.obs.iloc[cur_ids][extra_obs_col].mean()

        val_bin_row.append(cur_bin_row)
        val_bin_col.append(cur_bin_col)
        val_STHD_pred_ct.append(cur_STHD_pred_ct)
        val_barcodes_included.append(cur_barcodes_included)
        val_n_counts.append(cur_n_counts)
        val_bin_n_spot.append(cur_bin_n_spot)
        val_extra_obs_col.append(cur_col_value)

    return pd.DataFrame(
        {
            "bin_row": val_bin_row,
            "bin_col": val_bin_col,
            "STHD_pred_ct": val_STHD_pred_ct,
            "barcodes_included": val_barcodes_included,
            "n_counts": val_n_counts,
            "bin_n_spot": val_bin_n_spot,
            "extra_obs_col": val_extra_obs_col,
        },
        index=bin_id_list,
    )


def bin_X_csr(adata, bins):
    """Use data in adata.X."""
    bin_values = []
    print("[LOG] Calculating each bin's sum values")
    for bin_name in tqdm(bins, total=len(bins)):
        bin_values.append(sum_csr(bins[bin_name], adata))
    res_csr = concatenate_csr_matrices(bin_values)
    return res_csr


def average_csr(cur_ids, adata):
    cur_value = np.asarray(adata.X[cur_ids].mean(axis=0))
    cur_value = sp.csr_array(cur_value)
    return cur_value


def sum_csr(cur_ids, adata):
    cur_value = np.asarray(adata.X[cur_ids].sum(axis=0))
    cur_value = sp.csr_array(cur_value)
    return cur_value


def concatenate_csr_matrices(matrix_list):
    ncol = matrix_list[0].shape[1]
    for m in matrix_list:
        if m.shape[1] != ncol:
            raise ValueError("Number of columns must be the same for all matrices.")

    n_val = 0
    n_row = 0
    for m in matrix_list:
        n_val += len(m.data)
        n_row += len(m.indptr) - 1

    res_data = np.zeros(n_val)
    res_indices = np.zeros(n_val)
    res_indptr = np.zeros(n_row + 1)

    i = 0  # index for both data and indices
    j = 0  # index for indptr
    s = 0  # adjustment for indptr
    print("[LOG] Constructing adata.X as a sparse matrix")
    for m in tqdm(matrix_list, total=len(matrix_list)):
        res_data[i : i + len(m.data)] = m.data
        res_indices[i : i + len(m.indices)] = m.indices
        res_indptr[j : j + len(m.indptr) - 1] = m.indptr[1:] + s

        i += len(m.data)
        j += len(m.indptr) - 1
        s += m.indptr[-1]

    res_indptr[1:] = res_indptr[:-1]
    res_indptr[0] = 0
    concatenated_matrix = sp.csr_matrix(
        (res_data, res_indices, res_indptr), shape=(len(res_indptr) - 1, m.shape[1])
    )
    return concatenated_matrix


def get_raw_bin_adata(sthdata, nspot=4, min_nspot_to_aggregate=2, use_csr=True):
    bins = get_raw_bins(sthdata.adata, nspot)
    if use_csr:
        X_binned = bin_X_csr(sthdata.adata, bins)
    else:
        X_binned = bin_X(sthdata.adata, bins)
    obs_binned = bin_raw_obs(sthdata.adata, bins)
    obsm_binned = bin_obsm(sthdata.adata, bins)
    binvar = sthdata.adata.var.copy()

    ###########  create binadata ##########
    binadata = anndata.AnnData(
        X_binned,  # raw bin count
        var=binvar,  # gene var
        obs=obs_binned,  # new bin, barcode is array_row||array_col
        obsm=obsm_binned,  # original obsm plus spatial , mean of bin barcode
        uns=sthdata.adata.uns.copy(),  # original uns, image
    )
    # filter minimum spot number
    binadata = binadata[binadata.obs["bin_n_spot"] >= min_nspot_to_aggregate]

    # change list to comma
    binadata.obs["barcodes_included"] = binadata.obs["barcodes_included"].str.join(
        sep=","
    )
    print(binadata)
    return binadata


def get_sthd_guided_bin_adata(
    sthdata,
    pred_col="STHD_pred_ct",
    nspot=4,
    remove_ambiguous_spot=True,
    ambiguous_spot_celltypes=["ambiguous", "filtered"],
    min_nspot_to_aggregate=2,
    use_csr=True,
):
    bins = get_bins(sthdata.adata, nspot)
    if use_csr:
        X_binned = bin_X_csr(sthdata.adata, bins)
    else:
        X_binned = bin_X(sthdata.adata, bins)
    obs_binned = bin_obs(sthdata.adata, bins)
    obsm_binned = bin_obsm(sthdata.adata, bins)
    binvar = sthdata.adata.var.copy()

    ###########  create binadata ##########
    binadata = anndata.AnnData(
        X_binned,  # raw bin count
        var=binvar,  # gene var
        obs=obs_binned,  # new bin, barcode is array_row||array_col
        obsm=obsm_binned,  # original obsm plus spatial , mean of bin barcode
        uns=sthdata.adata.uns.copy(),  # original uns, image
    )
    """
    ###########  create spatial, take mean for each bin ########## 
    spatial = pd.DataFrame(sthdata.adata.obsm['spatial'], index=sthdata.adata.obs.index, columns=['x','y'])
    for i in obs_binned.index:
        barcodes = obs_binned.loc[i]['barcodes_included']

        a = spatial.loc[barcodes]
        xc,yc = a.mean().values
        obs_binned.loc[i,'x'] = xc
        obs_binned.loc[i,'y'] = yc

    binadata.obsm['spatial'] = obs_binned[['x','y']].values
    """
    print("[Log] remove classes to be filtered out...")
    if remove_ambiguous_spot:  # Optional: remove ambiguous and filtered
        binadata = binadata[~binadata.obs[pred_col].isin(ambiguous_spot_celltypes)]
    print("[Log] filter minimum spot number...")
    binadata = binadata[binadata.obs["bin_n_spot"] >= min_nspot_to_aggregate]

    # change list to comma
    binadata.obs["barcodes_included"] = binadata.obs["barcodes_included"].str.join(
        sep=","
    )
    print(binadata)
    return binadata


def get_sthd_guided_bin_adata_v2(
    adata,
    pred_col="STHD_pred_ct",
    nspot=4,
    remove_ambiguous_spot=True,
    ambiguous_spot_celltypes=["ambiguous", "filtered"],
    min_nspot_to_aggregate=2,
    use_csr=True,
    obs_extra_col_tocollapse="",
):
    bins = get_bins(adata, nspot)
    if use_csr:
        X_binned = bin_X_csr(adata, bins)
    else:
        X_binned = bin_X(adata, bins)
    if obs_extra_col_tocollapse != "":
        obs_binned = bin_obs_v2(adata, bins, obs_extra_col_tocollapse)
    else:
        obs_binned = bin_obs(adata, bins)
    obsm_binned = bin_obsm(adata, bins)
    binvar = adata.var.copy()

    ###########  create binadata ##########
    binadata = anndata.AnnData(
        X_binned,  # raw bin count
        var=binvar,  # gene var
        obs=obs_binned,  # new bin, barcode is array_row||array_col
        obsm=obsm_binned,  # original obsm plus spatial , mean of bin barcode
        uns=adata.uns.copy(),  # original uns, image
    )
    print("[Log] remove classes to be filtered out...")
    if remove_ambiguous_spot:  # Optional: remove ambiguous and filtered
        binadata = binadata[~binadata.obs[pred_col].isin(ambiguous_spot_celltypes)]
    print("[Log] filter minimum spot number...")
    binadata = binadata[binadata.obs["bin_n_spot"] >= min_nspot_to_aggregate]

    # change list to comma
    binadata.obs["barcodes_included"] = binadata.obs["barcodes_included"].str.join(
        sep=","
    )
    print(binadata)
    return binadata