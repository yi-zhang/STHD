import argparse
import os
from time import time

import pandas as pd
from STHD import model, qcmask, refscrna, sthdio


def sthdata_match_refgene(sthd_data, refile, ref_gene_filter=True, ref_renorm=False):
    genemeanpd_filtered = refscrna.load_scrna_ref(refile)
    ng1 = sthd_data.adata.shape[1]
    sthd_data.match_refscrna(genemeanpd_filtered, cutgene=True, ref_renorm=ref_renorm)
    ng2 = sthd_data.adata.shape[1]
    print(
        f"cut {ng1} genes to match to reference {ng2} genes",
    )
    if ref_gene_filter:
        genemeanpd_filtered = genemeanpd_filtered.loc[sthd_data.adata.var_names]
    return (sthd_data, genemeanpd_filtered)


def train(sthd_data, n_iter, step_size, beta, debug=False, early_stop=False):
    print("[Log] prepare_constants and training weights")
    X, Y, Z, F, Acsr_row, Acsr_col = model.prepare_constants(sthd_data)
    W, eW, P, Phi, ll_wat, ce_wat, m, v = model.prepare_training_weights(X, Y, Z)
    print("[Log] Training...")
    metrics = model.train(
        n_iter=n_iter,
        step_size=step_size,  # learnin rate
        beta=beta,  # weight of CE w.r.t log-likelihood
        constants=(X, Y, Z, F, Acsr_row, Acsr_col),
        weights=(W, eW, P, Phi, ll_wat, ce_wat, m, v),
        early_stop=early_stop,  # True will trigger early_stop criteria, False will run through all n_iter
    )
    if debug:
        return P, metrics
    else:
        return P


def fill_p_filtered_to_p_full(
    P_filtered, sthd_data_filtered, genemeanpd_filtered, sthd_data
):
    P_filtered_df = pd.DataFrame(
        P_filtered,
        index=sthd_data_filtered.adata.obs.index,
        columns=genemeanpd_filtered.columns,
    )
    P_filtered_df.columns = "p_ct_" + P_filtered_df.columns
    p_columns = P_filtered_df.columns

    obs_withp = sthd_data.adata.obs.merge(
        P_filtered_df, how="left", left_index=True, right_index=True
    )
    P = obs_withp[list(p_columns)].fillna(-1).values
    return P


def predict(sthd_data, p, genemeanpd_filtered, mapcut=0.8):
    adata = sthd_data.adata.copy()
    for i, ct in zip(
        range(len(genemeanpd_filtered.columns)), genemeanpd_filtered.columns
    ):
        adata.obs["p_ct_" + ct] = p[:, i]
    adata.obs["x"] = adata.obsm["spatial"][:, 0]
    adata.obs["y"] = adata.obsm["spatial"][:, 1]

    # get map predictions
    STHD_prob = adata.obs[[t for t in adata.obs.columns if "p_ct_" in t]]
    ct_max = STHD_prob.columns[STHD_prob.values.argmax(1)]
    STHD_pred_ct = pd.DataFrame({"ct_max": ct_max}, index=STHD_prob.index)
    STHD_pred_ct["ct"] = STHD_pred_ct["ct_max"]

    # assign ambiguous based on posterior cut
    ambiguous_mask = (STHD_prob.max(axis=1) < mapcut).values
    STHD_pred_ct.loc[ambiguous_mask, "ct"] = "ambiguous"

    # assign filtered region to 'filtered'
    filtered_mask = (
        STHD_prob.max(1) < 0
    ).values  # by default, filtered spots have prob as -1.
    STHD_pred_ct.loc[filtered_mask, "ct"] = "filtered"

    # assign final cell type prediction
    adata.obs["STHD_pred_ct"] = STHD_pred_ct["ct"]
    print("[Log]Predicted cell type in STHD_pred_ct in adata.obs")
    print(
        "[Log]Predicted cell type probabilities in columns starting with p_ct_ in adata.obs"
    )
    sthd_data.adata = adata

    return sthd_data


########## Training IO: Saving


def save_prediction_pdata(sthdata, file_path=None, prefix=""):
    predcols = (
        ["x", "y"]
        + ["STHD_pred_ct"]
        + [t for t in sthdata.adata.obs.columns if "p_ct_" in t]
    )
    pdata = sthdata.adata.obs[predcols]

    if file_path is not None:
        pdata_path = os.path.join(file_path, prefix + "_pdata.tsv")
        pdata.to_csv(pdata_path, sep="\t")
        print(f"[Log] prediction saved to {pdata_path}")
    return pdata


########## Training IO: Loading


def load_data(file_path):
    sthd_data = sthdio.STHD(
        spatial_path=os.path.join(file_path, "adata.h5ad.gzip"),
        counts_data=None,
        full_res_image_path=os.path.join(file_path, "fullresimg_path.json"),
        load_type="crop",
    )
    print("[log] Number of spots: ", sthd_data.adata.shape[0])
    return sthd_data


def load_pdata(file_path, prefix=""):
    pdata_path = os.path.join(file_path, prefix + "_pdata.tsv")
    pdata = pd.read_table(pdata_path, index_col=0)
    return pdata


def add_pdata(sthd_data, pdata):
    sthdata = sthd_data
    exist_cols = sthdata.adata.obs.columns.intersection(pdata.columns)
    print("[Log] Loading prediction into sthdata.adata.obs, overwriting")
    print(exist_cols)
    for col in sthdata.adata.obs[exist_cols]:
        del sthdata.adata.obs[col]
    sthdata.adata.obs = sthdata.adata.obs.merge(
        pdata, how="left", left_index=True, right_index=True
    )
    return sthdata


def load_data_with_pdata(file_path, pdata_prefix=""):
    sthdata = load_data(file_path)
    pdata = load_pdata(file_path, pdata_prefix)
    sthdata_with_pdata = add_pdata(sthdata, pdata)
    return sthdata_with_pdata