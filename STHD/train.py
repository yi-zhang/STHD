"""
Example
python3 STHD/train.py --refile ../testdata/crc_average_expr_genenorm_lambda_98ct_4618gs.txt --patch_list ../testdata/crop10large//patches/52979_9480 ../testdata/crop10large//patches/57479_9480 ../testdata/crop10large//patches/52979_7980 ../testdata/crop10large//patches/55979_7980 ../testdata/crop10large//patches/57479_7980 ../testdata/crop10large//patches/54479_9480 ../testdata/crop10large//patches/55979_9480 ../testdata/crop10large//patches/54479_7980
"""

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
    """For prediction performed for filtered data, put P_filtered back to full data size and fill -1.
    sthd_data cannot already have the probability columns
    """
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
    """Based on per barcode per cell type probability, predict cell type and put prediction in adata.
    sthd_data = predict(sthd_data, p, genemeanpd_filtered, mapcut= 0.8)
    """
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
    """Save from sthdata the pdata into dataframe with probabilities, predicted cell type, and x, y

    Example:
    -------
    pdata = train.save_prediction_pdata(sthdata, file_path = '', prefix = '')

    """
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
    """Load expr data. Only works with cropped data."""
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
    """Load from pdata into dataframe with probabilities, predicted cell type, and x, y, and put in sthdata
    to rename: put_pdata_to_sthdata

    Example:
    -------
    pdata = train.add_pdata(sthdata, pdata)

    """
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
    """A simplified
    Load full prediction for the patch. need os.path.join(file_path, pdata_prefix+_pdata.h5ad)
    Return adata with probability and predicted cell type in .obs
    contains all gene expression
    [] todo: deprecate
    """
    sthdata = load_data(file_path)
    pdata = load_pdata(file_path, pdata_prefix)
    sthdata_with_pdata = add_pdata(sthdata, pdata)
    return sthdata_with_pdata


##########


def main(args):
    start = time()
    for patch_path in args.patch_list:
        print(f"[log] {time() - start:.2f}, start processing patch {patch_path}")
        if args.filtermask:
            sthdata = load_data(patch_path)
            print(sthdata.adata.shape)
            sthdata.adata = qcmask.background_detector(
                sthdata.adata,
                threshold=args.filtermask_threshold,
                n_neighs=4,
                n_rings=args.filtermask_nrings,
            )
            print(sthdata.adata.shape)
            # visualize_background(sthdata)
            sthdata_filtered = qcmask.filter_background(
                sthdata, threshold=args.filtermask_threshold
            )
            sthdata_filtered, genemeanpd_filtered = sthdata_match_refgene(
                sthdata_filtered, args.refile, ref_renorm=args.ref_renorm
            )
            print("[Log]Training")
            P_filtered = train(sthdata_filtered, args.n_iter, args.step_size, args.beta)
            P = fill_p_filtered_to_p_full(
                P_filtered, sthdata_filtered, genemeanpd_filtered, sthdata
            )
        else:
            sthdata = load_data(patch_path)
            sthdata, genemeanpd_filtered = sthdata_match_refgene(
                sthdata, args.refile, ref_renorm=args.ref_renorm
            )
            print("[Log]Training")
            P = train(sthdata, args.n_iter, args.step_size, args.beta)

        sthdata = predict(sthdata, P, genemeanpd_filtered, mapcut=args.mapcut)
        _ = save_prediction_pdata(sthdata, file_path=patch_path, prefix="")
        print("[Log]prediction saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_iter",
        default=23,
        type=int,
        help="iteration to optimize LL and CE. recommended 23 (tuned for human colon cancer sample)0",
    )
    parser.add_argument("--step_size", default=1, type=int)
    parser.add_argument(
        "--beta",
        default=0.1,
        type=float,
        help="beta parameter for borrowing neighbor info. recommended 0.1",
    )
    parser.add_argument(
        "--mapcut",
        default=0.8,
        type=float,
        help="posterior cutoff for celltype prediction",
    )
    parser.add_argument(
        "--refile", type=str, help="reference normalized gene mean expression."
    )
    parser.add_argument(
        "--ref_renorm", default=False, type=bool, help="recommended False"
    )
    parser.add_argument(
        "--filtermask", type=bool, default=True, help="whether to filter masked spots"
    )
    parser.add_argument(
        "--filtermask_nrings",
        default=2,
        type=int,
        help="auto detection of low-count region: number of rings to consider neighbor",
    )
    parser.add_argument(
        "--filtermask_threshold",
        default=51,
        type=int,
        help="auto detection of low-count region: number of total counts",
    )
    parser.add_argument(
        "--patch_list", nargs="+", default=[], help="a space separated patch path list"
    )

    args = parser.parse_args()
    
    main(args)
    
    """
    # quick test
    class Args:
    def __init__(self):
        self.n_iter=10
        self.step_size = 1
        self.beta = 0.1
        self.refile =  '../testdata/crc_average_expr_genenorm_lambda_98ct_4618gs.txt'
        self.filtermask = True
        self.patch_list = ['../testdata/crop10']
    train.main(Args())
    """
