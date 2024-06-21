import argparse
import os
import pathlib
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
from STHD import sthdio, train


def _load_into_dict(res_dict, file, columns):
    data = train.load_pdata(file)
    indices = data.index.tolist()
    cur_columns = data.columns.tolist()
    if columns != cur_columns:
        raise ValueError("Patches have mismatched column names")
    if columns[:3] != ["x", "y", "STHD_pred_ct"]:
        raise ValueError("Patch column orders are not correct")
    values = data.values

    for i, barcode in enumerate(indices):
        res_dict[barcode].append(values[i])


def _process_barcode(res_dict, columns):
    id_STHD_pred_ct = columns.index("STHD_pred_ct")

    for barcode in tqdm(res_dict):
        data = np.array(res_dict[barcode])
        data = np.delete(data, id_STHD_pred_ct, axis=1)
        data_non_filtered = data[data[:, -1] != -1]

        if len(data) == 1:
            res_dict[barcode] = data[0]
        elif len(data_non_filtered) == 0:
            res_dict[barcode] = data[0]
        else:
            res_dict[barcode] = data_non_filtered.mean(axis=0)


def _combine_patch(patch_dir):
    files = [os.path.join(patch_dir, f) for f in os.listdir(patch_dir)]
    res_dict = defaultdict(list)
    columns = train.load_pdata(files[0]).columns.tolist()

    print("[log] Loading patches")
    for file in tqdm(files):
        _load_into_dict(res_dict, file, columns)

    print("[log] Process probabilities in each barcodes")
    _process_barcode(res_dict, columns)

    print("[log] Reshape data into pandas dataframe")
    columns_remove_prediction = columns.copy()
    columns_remove_prediction.remove("STHD_pred_ct")
    pdata = pd.DataFrame.from_dict(
        res_dict, orient="index", columns=columns_remove_prediction
    )

    return pdata


def _patchify(x1, y1, x2, y2, w, h, dw=10, dh=10):
    # (x1, y1, x2, y2): coordinate of the region to patchify
    # w: width of each path. x1, x2 is in width direction.
    # h: height of each path. y1, y2 is in height direction.
    # dw: overlap pixels in width direction
    # dh: overlap pixels in height direction
    print("Caution!! Patchify should be ran on non-cropped data only!")
    xs = list(range(x1, x2, w)) + [x2]
    ys = list(range(y1, y2, h)) + [y2]
    patches = []
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            patches.append([xs[i] - dw, ys[j] - dh, xs[i + 1] + dw, ys[j + 1] + dh])
    return patches


def patchify(sthd_data, save_path, x1, y1, x2, y2, dx, dy, scale_factor):
    allregion_path = f"{save_path}/all_region"
    patch_path = f"{save_path}/patches"

    pathlib.Path(allregion_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(patch_path).mkdir(parents=True, exist_ok=True)

    all_region = sthd_data.crop(
        x1,
        x2,
        y1,
        y2,
        scale_factor,
    )
    all_region.save(allregion_path)

    patches = _patchify(x1, y1, x2, y2, dx, dy)
    # cautious! The patch must contains sequensing data, otherwise it will return error.
    for patch in patches:
        x1_patch, y1_patch, x2_patch, y2_patch = patch
        crop_data = sthd_data.crop(
            x1_patch,
            x2_patch,
            y1_patch,
            y2_patch,
            scale_factor,
        )
        crop_data.save(f"{patch_path}/{x1_patch}_{y1_patch}")


def merge(save_path, refile):
    allregion_path = f"{save_path}/all_region"
    patch_path = f"{save_path}/patches"

    # load sthd_data that were patchified
    sthdata = train.load_data(allregion_path)
    sthdata, genemeanpd_filtered = train.sthdata_match_refgene(
        sthdata, refile, ref_gene_filter=True, ref_renorm=False
    )
    pdata = _combine_patch(patch_path)

    # Align the pdata's barcode order to sthdata
    align_sthdata_pdata = train.add_pdata(sthdata, pdata)
    pdata_reorder = align_sthdata_pdata.adata.obs[
        [t for t in align_sthdata_pdata.adata.obs.columns if "p_ct_" in t]
    ]
    if [
        "p_ct_" + i for i in genemeanpd_filtered.columns.tolist()
    ] != pdata_reorder.columns.tolist():
        raise ValueError("Cell type order miss aligned")

    # Predict cell types and save results
    sthdata_with_pdata = train.predict(
        sthdata, pdata_reorder.values, genemeanpd_filtered, mapcut=0.8
    )
    _ = train.save_prediction_pdata(sthdata_with_pdata, file_path=allregion_path)
