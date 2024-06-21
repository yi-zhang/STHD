import matplotlib.pyplot as plt
import numpy as np
import squidpy as sq
from numba import njit


def background_detector(
    adata, layer=None, threshold=50, n_neighs=4, n_rings=4, coord_type="grid"
):
    if layer == "count":
        adataX = adata.layers["count"]
    else:
        adataX = adata.X
    X = adata.obs.shape[0]  # n of spot
    D_raw = np.squeeze(
        np.asarray(adataX.sum(axis=1))
    )  # [X] per spot counts. IMPORTANT: DO NOT use .to_df - slow
    # print(D_raw[:10])
    p_background = D_raw.copy()
    sq.gr.spatial_neighbors(
        adata,
        spatial_key="spatial",
        coord_type=coord_type,
        n_neighs=n_neighs,
        n_rings=n_rings,
    )
    A_csr = adata.obsp["spatial_distances"]  # [X, X]
    Acsr_row = A_csr.indptr
    Acsr_col = A_csr.indices
    Acsr_val = A_csr.data
    aggregate_neighbor(D_raw, p_background, Acsr_row, Acsr_col, Acsr_val, X)
    adata.obs["neighbor_agg_counts"] = p_background
    adata.obs["neighbor_agg_log_counts"] = np.log(p_background + 1)
    adata.obs["low_count_region"] = (p_background < threshold).astype(int)
    return adata


def visualize_background(sthdata):
    x1, y1, x2, y2 = sthdata.get_sequencing_data_region()
    f, ax = plt.subplots(2, 2, figsize=(15, 15))
    # histogram
    _ = ax[0, 0].hist(
        sthdata.adata.obs["neighbor_agg_counts"],
        bins=500,
    )
    ax[0, 0].set_xlim([-10, 300])
    ax[0, 0].set_title("aggregated neighbor coutns")
    # neighbor agg log counts
    sq.pl.spatial_scatter(
        sthdata.adata,
        color=["neighbor_agg_log_counts"],
        crop_coord=[(x1, y1, x2, y2)],
        # shape='square',
        legend_fontsize=6,
        ax=ax[0, 1],
    )
    #
    img = sthdata.load_img()
    img_crop = sthdata.crop_img(img, x1, y1, x2, y2)
    ax[1, 0].imshow(img_crop)
    # islowcount
    sq.pl.spatial_scatter(
        sthdata.adata,
        color=["low_count_region"],
        crop_coord=[(x1, y1, x2, y2)],
        legend_fontsize=6,
        ax=ax[1, 1],
    )


def filter_background(
    sthdata,
    threshold=50,
    n_neighs=4,
    n_rings=4,
    coord_type="grid",
    low_cts_col="low_count_region",
    inplace=False,
):
    if inplace:
        sthdata_filtered = sthdata
    else:
        sthdata_filtered = sthdata.copy()
    sthdata_filtered.adata = background_detector(
        sthdata_filtered.adata,
        threshold=threshold,
        n_neighs=n_neighs,
        n_rings=n_rings,
        coord_type=coord_type,
    )
    before = sthdata_filtered.adata.shape[0]
    sthdata_filtered.adata = sthdata_filtered.adata[
        sthdata_filtered.adata.obs[low_cts_col] < 1
    ]
    after = sthdata_filtered.adata.shape[0]
    print(f"[Log] filtering background: {before} spots to {after} spots")
    return sthdata_filtered


@njit
def csr_obtain_column_val_for_row(row, column, val, i):
    # get row i's non-zero items' column index
    row_start = row[i]
    row_end = row[i + 1]
    column_indices = column[row_start:row_end]
    values = val[row_start:row_end]
    return column_indices, values


@njit
def aggregate_neighbor(D_raw, p_background, row, column, val, X):
    for a in range(X):
        neighbors, distances = csr_obtain_column_val_for_row(row, column, val, a)
        for i in range(len(neighbors)):
            neighbor = neighbors[i]
            distance = distances[i]
            p_background[a] += D_raw[neighbor] / (distance + 1)
