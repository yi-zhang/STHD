import matplotlib.pyplot as plt
import squidpy as sq


def spatial_scatter_all_prob(
    sthdata,
    ctcols=None,
    n_plot_cols=6,
    pred_ct_only=True,
    outfile="crop_all_probabilities.png",
    min_num=3,
    single_plot_height=5,
    fig_width=30,
    max_fig_height=60,
):
    """Plot Posterior Probabilities for all STHD pred channels.

    Example:
    -------
    viz.spatial_scatter_all_prob(sthdata,
                         ctcols = config.crc98_order_col,
                         pred_ct_only=False,
                         n_plot_cols = 8,
                         outfile='../crop10_all_probabilities.png'
                        )

    """
    x1, y1, x2, y2 = sthdata.get_sequencing_data_region()
    _ = sthdata.adata.copy()
    if pred_ct_only:
        ctcols = [
            t
            for t in ctcols
            if t
            in (
                set(sthdata.adata.obs["STHD_pred_ct"].values)
                - set(["ambiguous", "filtered"])
            )
        ]
        print(f"[Log] {len(ctcols)} cell types exist in prediction")
        if min_num > 0:
            cts = sthdata.adata.obs.groupby("STHD_pred_ct").count()
            cts = cts[cts[cts.columns[0]] >= min_num]
            ctcols = [t for t in ctcols if t in cts.index.tolist()]
            print(f"[Log] {len(ctcols)} cell types with min number {min_num}")

    if len(ctcols) < n_plot_cols:
        print("[Warning] plotting: select less column. using n ctcols.")
        n_plot_cols = len(ctcols)
    n_row = len(ctcols) // n_plot_cols + 1
    n_col = n_plot_cols
    f, ax = plt.subplots(
        n_row,
        n_col,
        figsize=(fig_width, min(max_fig_height, single_plot_height * n_row)),
    )
    for cur_row in range(n_row):
        for cur_col in range(n_col):
            ind = cur_row * n_col + cur_col
            if ind >= len(ctcols):
                break
            cur_ct = ctcols[ind]
            if cur_ct not in sthdata.adata.obs.columns:
                continue
            sq.pl.spatial_scatter(
                sthdata.adata,
                color=cur_ct,
                crop_coord=[(x1, y1, x2, y2)],
                ax=ax[cur_row, cur_col],
                vmin=0,
                vmax=1,
                colorbar=False,
                legend_fontsize="xx-small",
            )
    plt.savefig(outfile, dpi=300)
