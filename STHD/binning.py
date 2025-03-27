# apply binning on spot level data, following current standards, for comparison.

import anndata
import pandas as pd
import scanpy as sc


def cluster_raw_spots(sthdata, resolution=0.5):
    """Example:
    -------
    adata_spot = binning.cluster_raw_spots(sthdata, resolution = 0.2)
    fig = sq.pl.spatial_scatter(adata_spot,
                          color='leiden',
                          crop_coord = [(x1,y1,x2,y2)], # those are original full res image's pixel coordinates
                          title='leiden_'+str(resolution)
                         )

    """
    adata_spot = sthdata.adata.copy()
    sc.pp.normalize_total(adata_spot, target_sum=10e4)  # seurat style
    sc.pp.log1p(adata_spot)
    sc.pp.highly_variable_genes(adata_spot, n_top_genes=4000)
    sc.tl.pca(adata_spot)
    sc.pp.neighbors(adata_spot)
    sc.tl.leiden(adata_spot, resolution=resolution)
    return adata_spot


def get_bin_adata(sthdata, nspot=4):
    """Simple binning

    Example:
    -------
    binadata = get_bin_adata(sthdata, nspot=4) # e.g. 2umx2um into 8umx8um

    """
    adata = sthdata.adata.copy()
    # whether each row index is multiplicate of 4 (plus min), and aggregate based on the two number as index
    # nspot= 4 # 8um=4*2um

    # get the left top bin start, row and col.
    array_row_min, array_col_min = (
        adata.obs['array_row'].min().astype(int),
        adata.obs['array_col'].min().astype(int),
    )
    adata.obs['array_row_step4'] = (
        nspot *
        (((adata.obs['array_row'] - array_row_min) / nspot).astype(int))
        + array_row_min
    )
    adata.obs['array_col_step4'] = (
        nspot *
        (((adata.obs['array_col'] - array_col_min) / nspot).astype(int))
        + array_col_min
    )

    # create adata df
    tmp = adata.to_df().copy()
    tmp = tmp.merge(
        adata.obs[['array_row_step4', 'array_col_step4']],
        how='left',
        left_index=True,
        right_index=True,
    )
    bindf = tmp.groupby(['array_row_step4', 'array_col_step4']).sum()

    # put in original barcodes and create obs
    binobs = bindf.index.to_frame()
    barcodes_included = tmp.groupby(
        ['array_row_step4', 'array_col_step4'], observed=True
    ).apply(lambda x: x.index.tolist())
    binobs['barcodes_included'] = barcodes_included

    binobs['spot_index'] = (
        binobs['array_row_step4'].astype(str)
        + '_'
        + binobs['array_col_step4'].astype(str)
    )
    binobs = binobs.set_index('spot_index')

    bindf.index = binobs.index
    # create var, recycle
    binvar = adata.var.copy()

    # create binadata
    binadata = anndata.AnnData(bindf, var=binvar, obs=binobs)
    # create spatial, take mean for each bin
    spatial = pd.DataFrame(
        adata.obsm['spatial'], index=adata.obs.index, columns=['x', 'y']
    )
    for i in binobs.index:
        barcodes = binobs.loc[i]['barcodes_included']
        a = spatial.loc[barcodes]
        xc, yc = a.mean().values
        binobs.loc[i, 'x'] = xc
        binobs.loc[i, 'y'] = yc

    binadata.obsm['spatial'] = binobs[['x', 'y']].values

    # put back image
    binadata.uns = adata.uns.copy()

    # get counts
    binadata.obs['n_counts'] = binadata.X.sum(axis=1)
    print(binadata)

    """
    # array_row_col_2_barcode_pd.index.names = ['array_row_step4', 'array_col_step4']
    # original location does not exist 1878,2200
    # sthdata.adata.obs[(sthdata.adata.obs['array_row_int']==1878) & (sthdata.adata.obs['array_col_int']==2200)]
    # so if getting the barcode name, no one
    # bindatadf.merge(array_row_col_2_barcode_pd, how='left',left_index=True, right_index=True)
    """
    return binadata


def get_sthd_guided_bin_adata_v1(
    sthdata,
    pred_col='STHD_pred_ct',
    nspot=4,
    remove_ambiguous_spot=True,
    ambiguous_spot_celltypes=['ambiguous'],
    min_nspot_to_aggregate=2,
):
    """Bin aggregating data, guided by STHD predicted cell type identities of spots. E.g. if one bin has 2 types of spots, they will be aggregated separately, and location is determined by taking mean of x or y for the included spot barcodes
    Params
    ----------
    sthdata:
        STHD class, including adata with spatial, coodinate, fullregimg
    pred_col:
        column in STHD.obs marking STHD predicted cell type
    nspot: int
        Number of  spots to square-bin together, along x or along y.
    remove_ambiguous_spot:
        Whether to remove ambiguous spots or not, based on pred_col
    ambiguous_spot_celltypes:
        name of ambiguous cell identify for spots, e.g. ['ambiguous']
    min_nspot_to_aggregate:
        require that at least min_nspot_to_aggregate number of spots are in the bin to get an aggregated count.
    """
    adata = sthdata.adata.copy()
    array_row_min, array_col_min = (
        adata.obs['array_row'].min().astype(int),
        adata.obs['array_col'].min().astype(int),
    )
    adata.obs['array_row_step4'] = (
        nspot *
        (((adata.obs['array_row'] - array_row_min) / nspot).astype(int))
        + array_row_min
    )
    adata.obs['array_col_step4'] = (
        nspot *
        (((adata.obs['array_col'] - array_col_min) / nspot).astype(int))
        + array_col_min
    )

    ########## create adata df ##########
    tmp = adata.to_df()
    tmp = tmp.merge(
        adata.obs[['array_row_step4', 'array_col_step4', pred_col]],
        how='left',
        left_index=True,
        right_index=True,
    )
    if remove_ambiguous_spot:  # Optional: remove ambiguous
        tmp = tmp.loc[~tmp[pred_col].isin(
            ambiguous_spot_celltypes)]  # ambiguous
    bindf = tmp.groupby(
        ['array_row_step4', 'array_col_step4', pred_col], observed=True
    ).sum()

    ##########  put in original barcodes and create obs ##########
    binobs = bindf.index.to_frame()
    barcodes_included = tmp.groupby(
        ['array_row_step4', 'array_col_step4', pred_col], observed=True
    ).apply(lambda x: x.index.tolist())
    binobs['barcodes_included'] = barcodes_included

    binobs['spot_index'] = (
        binobs['array_row_step4'].astype(str)
        + '_'
        + binobs['array_col_step4'].astype(str)
        + '_'
        + binobs[pred_col].astype(str)
    )
    binobs = binobs.set_index('spot_index')

    bindf.index = binobs.index
    ##########  create var, recycle ##########
    binvar = adata.var.copy()

    ###########  create binadata ##########
    binadata = anndata.AnnData(bindf, var=binvar, obs=binobs)
    ###########  create spatial, take mean for each bin ##########
    spatial = pd.DataFrame(
        adata.obsm['spatial'], index=adata.obs.index, columns=['x', 'y']
    )
    for i in binobs.index:
        barcodes = binobs.loc[i]['barcodes_included']

        a = spatial.loc[barcodes]
        xc, yc = a.mean().values
        binobs.loc[i, 'x'] = xc
        binobs.loc[i, 'y'] = yc

    binadata.obsm['spatial'] = binobs[['x', 'y']].values

    ###########  put back image ##########
    binadata.uns = adata.uns.copy()

    ###########  get counts ##########
    binadata.obs['n_counts'] = binadata.X.sum(axis=1)
    binadata.obs['bin_n_spot'] = binadata.obs['barcodes_included'].apply(
        lambda t: len(t)
    )
    # filter minimum spot number
    binadata = binadata[binadata.obs['bin_n_spot'] >= min_nspot_to_aggregate]

    # change list to comma
    binadata.obs['barcodes_included'] = binadata.obs['barcodes_included'].str.join(
        sep=','
    )

    print(binadata)
    return binadata


def remove_celltype_sthd_guided_bin_adata(
    binadata,
    pred_col='STHD_pred_ct',
    remove_ambiguous_spot=True,
    ambiguous_spot_celltypes=['ambiguous'],
):
    """Remove aggregated spots with ambiguous celll types from bindata"""
    if remove_ambiguous_spot:  # Optional: remove ambiguous
        mask = binadata.obs.loc[
            ~binadata.obs[pred_col].isin(ambiguous_spot_celltypes)
        ]  # ambiguous
        binadata = binadata[mask]
    return binadata


def cluster_bin_data(binadata, resolution=1):
    """# cluster bined data

    Example:
    -------
    adata_bin, fig = cluster_bin_data(binadata, coord, resolution=1)
    sc.pl.umap( binadata,        color="leiden")
    fig = sq.pl.spatial_scatter(binadata,
                          color='leiden',
                          crop_coord = coord,
                          title='leiden_'+str(resolution)
                         )

    """
    if ('counts') not in binadata.layers.keys():
        print('[Log] Copying raw counts to adata.layers["counts"]')
        binadata.layers['counts'] = binadata.X.copy()
    print('[Log] Normalizing')
    sc.pp.normalize_total(binadata, target_sum=10e4)
    sc.pp.log1p(binadata)
    sc.pp.highly_variable_genes(binadata, n_top_genes=4000)
    sc.tl.pca(binadata)
    sc.pp.neighbors(binadata)
    print('[Log] umap for visualization')
    sc.tl.umap(binadata)
    print('[Log] leiden clustering')
    sc.tl.leiden(binadata, resolution=resolution)
    return binadata
