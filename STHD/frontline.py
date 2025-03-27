from collections import Counter

import numpy as np
from numba import njit, prange
from tqdm import tqdm

from STHD import model

# also try: T cell spot distance to myeloid


def get_neighbor_ct(adata, ctstr='cTNI', ctlst=[]):
    subset_cell_idx1 = np.where(adata.obs['STHD_pred_ct'].str.contains(ctstr))[
        0
    ]  # T cell id
    subset_cell_idx2 = np.where(adata.obs['STHD_pred_ct'].isin(ctlst))[0]
    subset_cell_idx = np.array(
        list(set(np.append(subset_cell_idx1, subset_cell_idx2))))
    _, idx = adata.obsp['spatial_connectivities'][
        subset_cell_idx.T, :
    ].nonzero()  # T cell neighbor id

    idx = np.array(
        list(set(np.append(idx, subset_cell_idx)))
    )  # T cell neighbor id plus T cell id
    subset_neighbor = adata.obs.iloc[idx]
    return subset_neighbor


def get_ambiguous_near_ct(adata, ctstr='cTNI', ctlst=[]):
    """Grep all ambiguous for an input neighbor identity. require sthdata.adata.obsp["spatial_connectivities"] from:
    # sq.gr.spatial_neighbors(sthdata.adata, n_rings=1, coord_type="grid", n_neighs=4)
    """
    #    sthdata = adata.copy()
    subset_cell_idx1 = np.where(adata.obs['STHD_pred_ct'].str.contains(ctstr))[
        0
    ]  # T cell id
    subset_cell_idx2 = np.where(adata.obs['STHD_pred_ct'].isin(ctlst))[0]
    subset_cell_idx = np.array(
        list(set(np.append(subset_cell_idx1, subset_cell_idx2))))
    _, idx = adata.obsp['spatial_connectivities'][
        subset_cell_idx.T, :
    ].nonzero()  # T cell neighbor id

    idx = np.array(
        list(set(np.append(idx, subset_cell_idx)))
    )  # T cell neighbor id plus T cell id

    # subset_neighbor_cell_idx = np.where(sthdata.adata.obs['STHD_pred_ct'].iloc[idx,'STHD_pred_ct']=='ambiguous')[0] # among neighbor, which one is the ambiguous cell type - those id
    # idx = np.append(idx, subset_cell_idx)
    amb_near_subset = adata.obs.iloc[idx][
        adata.obs['STHD_pred_ct'].iloc[idx] == 'ambiguous'
    ].index  # id of ambiguous in T cell neighbor

    # ambiguous spots nearby T cell cTNI cells
    subset_neighbor = adata[amb_near_subset].copy()
    return subset_neighbor


######## frontline 1,3, 2,4 ############
def sthd_neighbor_ct_count(adata):
    # STHD probability columns
    sthd_p_cols = [t for t in adata.obs.columns if t[:5] == 'p_ct_']

    # obtain adjacency indices
    adj = adata.obsp['spatial_connectivities']
    adj_row = adj.indptr
    adj_col = adj.indices

    # extract STHD probability, and use the max one to assign a celltype.
    p = adata.obs[sthd_p_cols]
    cell_type_names = list(p.columns)
    pmap = np.array([cell_type_names[i] for i in p.values.argmax(1)])
    pmap_value = p.values.max(1)
    print('[Log] remove filtered')
    pmap[pmap_value == -1] = 'filtered'

    # count neighbour cell types
    neighbors = [
        model.csr_obtain_column_index_for_row(adj_row, adj_col, i)
        for i in range(len(p))
    ]
    neighbor_celltypes = [Counter(pmap[n]) for n in tqdm(neighbors)]
    neighbor_celltype_count = np.array(
        [len(n_ct) for n_ct in tqdm(neighbor_celltypes)])
    print('[Log] Counting neighbor diversity...')
    print(Counter(neighbor_celltype_count))
    # give a name
    neighbor_celltype_name = np.array(
        ['|'.join(sorted(list(n_ct.keys())))
         for n_ct in tqdm(neighbor_celltypes)]
    )

    # put to data
    adata.obs['neighbor_celltype_name'] = neighbor_celltype_name
    adata.obs['neighbor_celltype_count'] = neighbor_celltype_count
    adata.obs['STHD_pred_ct_raw'] = pmap

    print(
        '[Log] neighbor information in adata.obs columns: neighbor_celltype_name, neighbor_celltype_count, STHD_pred_ct_raw'
    )


def sthd_neighbor_ct_count_binned(adata):
    # obtain adjacency indices
    adj = adata.obsp['spatial_connectivities']
    adj_row = adj.indptr
    adj_col = adj.indices

    # count neighbour cell types
    neighbors = [
        model.csr_obtain_column_index_for_row(adj_row, adj_col, i)
        for i in range(len(adata))
    ]
    neighbor_celltypes = [
        Counter(adata.obs['STHD_pred_ct'].values[n]) for n in tqdm(neighbors)
    ]
    neighbor_celltype_count = np.array(
        [len(n_ct) for n_ct in tqdm(neighbor_celltypes)])
    print('[Log] Counting neighbor diversity...')
    print(Counter(neighbor_celltype_count))
    # give a name
    neighbor_celltype_name = np.array(
        ['|'.join(sorted(list(n_ct.keys())))
         for n_ct in tqdm(neighbor_celltypes)]
    )

    # put to data
    adata.obs['neighbor_celltype_name'] = neighbor_celltype_name
    adata.obs['neighbor_celltype_count'] = neighbor_celltype_count
    print(
        '[Log] neighbor information in adata.obs columns: neighbor_celltype_name, neighbor_celltype_count, STHD_pred_ct_raw'
    )


def get_frontline(
    adata, A='Tumor cE', B='Macrophage', frontline_name='frontline_ctA_ctB'
):
    # given cell types A,B (string containing pattern),
    itself_A_B = (adata.obs['STHD_pred_ct'].str.contains(A).values) | (
        adata.obs['STHD_pred_ct'].str.contains(B).values
    )
    subset = adata[(itself_A_B) & (
        adata.obs['neighbor_celltype_count'] <= 2)].copy()
    neighbor_A_B = (subset.obs['neighbor_celltype_name'].str.contains(A).values) & (
        subset.obs['neighbor_celltype_name'].str.contains(B).values
    )
    label = np.zeros(len(subset))  # initialize as all zeros

    # A
    label[
        (subset.obs['STHD_pred_ct'].str.contains(A).values)
        & (subset.obs['neighbor_celltype_count'] == 1)
    ] = 1
    # B
    label[
        (subset.obs['STHD_pred_ct'].str.contains(B).values)
        & (subset.obs['neighbor_celltype_count'] == 1)
    ] = 2
    # xA
    label[(subset.obs['STHD_pred_ct'].str.contains(A).values) & neighbor_A_B] = 3
    # xB
    label[(subset.obs['STHD_pred_ct'].str.contains(B).values) & neighbor_A_B] = 4
    print('[Log] Counting frontline spot... ')
    print(Counter(label))

    subset.obs['label'] = label

    adata.obs[frontline_name] = 0
    labels = adata.obs[frontline_name].values
    labels[
        np.where(adata.obs.index.isin(
            subset[subset.obs['label'] == 1].obs.index))[0]
    ] = 1
    labels[
        np.where(adata.obs.index.isin(
            subset[subset.obs['label'] == 2].obs.index))[0]
    ] = 2
    labels[
        np.where(adata.obs.index.isin(
            subset[subset.obs['label'] == 3].obs.index))[0]
    ] = 3
    labels[
        np.where(adata.obs.index.isin(
            subset[subset.obs['label'] == 4].obs.index))[0]
    ] = 4
    adata.obs[frontline_name] = labels
    print(
        f'frontline types in adata.obs["{frontline_name}"]',
        set(adata.obs[frontline_name]),
    )


def get_frontline_binned(
    adata, A='Tumor cE', B='Macrophage', frontline_name='frontline_ctA_ctB'
):
    # given cell types A,B (string containing pattern),
    itself_A_B = (adata.obs['STHD_pred_ct'].str.contains(A).values) | (
        adata.obs['STHD_pred_ct'].str.contains(B).values
    )
    subset = adata[(itself_A_B)].copy()
    neighbor_A_B = (subset.obs['neighbor_celltype_name'].str.contains(A).values) & (
        subset.obs['neighbor_celltype_name'].str.contains(B).values
    )
    label = np.zeros(len(subset))  # initialize as all zeros

    # A
    label[
        (subset.obs['STHD_pred_ct'].str.contains(A).values)
        & (subset.obs['neighbor_celltype_count'] == 1)
    ] = 1
    # B
    label[
        (subset.obs['STHD_pred_ct'].str.contains(B).values)
        & (subset.obs['neighbor_celltype_count'] == 1)
    ] = 2
    # xA
    label[(subset.obs['STHD_pred_ct'].str.contains(A).values) & neighbor_A_B] = 3
    # xB
    label[(subset.obs['STHD_pred_ct'].str.contains(B).values) & neighbor_A_B] = 4
    print('[Log] Counting frontline spot... ')
    print(Counter(label))

    subset.obs['label'] = label

    adata.obs[frontline_name] = 0
    labels = adata.obs[frontline_name].values
    labels[
        np.where(adata.obs.index.isin(
            subset[subset.obs['label'] == 1].obs.index))[0]
    ] = 1
    labels[
        np.where(adata.obs.index.isin(
            subset[subset.obs['label'] == 2].obs.index))[0]
    ] = 2
    labels[
        np.where(adata.obs.index.isin(
            subset[subset.obs['label'] == 3].obs.index))[0]
    ] = 3
    labels[
        np.where(adata.obs.index.isin(
            subset[subset.obs['label'] == 4].obs.index))[0]
    ] = 4
    adata.obs[frontline_name] = labels
    print(
        f'frontline types in adata.obs["{frontline_name}"]',
        set(adata.obs[frontline_name]),
    )


def frontline_summarize(adata, frontlines):
    # summarize the frontline type.
    adata.obs['frontline_sum_type'] = 'non_fl'
    labels = adata.obs['frontline_sum_type'].values
    for frontline in frontlines:
        labels[adata.obs[frontline].isin([3, 4])] = frontline
    adata.obs['frontline_sum_type'] = labels

    adata.obs['frontline_sum_type_AB'] = 'non_fl'
    labels = adata.obs['frontline_sum_type_AB'].values
    for frontline in frontlines:
        labels[adata.obs[frontline] == 3] = f'{frontline}_A'
        labels[adata.obs[frontline] == 4] = f'{frontline}_B'
    adata.obs['frontline_sum_type_AB'] = labels
    print(
        '[Log] Summarized frontline type in frontline_sum_type, and frontline_sum_type_AB'
    )


######## Distance to front lines ############
@njit(parallel=True, fastmath=True)
def min_pairwise_distance(X, Y):
    # X and Y represents 2 sets of 2D dots.
    # For each dot in X, find the cloest dot in Y and calculate their euclidian distance
    n, m = len(X), len(Y)
    res = np.zeros(n, dtype=np.float32)
    for i in prange(n):
        x = X[i]
        d = np.inf
        for j in range(m):
            y = Y[j]
            cur_d = ((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2) ** (0.5)
            d = min(d, cur_d)
        res[i] = d
    return res


def calculate_distance(adata, frontline_name):
    # assuming, label of 3 and 4 indicates xA and xB, which is the frontline.
    frontline_label = np.isin(adata.obs[frontline_name].values, [3, 4])
    # location = adata.obs[['x','y']].values
    location = adata.obs[['array_row', 'array_col']].values
    frontline_location = location[frontline_label]
    distance = min_pairwise_distance(location, frontline_location)
    adata.obs[f'dTo_{frontline_name}'] = distance
    print(
        f'Distance to front line {frontline_name} is saved in adata.obs["dTo_{frontline_name}"]'
    )


def calculate_distance_binned(adata, frontline_name):
    # assuming, label of 3 and 4 indicates xA and xB, which is the frontline.
    frontline_label = np.isin(adata.obs[frontline_name].values, [3, 4])
    # location = adata.obs[['x','y']].values
    location = adata.obs[['bin_row', 'bin_col']].values.astype(int)
    frontline_location = location[frontline_label]
    distance = min_pairwise_distance(location, frontline_location)
    adata.obs[f'dTo_{frontline_name}'] = distance
    print(
        f'Distance to front line {frontline_name} is saved in adata.obs["dTo_{frontline_name}"]'
    )
