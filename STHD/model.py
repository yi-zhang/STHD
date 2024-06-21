import numpy as np
import squidpy as sq
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def fill_F(X, Y, Z, N, Lambda, D, F):
    for a in prange(X):
        for t in range(Z):
            for g in range(Y):
                tmp = D[a] * Lambda[t, g]
                F[a, t] = F[a, t] + (N[a, g] * np.log(tmp) - tmp)


def prepare_constants(sthd_data):
    X = sthd_data.adata.obs.shape[0]  # n of spot
    Y = sthd_data.adata.shape[1]  # n of gene (filtered )
    Z = sthd_data.lambda_cell_type_by_gene_matrix.shape[0]  # n of cell type

    # get raw data
    N = sthd_data.adata.to_df().values  # [X,Y] (a,g) number of total reads in each spot
    Lambda = sthd_data.lambda_cell_type_by_gene_matrix.astype(
        "float32"
    )  # [Z,Y] (t,g) gene's relative expression in each cell type.
    D = (
        np.squeeze(np.asarray(sthd_data.adata.X.sum(axis=1))) + 0.1
    )  # [X] (a) spot depth. adding 0.1 to prevent explod

    # get neighbor adjacency matrix
    sq.gr.spatial_neighbors(
        sthd_data.adata, spatial_key="spatial", coord_type="grid", n_neighs=4, n_rings=2
    )
    A_csr = sthd_data.adata.obsp["spatial_connectivities"]  # [X, X]
    print("Currently we only support symmetric adjacency matrix of neighbors")
    Acsr_row = A_csr.indptr
    Acsr_col = A_csr.indices

    F = np.zeros([X, Z], dtype="float32")
    fill_F(X, Y, Z, N, Lambda, D, F)
    return X, Y, Z, F, Acsr_row, Acsr_col


def prepare_training_weights(X, Y, Z):
    # prepare training parameters
    W = np.ones([X, Z]).astype("float32")  # initialization
    # prepare derived parameters
    eW = np.zeros([X, Z], dtype="float32")
    P = np.zeros([X, Z], dtype="float32")
    Phi = np.zeros([X], dtype="float32")
    # prepare deriatives
    ll_wat = np.zeros([X, Z], dtype="float32")
    ce_wat = np.zeros([X, Z], dtype="float32")
    # additional variables for adam
    m = np.zeros([X, Z], dtype="float32")
    v = np.zeros([X, Z], dtype="float32")
    return W, eW, P, Phi, ll_wat, ce_wat, m, v


def early_stop_criteria(metrics, beta, n=5, threshold=0.01):
    if len(metrics) < n:
        return False
    else:
        metrics_check = metrics[-n:]
        ll = [i[0] for i in metrics_check]
        ll_range = max(ll) - min(ll)
        ll_ratio = ll_range / np.abs(ll[-1])
        if beta == 0:
            # in this case we won't check ce for early stopping
            ce_ratio = 0
        else:
            ce = [i[1] for i in metrics_check]
            ce_range = max(ce) - min(ce)
            ce_ratio = ce_range / np.abs(ce[-1])
        if (ll_ratio < threshold) and (ce_ratio < threshold):
            return True
        else:
            return False


def early_stop_criteria_2(metrics, beta, n=10, threshold=0.01):
    if len(metrics) < n:
        return False
    else:
        metrics_check = metrics[-n:]
        loss = [-i[0] + beta * i[1] for i in metrics_check]
        loss_range = max(loss) - min(loss)
        loss_ratio = loss_range / np.abs(loss[-1])
        if loss_ratio < threshold:
            return True
        else:
            return False


def train(
    n_iter,
    step_size,
    beta,
    constants,
    weights,
    early_stop=False,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
):
    X, Y, Z, F, Acsr_row, Acsr_col = constants
    W, eW, P, Phi, ll_wat, ce_wat, m, v = weights

    metrics = []
    for i in range(n_iter):
        update_eW(eW, W, X, Y, Z)
        update_Phi(Phi, eW, X, Y, Z)
        update_P(P, eW, Phi, X, Y, Z)
        update_ll_wat(ll_wat, P, F, X, Y, Z)
        update_ce_wat(ce_wat, P, Acsr_row, Acsr_col, X, Y, Z)
        update_m_v(m, v, beta1, beta2, beta, ll_wat, ce_wat, X, Y, Z)
        update_W2(W, m, v, beta1, beta2, i + 1, step_size, epsilon, X, Y, Z)
        ll = calculate_ll(P, F, X, Y, Z)
        ce = calculate_ce(P, Acsr_row, Acsr_col, X, Y, Z)
        metrics.append((ll, ce))
        print(i, -ll + beta * ce, ll, ce)

        if early_stop and early_stop_criteria_2(metrics, beta):
            return metrics

    return metrics

@njit(parallel=True, fastmath=True)
def update_eW(eW, W, X, Y, Z):
    # inplace update eW using W
    for a in prange(X):
        for t in range(Z):
            eW[a, t] = np.exp(W[a, t])


@njit(parallel=True, fastmath=True)
def update_Phi(Phi, eW, X, Y, Z):
    # inplace update Phi using eW
    for a in prange(X):
        Phi[a] = 0
        for t in range(Z):
            Phi[a] = Phi[a] + eW[a, t]


@njit(parallel=True, fastmath=True)
def update_P(P, eW, Phi, X, Y, Z):
    # inplace update P using eW, Phi
    for a in prange(X):
        for t in range(Z):
            P[a, t] = eW[a, t] / Phi[a]

@njit
def csr_obtain_column_index_for_row(row, column, i):
    # get row i's non-zero items' column index
    row_start = row[i]
    row_end = row[i + 1]
    column_indices = column[row_start:row_end]
    return column_indices


@njit
def csr_obtain_value_by_i_j(row, column, value, i, j):
    # get row i's non-zero items' column index
    row_start = row[i]
    row_end = row[i + 1]
    column_indices = column[row_start:row_end]
    row_values = value[row_start:row_end]
    if j in set(column_indices):
        for k, cid in enumerate(column_indices):
            if cid == j:
                v = row_values[k]
                break
    else:
        v = 0
    return v


@njit
def calculate_ce(P, Acsr_row, Acsr_col, X, Y, Z):
    # ! DO NOT use numba parallel in this function!!
    res = 0
    for a in range(X):
        neighbors = csr_obtain_column_index_for_row(Acsr_row, Acsr_col, a)
        for t in range(Z):
            cur = 0
            for a_star in neighbors:
                cur = cur + np.log(P[a_star, t])
            res = res - P[a, t] * cur
    res = res / X
    return res


@njit(parallel=True, fastmath=True)
def calculate_ll(P, F, X, Y, Z):
    res = 0
    for a in prange(X):
        for t in range(Z):
            res = res + P[a, t] * F[a, t]
    res = res / X
    return res


@njit(parallel=True, fastmath=True)
def update_ll_wat(ll_wat, P, F, X, Y, Z):
    for a_tilda in prange(X):
        for t_tilda in range(Z):
            cur = 0
            for t in range(Z):
                cur = cur + P[a_tilda, t] * P[a_tilda, t_tilda] * F[a_tilda, t]
            ll_wat[a_tilda, t_tilda] = (
                -1 * cur + P[a_tilda, t_tilda] * F[a_tilda, t_tilda]
            )
            ll_wat[a_tilda, t_tilda] = ll_wat[a_tilda, t_tilda] / X


@njit(parallel=True, fastmath=True)
def update_ce_wat(ce_wat, P, Acsr_row, Acsr_col, X, Y, Z):
    for a_tilda in prange(X):
        neighbors = csr_obtain_column_index_for_row(Acsr_row, Acsr_col, a_tilda)
        # Since Acsr is symetric, neighbors of a_tilda is also the spots whose neighbor contains a_tilda
        for t_tilda in range(Z):
            comp1 = 0  # neighbors of a_tilda
            comp2 = 0  # a_tilda is other's neighbo
            for a_star in neighbors:
                cur1 = 0
                for t in range(Z):
                    cur1 = cur1 + P[a_tilda, t] * np.log(P[a_star, t])
                comp1 = comp1 + np.log(P[a_star, t_tilda]) - cur1
                comp2 = comp2 + P[a_star, t_tilda] - P[a_tilda, t_tilda]
            ce_wat[a_tilda, t_tilda] = (-P[a_tilda, t_tilda] * comp1 - comp2) / X

@njit(parallel=True, fastmath=True)
def update_W1(W, ll_wat, ce_wat, alpha, beta, X, Y, Z):
    for a in prange(X):
        for t in range(Z):
            W[a, t] = W[a, t] - alpha * (-ll_wat[a, t] + beta * ce_wat[a, t])

@njit(parallel=True, fastmath=True)
def update_m_v(m, v, beta1, beta2, beta, ll_wat, ce_wat, X, Y, Z):
    for a in prange(X):
        for t in range(Z):
            cur_gredient = -ll_wat[a, t] + beta * ce_wat[a, t]
            m[a, t] = m[a, t] * beta1 + (1 - beta1) * cur_gredient
            v[a, t] = v[a, t] * beta2 + (1 - beta2) * cur_gredient**2


@njit(parallel=True, fastmath=True)
def update_W2(W, m, v, beta1, beta2, i, alpha, epsilon, X, Y, Z):
    for a in prange(X):
        for t in range(Z):
            m_correct = m[a, t] / (1 - beta1**i)
            v_correct = v[a, t] / (1 - beta2**i)
            W[a, t] = W[a, t] - alpha * m_correct / (v_correct**0.5 + epsilon)
