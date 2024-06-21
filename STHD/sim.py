import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def simulate_spot_bin_2cell(
    side_length=15,
    num_points=16,
    center1=(4, 4),
    radius1=4,
    center2=(9, 9),
    radius2=4,
):
    ########## 2um spots ##########
    # Define the size of the square region
    # side_length = 15
    # Define the number of grid points along each axis
    # num_points = 16  # You can adjust this to change the density of the grid

    # Generate grid points within the square region
    # x = np.linspace(-side_length / 2, side_length / 2, num_points)
    x = np.linspace(0, side_length, num_points)
    y = np.linspace(0, side_length, num_points)
    # y = np.linspace(-side_length / 2, side_length / 2, num_points)
    x_grid, y_grid = np.meshgrid(x, y)

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x_grid, y_grid, color="blue")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    # plt.xlim(-side_length / 2-1, side_length / 2+1)
    # plt.xlim(0-1, side_length+1)
    # plt.ylim(-side_length / 2-1, side_length / 2+1)
    # plt.ylim(0-1, side_length+1)
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.grid(False)

    ########## 8um bins ##########
    ## add bin boundary
    lst = x + 0.5
    N = 4
    binxs = lst[N - 1 :: N]
    for xt in binxs:
        ax.axvline(xt, color="gray")

    lst = y + 0.5
    binys = lst[N - 1 :: N]
    N = 4
    for yt in binys:
        ax.axhline(yt, color="gray")

    ######## cell mask ##########

    # center1 = (4, 4)
    # radius1 = 4
    mask1 = ((x_grid - center1[0]) ** 2 + (y_grid - center1[1]) ** 2) <= radius1**2

    # center2 = (9, 9)
    # radius2 =4
    mask2 = ((x_grid - center2[0]) ** 2 + (y_grid - center2[1]) ** 2) <= radius2**2
    # remove mask1 in mask2, to make sure per spot is from one cell.
    mask_overlap = mask1 & mask2
    mask1[mask_overlap] = False

    ax.scatter(x_grid[mask1], y_grid[mask1], color="orange")
    ax.scatter(x_grid[mask2], y_grid[mask2], color="green")
    plt.show()
    return (x, y, x_grid, y_grid, binxs, binys, mask1, mask2)


def simulate_spot_expr_2cell(
    x_grid,
    y_grid,
    mask1,
    mask2,
    lam_ct1geneA=2,
    lam_ct1geneB=0,
    lam_ct2geneA=0.01,
    lam_ct2geneB=4,
    lam_geneC=4,
    lam_noise=0.001,
):
    ## create an empty expr adata
    total_spot_num = x_grid.shape[0] * x_grid.shape[1]
    spotid_lst = [t for t in range(total_spot_num)]
    barcode_lst = ["barcode_" + str(t) for t in range(total_spot_num)]

    obs = pd.DataFrame(
        {"cellid": spotid_lst},
        index=barcode_lst,
    )
    var = pd.DataFrame(index=["geneA", "geneB", "geneC"])
    expr = pd.DataFrame(np.zeros([obs.shape[0], var.shape[0]]))
    expr.index = obs.index
    expr.columns = var.index

    # add true cell info
    iscell1_i = np.array(spotid_lst)[(mask1).flatten()]
    iscell2_i = np.array(spotid_lst)[(mask2).flatten()]

    obs["celltype"] = ""
    obs["iscell"] = False
    obs.loc[obs.iloc[iscell1_i].index, "celltype"] = "ct1"
    obs.loc[obs.iloc[iscell2_i].index, "celltype"] = "ct2"
    obs.loc[obs.iloc[iscell1_i].index, "iscell"] = True

    ## location of each spot
    obs["x"] = x_grid.flatten()
    obs["y"] = y_grid.flatten()
    obsm = {"spatial": obs[["x", "y"]].values}

    # simulate expression
    ct1_ids = obs[obs["celltype"] == "ct1"].index
    ct2_ids = obs[obs["celltype"] == "ct2"].index

    ct1geneA = np.random.poisson(lam=lam_ct1geneA, size=[len(ct1_ids)])
    ct1geneB = np.random.poisson(lam=lam_ct1geneB, size=[len(ct1_ids)])
    ct2geneA = np.random.poisson(lam=lam_ct2geneA, size=[len(ct2_ids)])
    ct2geneB = np.random.poisson(lam=lam_ct2geneB, size=[len(ct2_ids)])
    # print(ct1geneA)
    # print( expr.loc[ct1_ids]['geneA'].values)
    expr.loc[ct1_ids, "geneA"] = expr.loc[ct1_ids]["geneA"].values + ct1geneA
    # print( expr.loc[ct1_ids]['geneA'].values)
    expr.loc[ct1_ids, "geneB"] = expr.loc[ct1_ids]["geneB"].values + ct1geneB
    expr.loc[ct2_ids, "geneA"] = expr.loc[ct2_ids]["geneA"].values + ct2geneA
    expr.loc[ct2_ids, "geneB"] = expr.loc[ct2_ids]["geneB"].values + ct2geneB

    # simulate background noise
    geneN = np.random.poisson(lam=lam_noise, size=obs.shape[0])
    geneC = np.random.poisson(lam=lam_geneC, size=obs.shape[0])
    expr["geneA"] = expr["geneA"].values + geneN
    expr["geneB"] = expr["geneB"].values + geneN
    # add a 3rd unrelated gene
    expr.loc[:, "geneC"] = geneC

    adata = anndata.AnnData(expr, obs=obs, var=var, obsm=obsm)

    return adata
