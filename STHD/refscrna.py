import pandas as pd


def load_scrna_ref(refile):
    genemeanpd_filtered = pd.read_table(refile, index_col=0)
    return genemeanpd_filtered

def gene_lambda_by_ct(adata, ctcol = 'group'):
    pdlst=[]
    #celltypes_group = celltypes.groupby(ctcol).count().index.tolist()
    celltypes_group = list(set(adata.obs[ctcol].values))
    for ct in celltypes_group:
        ctbars = adata[adata.obs[ctcol] == ct].obs.index.tolist()
        print(f'[Log] {len(ctbars)} cells for cell type {ct}')
        cellbygenecount = adata.layers['counts'][adata.obs.index.isin(ctbars), :]
        print(f'dimension of cell by gene count', cellbygenecount.shape)
        cellbygene = pd.DataFrame(cellbygenecount.todense(), index = ctbars, columns = adata.var_names)
        genebycell = cellbygene.T
        # gene normalize expr per cell (gene count ratio over all gene count), then mean over all cells from that type.
        genenorm = (genebycell/(genebycell.sum(axis=0)))
        genenorm = genenorm.mean(axis=1) # normalized gene expr for this cell type.
        pdlst.append( 
            pd.DataFrame(genenorm, columns = [ct], index = adata.var_names)
        )

    genemeanpd = pd.concat(pdlst,axis=1)
    genemeanpdfc = genemeanpd/genemeanpd.mean()
    return(genemeanpd, genemeanpdfc)

def select_ct_informative_genes_basic(genemeanpd, genemeanpdfc, exprcut = 0.000125, fccut = 2**0.5):
    mask_nomt = (~genemeanpd.index.str.startswith('MT-'))
    mask_noribo = (~genemeanpd.index.str.startswith(('RPS','RPL'))) 
    print(f'{ (mask_nomt&mask_noribo).sum() } genes after removing MT- and Ribo')
    # basic expression
    mask_expr_rctd = ( (genemeanpd>exprcut).sum(axis=1) )>0 

    print(f'{ (mask_nomt&mask_noribo&mask_expr_rctd).sum() } genes after further removing RCTD threshold gene expression 0.0125')
    ## log fold change over all cells
    mask_fc = (genemeanpdfc> fccut).sum(axis=1)>0 # rctd logfc
    print(f'{ (mask_nomt&mask_noribo&mask_expr_rctd&mask_fc).sum() } genes after further removing RCTD logfc log2 0.5')
    ## extra basic mean expr


    genemeanpd_filtered = genemeanpd[mask_nomt&mask_noribo&mask_expr_rctd&mask_fc]
    print(genemeanpd_filtered.shape)
    return(genemeanpd_filtered)
