"""Plot Results as Interactive HTML via The STHD Visualization Module.

Usage:
python3 sthdviz.py \
    --patch_path ../analysis/full_patchify \
    --title full_patchify
"""

import argparse

import numpy as np
from bokeh.layouts import column
from bokeh.models import (
    BoxZoomTool,
    CategoricalColorMapper,
    ColorBar,
    CustomJS,
    CustomJSHover,
    CustomJSTickFormatter,
    FixedTicker,
    HoverTool,
    LinearColorMapper,
    LogColorMapper,
    MultiChoice,
    WheelZoomTool,
)
from bokeh.models.labeling import AllLabels
from bokeh.plotting import figure, output_file, show
from tqdm import tqdm

from STHD import color_palette, train


def rasterize(df, val_col):
    """Rasterize the input dataframe with categorical values
    df: input dataframe to rasterize
        array_row: index for row
        array_col: index for col
        {val_col}: the cell value (usually it is cell type prediction label)
    """
    df = df.dropna()
    row_min, row_max = df['array_row'].min(), df['array_row'].max()
    col_min, col_max = df['array_col'].min(), df['array_col'].max()
    val_char_len = df[val_col].str.len().max()
    print(row_min, row_max, len(set(df['array_row'].values)))
    print(col_min, col_max, len(set(df['array_col'].values)))

    row_ids = (df['array_row'].values - row_min).astype(int)
    col_ids = (df['array_col'].values - col_min).astype(int)
    vals = df[val_col].values

    nrow = max(row_ids) + 1
    ncol = max(col_ids) + 1

    res = np.array([[''] * ncol] * nrow, dtype=f'<U{val_char_len}')

    for i, val in tqdm(enumerate(vals), total=len(vals)):
        res[row_ids[i]][col_ids[i]] = val
    return res


def rasterize_numerical(df, val_col):
    """Rasterize the input dataframe with numerical values
    df: input dataframe to rasterize
        array_row: index for row
        array_col: index for col
        {val_col}: the cell value (usually it is cell type prediction label)
    """
    df = df.dropna()
    row_min, row_max = df['array_row'].min(), df['array_row'].max()
    col_min, col_max = df['array_col'].min(), df['array_col'].max()
    print(row_min, row_max, len(set(df['array_row'].values)))
    print(col_min, col_max, len(set(df['array_col'].values)))

    row_ids = (df['array_row'].values - row_min).astype(int)
    col_ids = (df['array_col'].values - col_min).astype(int)
    vals = df[val_col].values

    nrow = max(row_ids) + 1
    ncol = max(col_ids) + 1

    res = np.zeros((nrow, ncol), dtype=np.float32)

    for i, val in tqdm(enumerate(vals), total=len(vals)):
        res[row_ids[i]][col_ids[i]] = val
    return res


def convert_numerical(df, cmap):
    """Convert cell types into numerical representations using two digits of hex numbers."""
    if len(cmap) > 254:
        raise ValueError('this function can only process up to 254 cell types')

    hex_dict = dict()
    num_dict = dict()
    for i, cell_type in enumerate(cmap):
        hex_dict[cell_type] = hex(i)[2:]
        num_dict[cell_type] = str(i)
    hex_dict[''] = hex(len(cmap))[2:]
    num_dict[''] = str(len(cmap))

    df_hex = np.empty(df.shape, dtype='<U2')
    for i in tqdm(
        range(df.shape[0]),
        total=len(df),
        desc='convert cell types to numerical representations',
    ):
        for j in range(df.shape[1]):
            df_hex[i][j] = hex_dict[df[i][j]]
    return df_hex, hex_dict, num_dict


def fast_plot(
    rastered_df,
    cmap=color_palette.get_config_colormap(name='colormap_coloncatlas_98'),
    cmap_light=color_palette.get_config_colormap(
        name='colormap_coloncatlas_98_light'),
    cmap_dark=color_palette.get_config_colormap(
        name='colormap_coloncatlas_98_dark'),
    title='STHD_visualization',
    square_size=7,
    save_root_dir='',
):
    """Interactive plotting by rasterizing the scatter plot into an image.

    rastered_df: generated by the rasterize function above.
    """
    # --------------------------- Save Location ----------------------------------
    if len(save_root_dir) > 0:
        output_file(filename=f'{save_root_dir}/{title}.html', title=title)

    # ---------------------------Numerical representation of input df------------
    rastered_df_hex, ct_hex_map, ct_num_map = convert_numerical(
        rastered_df, cmap)
    cmap_hex = {ct_hex_map[cell_type]: cmap[cell_type] for cell_type in cmap}
    cmap_hex_light = {
        ct_hex_map[cell_type]: cmap_light[cell_type] for cell_type in cmap_light
    }
    cmap_hex_dark = {
        ct_hex_map[cell_type]: cmap_dark[cell_type] for cell_type in cmap_dark
    }
    hex_ct_map = {ct_hex_map[ct]: ct for ct in ct_hex_map}
    num_ct_map = {ct_num_map[ct]: ct for ct in ct_num_map}

    # -------------------------- prepare colormap -----------------------------------
    cell_type_hex = list(cmap_hex.keys())
    color_list = list(cmap_hex.values())

    mapper = CategoricalColorMapper(factors=cell_type_hex, palette=color_list)
    color_key_pos = dict(
        zip(cell_type_hex, list(zip(range(len(cmap_hex)), list(color_list))))
    )

    ticker = FixedTicker(ticks=np.arange(len(num_ct_map)))
    formatter = CustomJSTickFormatter(
        args={'num_ct_map': num_ct_map}, code='return num_ct_map[tick]'
    )

    color_bar = ColorBar(
        color_mapper=mapper,
        ticker=ticker,
        formatter=formatter,
        major_label_text_font_size='5px',
        major_label_policy=AllLabels(),
        label_standoff=6,
        border_line_color=None,
    )

    # -------------------------- Prepare figure ------------------------------------
    TOOLS = 'save,pan,reset,wheel_zoom'
    p = figure(
        title=title,
        x_axis_location='above',
        frame_width=1700,
        frame_height=800,
        match_aspect=True,
        tools=TOOLS,
        toolbar_location='above',
    )
    p.toolbar.active_scroll = p.select_one(WheelZoomTool)
    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = '7px'
    p.axis.major_label_standoff = 0
    p.title.align = 'center'

    p.add_tools(BoxZoomTool(match_aspect=True))
    p.add_tools(
        HoverTool(
            tooltips=[('class', '@image{custom}')],
            formatters={
                '@image': CustomJSHover(
                    code='return hex_ct_map[value];', args={'hex_ct_map': hex_ct_map}
                )
            },
        )
    )

    # ---------------------------Draw image -------------------------------------------
    img = p.image(
        image=[rastered_df_hex],
        x=0,
        y=0,
        dw=rastered_df_hex.shape[1],
        dh=rastered_df_hex.shape[0],
        color_mapper=mapper,
    )

    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = '7px'
    p.axis.major_label_standoff = 0
    p.add_layout(color_bar, 'right')
    p.y_range.flipped = False

    # ----------------------Multiple Choice module ------------------------------------------------
    # multi choice
    OPTIONS = sorted(list((cmap.keys())))
    multi_choice = MultiChoice(
        value=[], options=OPTIONS, title='Selection:', min_width=300
    )
    callback = CustomJS(
        args=dict(
            img=img,
            color_key_pos=color_key_pos,
            cmap_light=cmap_hex_light,
            cmap_dark=cmap_hex_dark,
            ct_hex_map=ct_hex_map,
        ),
        code="""
    // reset colors
    const length = this.value.length;
    console.log(length);
    if (length===0) { // if no cell type is selected, use default colors
        for (const ct_hex in color_key_pos) {
            img.glyph.color_mapper.palette[color_key_pos[ct_hex][0]] = color_key_pos[ct_hex][1];
        }
    } else { // if there are cell type selected, set all cell types to have ligth colors (and later highlight selected cell types)
        for (const ct_hex in color_key_pos) {
            img.glyph.color_mapper.palette[color_key_pos[ct_hex][0]] = cmap_light[ct_hex];
        }
    }

    // highlight selected genes
    for (const ct of this.value) {
        const ct_hex = ct_hex_map[ct];
        img.glyph.color_mapper.palette[color_key_pos[ct_hex][0]] = cmap_dark[ct_hex];
    }
    
    // apply changes and update plot
    img.glyph.change.emit();
    """,
    )
    multi_choice.js_on_change('value', callback)
    show(column(multi_choice, p))


def fast_plot_numerical(
    rastered_df,
    title='STHD_visualization',
    square_size=7,
    save_root_dir='',
    mapper_type='linear',
    palette='Viridis256',
):
    """Interactive plotting by rasterizing the scatter plot into an image.

    rastered_df: generated by the rasterize function above.
    """
    if mapper_type == 'log':
        cls = LogColorMapper
        if rastered_df.min() < 0:
            raise ValueError(
                'Cannot us log scale colormap since there are negative values in rastered_df'
            )
        min_val = np.min(
            rastered_df[np.nonzero(rastered_df)]
        )  # get non-zero minimum values
        if min_val > 1:
            rastered_df += 1  # in log scale, the minimum value is still zero
        else:
            rastered_df += min_val * 0.1
    else:
        cls = LinearColorMapper

    mapper = cls(palette=palette, low=rastered_df.min(),
                 high=rastered_df.max())

    if len(save_root_dir) > 0:
        output_file(filename=f'{save_root_dir}/{title}.html', title=title)

    TOOLS = 'hover,save,pan,reset,wheel_zoom'
    p = figure(
        title=title,
        x_axis_location='above',
        frame_width=1700,
        frame_height=800,
        match_aspect=True,
        tools=TOOLS,
        toolbar_location='above',
        tooltips=[('x', '$x{0}'), ('y', '$y{0}'), ('class', '@image')],
    )
    p.add_tools(BoxZoomTool(match_aspect=True))
    p.toolbar.active_scroll = p.select_one(WheelZoomTool)
    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = '7px'
    p.axis.major_label_standoff = 0
    p.title.align = 'center'

    _ = p.image(
        image=[rastered_df],
        x=0,
        y=0,
        dw=rastered_df.shape[1],
        dh=rastered_df.shape[0],
        color_mapper=mapper,
    )

    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = '7px'
    p.axis.major_label_standoff = 0

    color_bar = ColorBar(
        color_mapper=mapper,
        major_label_text_font_size='3px',
        label_standoff=6,
        border_line_color=None,
    )
    p.add_layout(color_bar, 'right')
    p.y_range.flipped = False

    show(p)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_path', type=str, required=True)
    parser.add_argument('--title', default='sthd_pred_cell_type', type=str)
    args = parser.parse_args()

    sthdata = train.load_data_with_pdata(args.patch_path)
    df = sthdata.adata.obs[['array_row', 'array_col', 'STHD_pred_ct']]
    df_rasterize = rasterize(df, 'STHD_pred_ct')
    fast_plot(df_rasterize, title=args.title, save_root_dir=args.patch_path)
