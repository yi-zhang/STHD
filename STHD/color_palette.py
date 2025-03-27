"""Generate Color Palettes w.r.t. input names. 
"""

import colorsys
import matplotlib
from STHD import config


def _divide_hue(hue_range, n, hue_shift):
    hue_delta = (hue_range[1] - hue_range[0]) / (n - 1)
    res = []
    for i in range(n):
        res.append(hue_range[0] + i * hue_delta + hue_shift)
    return res


def _construct_color(hue_list, saturation, lightness, color_format):
    res = []
    for hue in hue_list:
        r, g, b = colorsys.hls_to_rgb(hue, saturation, lightness)
        if color_format == 'hex':
            color_code = '#{:02x}{:02x}{:02x}'.format(
                int(r * 255), int(g * 255), int(b * 255)
            )
        elif color_format == 'rgb':
            color_code = (r, g, b)
        res.append(color_code)
    return res


def _generate_color_palette(
    names,
    hue_range,
    saturation,
    lightness,
    hue_shift=0,
    color_format='rgb',  # either 'rgb' or 'hex'
):
    hues = _divide_hue(hue_range, len(names), hue_shift)
    colors = _construct_color(hues, saturation, lightness, color_format)

    res = dict(zip(names, colors))
    return res


def get_color_map_1(genemeanpd_filtered):
    # define colormap
    names = [f'p_ct_{i}' for i in list(genemeanpd_filtered.columns)]
    tumor_colors = _generate_color_palette(
        names=names[:11],
        hue_range=[0, 0.2],
        saturation=0.8,
        lightness=1,
        hue_shift=0,
        color_format='hex',
    )
    normal_colors = _generate_color_palette(
        names=names[11:],
        hue_range=[0.25, 0.8],
        saturation=0.7,
        lightness=0.7,
        hue_shift=0,
        color_format='hex',
    )
    cmap = {**tumor_colors, **normal_colors}
    cmap['ambiguous'] = [0.65] * 3
    return cmap


def prepare_palette(cmap, adata, ctcol='STHD_pred_ct'):
    data_cmap = []
    for p in sorted(list(set(adata.obs[ctcol]))):
        cur_color = cmap[p]
        data_cmap.append(cur_color)
    palette = matplotlib.colors.ListedColormap(data_cmap)
    return palette


def get_config_colormap(name='colormap_coloncatlas_98'):
    cmap = None

    if name == 'colormap_coloncatlas_98':
        cmap = config.colormap_coloncatlas_98
    elif name == 'colormap_coloncatlas_98_light':
        cmap = config.colormap_coloncatlas_98_light
    elif name == 'colormap_coloncatlas_98_dark':
        cmap = config.colormap_coloncatlas_98_dark
    elif name == 'colormap_crc98_ct_group':
        cmap = config.colormap_crc98_ct_group
    return cmap


def adjust_lightness(hex_color, adjustment):
    """Adjust the lightness of a hex color.

    Args:
    ----
        hex_color (str): The hex color code (e.g., '#RRGGBB').
        adjustment (float): The amount by which to adjust the lightness.
                            Positive values make the color lighter, negative values make it darker.
                            Should be between -1 and 1.

    Returns:
    -------
        str: The adjusted hex color code.

    """
    # Remove '#' from the hex color string
    hex_color = hex_color.lstrip('#')

    # Convert the hex color to RGB values
    red = int(hex_color[0:2], 16)
    green = int(hex_color[2:4], 16)
    blue = int(hex_color[4:6], 16)

    # Adjust the lightness
    red = min(255, max(0, int(red + (255 - red) * adjustment)))
    green = min(255, max(0, int(green + (255 - green) * adjustment)))
    blue = min(255, max(0, int(blue + (255 - blue) * adjustment)))

    # Convert the adjusted RGB values back to hex
    adjusted_hex_color = f'#{red:02x}{green:02x}{blue:02x}'

    return adjusted_hex_color
