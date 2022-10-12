import os.path
import matplotlib.font_manager as font_manager
import matplotlib as mpl

import numpy as np
import pandas as pd

def set_font():
    """
    Code from A. Sauerburger's Atlasify package
    Set sensible default values for matplotlib mainly concerning fonts.
    """
    # Add internal font directory
    font_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "fonts")
    font_files = font_manager.findSystemFonts(fontpaths=[font_dir])

    new_fonts = []
    try:
        # For mpl 3.2 and higher
        for font_file in font_files:
            # Resemble add_font
            font = mpl.ft2font.FT2Font(font_file)
            prop = font_manager.ttfFontProperty(font)
            new_fonts.append(prop)
    except AttributeError:
        # Legacy
        # pylint: disable=no-member
        new_fonts = font_manager.createFontList(font_files)

    # Give precedence to fonts shipped in this package
    font_manager.fontManager.ttflist = new_fonts \
                                       + font_manager.fontManager.ttflist

    # Change font
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Helvetica', 'Nimbus Sans',
                                       'Nimbus Sans L', 'Arial']

# integral in range
def get_integral(p_vals,p_bins,p_binval_low=-99999.,p_binval_high=99999.):
    index=0
    integral=0.
    for bin in p_bins:
        if (bin>=p_binval_low and bin<p_binval_high and index<len(p_vals)):
            integral+=p_vals[index]
        index+=1
    return(integral)


# get weights for unit normalization of array:
def get_w(p_arr,p_weights=pd.Series([])):
    sum_weights=float(len(p_arr))
    numerators = np.ones_like(p_arr)
    if (0!=len(p_weights.index)):
        sum_weights=np.sum(p_weights)
        numerators = p_weights
    return(numerators/sum_weights)
