import matplotlib.pyplot as plt
import matplotlib as mpl

# Color blind friendly palettes
COLOR_BLIND_FRIENDLY = {
    'main': [
        "#0077BB",  # Blue
        "#EE7733",  # Orange
        "#009988",  # Teal
        "#CC3311",  # Red
        "#33BBEE",  # Cyan
        "#EE3377",  # Magenta
        "#BBBBBB",  # Grey
        "#000000"   # Black
    ],
    # Alternative palette (Wong, 2011)
    'wong': [
        '#E69F00',  # Orange
        '#56B4E9',  # Sky Blue
        '#009E73',  # Green
        '#F0E442',  # Yellow
        '#0072B2',  # Blue
        '#D55E00',  # Vermillion
        '#CC79A7',  # Pink
        '#000000'   # Black
    ]
}

# Line styles for different plot elements
LINE_STYLES = ['-', '--', ':', '-.', (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (3, 1, 1, 1, 1, 1))]

def set_default_params():
    """Set default matplotlib parameters for consistent plotting."""
    plt.style.use('default')  # Reset to default style
    
    params = {
        # Figure
        'figure.figsize': (8, 6),
        'figure.dpi': 300,
        
        # Font settings
        'font.size': 14,
        'font.family': 'serif',
        'text.usetex': True,
        
        # Lines
        'lines.linewidth': 2,
        'lines.markersize': 8,
        
        # Axes
        'axes.linewidth': 1.5,
        'axes.labelsize': 18,
        'axes.titlesize': 18,
        'axes.grid': True,
        
        # Grid
        'grid.alpha': 0.7,
        'grid.linestyle': '--',
        'grid.color': '#CCCCCC',
        
        # Ticks
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.minor.size': 3,
        'ytick.minor.size': 3,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.minor.width': 1,
        'ytick.minor.width': 1,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        
        # Legend
        'legend.fontsize': 12,
        'legend.frameon': True,
        'legend.facecolor': 'white',
        'legend.framealpha': 1.0,
        
        # Save settings
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    }
    
    plt.rcParams.update(params)

def set_thick_borders(ax):
    """Set thicker borders for given axis."""
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

def create_figure(nrows=1, ncols=1, height_ratios=None, width_ratios=None, **kwargs):
    """
    Create a figure with GridSpec.
    
    Parameters
    ----------
    nrows : int
        Number of rows in the grid
    ncols : int
        Number of columns in the grid
    height_ratios : list, optional
        List of height ratios for rows
    width_ratios : list, optional
        List of width ratios for columns
    **kwargs : dict
        Additional arguments passed to plt.figure()
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    gs : matplotlib.gridspec.GridSpec
        The created GridSpec
    """
    fig = plt.figure(**kwargs)
    gs = plt.GridSpec(nrows, ncols, 
                     height_ratios=height_ratios,
                     width_ratios=width_ratios,
                     hspace=0.3,
                     wspace=0.3)
    return fig, gs

def style_legend(ax, ncols=1, **kwargs):
    """
    Style the legend for given axis.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to style the legend for
    ncols : int
        Number of columns in the legend
    **kwargs : dict
        Additional arguments passed to ax.legend()
    """
    legend_kwargs = {
        'ncols': ncols,
        'fontsize': 12,
        'frameon': True,
        'facecolor': 'white'
    }
    legend_kwargs.update(kwargs)
    ax.legend(**legend_kwargs)
