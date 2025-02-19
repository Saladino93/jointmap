import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import pathlib



class CMBLensingPlot:
    def __init__(self, rows=1, cols=1, figsize=(14, 10), sharex=False, sharey=False, outdir = ""):
        self.fig, self.axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False, sharex=sharex, sharey=sharey)
        self.rows = rows
        self.cols = cols
        self.sharex = sharex
        self.sharey = sharey
        self.color_sets = {
            'default': ['#1a4059', '#f4a261', '#e76f51', '#2a9d8f', '#8a3c58', '#2c7da0', '#d1495b'],
            'pastel': ['#ff9aa2', '#81b29a', '#f6bd60', '#b5838d', '#7fa998', '#c5a3be', '#e0aaac'],
            'vibrant': ['#ff595e', '#0496ff', '#ffca3a', '#8ac926', '#9b5de5', '#00bbf9', '#f15bb5'],
            'cool': ['#05668d', '#427aa1', '#679436', '#8ac926', '#a3f7bf', '#38a795', '#24c5e0'],
            'warm': ['#ff7b00', '#ff8800', '#ff9500', '#ffa200', '#ffaa00', '#ffb700', '#ffc300']
        }
        self.set_color_cycle('default')
        self.setup_plot()
        self.outdir = pathlib.Path(outdir)

    def setup_plot(self):
        plt.rcParams['font.family'] = 'Latin Modern Mono Light'
        for ax in self.axes.flat:
            ax.tick_params(axis='both', which='major', labelsize=15)
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
            ax.set_facecolor('#f8f9fa')
            ax.grid(True, linestyle='--', alpha=0.6)

    def set_color_cycle(self, color_set_name):
        if color_set_name not in self.color_sets:
            raise ValueError(f"Color set '{color_set_name}' not found. Available sets: {', '.join(self.color_sets.keys())}")
        for ax in self.axes.flat:
            ax.set_prop_cycle(cycler(color=self.color_sets[color_set_name]))

    def set_labels(self, xlabel, ylabel, row=0, col=0, fontsize=22):
        ax = self.axes[row, col]
        if xlabel and (row == self.rows - 1 or not self.sharex):
            ax.set_xlabel(xlabel, fontsize=fontsize, labelpad=10)
        if ylabel and (col == 0 or not self.sharey):
            ax.set_ylabel(ylabel, fontsize=fontsize, labelpad=10)

    def add_curve(self, x, y, label = None, row=0, col=0, color=None, linestyle='-', linewidth=3, alpha = 1):
        ax = self.axes[row, col]
        return ax.plot(x, y, linewidth=linewidth, alpha=alpha, label=label, linestyle = linestyle, color=color)

    def set_legend(self, row=0, col=0, fontsize=16, ncol=2, loc = 'best', bbox_to_anchor = None):
        ax = self.axes[row, col]
        legend = ax.legend(fontsize=fontsize, loc=loc, frameon=True,
                           edgecolor='black', facecolor='white', framealpha=0.8,
                           ncol=ncol, bbox_to_anchor=bbox_to_anchor)
        #legend.get_frame().set_boxstyle('round,pad=0.5')
        legend.get_frame().set_linewidth(1.)

    def set_title(self, title, row=0, col=0, fontsize=24):
        self.axes[row, col].set_title(title, fontsize=fontsize, pad=20)

    def set_xlim(self, xmin, xmax, row=0, col=0):
        self.axes[row, col].set_xlim(xmin, xmax)

    def set_ylim(self, ymin, ymax, row=0, col=0):
        self.axes[row, col].set_ylim(ymin, ymax)

    def plot_between(self, x, y1, y2, label, row=0, col=0):
        ax = self.axes[row, col]
        ax.fill_between(x, y1, y2, alpha=0.3, label=label)

    def set_scale(self, xscale="linear", yscale="linear", row = 0, col = 0):
        ax = self.axes[row, col]
        if xscale:
            ax.set_xscale(xscale)
        if yscale:
            ax.set_yscale(yscale)


    def save_plot(self, filename, dpi=300, outdir = ""):
        plt.tight_layout()
        outdir = self.outdir if outdir == "" else pathlib.Path(outdir)
        self.fig.savefig(outdir/filename, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved as {filename} with DPI {dpi}")

    def show_plot(self):
        plt.tight_layout()
        plt.show()

"""
if __name__ == "__main__":
    plot = CMBLensingPlot(rows=2, cols=2, figsize=(20, 16), sharex=True, sharey=True)
    
    x = np.linspace(0, 10, 100)
    
    color_sets = ['default', 'pastel', 'vibrant', 'cool']
    
    for i in range(4):
        row, col = divmod(i, 2)
        plot.set_color_cycle(color_sets[i])
        plot.set_labels("Angular Scale (arcmin)", "Relative Lensing Bias", row=row, col=col)
        plot.set_title(f"Color Set: {color_sets[i]}", row=row, col=col)
        
        for j in range(7):
            y = np.sin(x + j/2) * np.exp(-0.1 * j * x)
            plot.add_curve(x, y, f"Component {j+1}", row=row, col=col)
        
        plot.set_legend(row=row, col=col)

    plot.save_plot("cmb_lensing_bias_multiplot_enhanced_color_sets.png", dpi=300)
    plot.show_plot()"""