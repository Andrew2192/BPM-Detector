from matplotlib import rcParams

def apply_plot_style():
    rcParams.update({
        'figure.dpi': 160,
        'savefig.dpi': 160,
        'axes.titlesize': 12,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'font.size': 9,
    })
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = [
        'SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC', 'PingFang SC',
        'Heiti SC', 'Arial Unicode MS', 'DejaVu Sans'
    ]
    rcParams['axes.unicode_minus'] = False