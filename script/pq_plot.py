import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

# Import parquet file
df = pd.read_parquet('data/line_world_mc.parquet')

# Prepare Data to Plot
l = df['len']

# Plot params
pparam = dict(
    xlabel = r'Episode',
    ylabel = r'Length',
    xscale = 'linear',
    yscale = 'log',
)

# Plot
with plt.style.context(["science", "nature"]):
    fig, ax = plt.subplots()
    ax.autoscale(tight=True)
    ax.set(**pparam)
    ax.plot(l)
    ax.legend()
    fig.savefig('figure/line_world_plot.png', dpi=600, bbox_inches='tight')
