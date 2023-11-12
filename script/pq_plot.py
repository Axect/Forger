import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
import argparse

# Command line arguments
parser = argparse.ArgumentParser(description='Plot data from parquet file')
parser.add_argument('-t', '--task', type=str, help='Task name', required=True)
args = parser.parse_args()

# Import parquet file
df = pd.read_parquet(f'data/{args.task}.parquet')

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
    fig.savefig(f'figure/{args.task}_plot.png', dpi=600, bbox_inches='tight')
