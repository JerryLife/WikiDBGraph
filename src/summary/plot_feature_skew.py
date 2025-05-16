import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Calculate skew stats across seeds
skew_ratios = []

for seed in range(10):
    # Load valid pairs for this seed
    df = pd.read_csv(f'out/valid_pairs_stats_seed{seed}.csv')
    
    # Calculate skew ratios (min/max) for this seed
    df['skew_ratio'] = df[['table1_features', 'table2_features']].min(axis=1) / df[['table1_features', 'table2_features']].max(axis=1)
    
    # Store all skew ratios
    skew_ratios.extend(df['skew_ratio'].tolist())

plt.rcParams['font.size'] = 20

os.makedirs('fig', exist_ok=True)
plt.figure(figsize=(6, 5))
plt.hist(skew_ratios, bins=50, edgecolor='black', density=False)
plt.xlabel(r'Feature Balance Ratio')
plt.ylabel(r'Frequency')
plt.title(rf'Balance Ratio (Mean: {np.mean(skew_ratios):.3f})')
plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.savefig('fig/feature_balance_ratio.png', bbox_inches='tight', dpi=600)
plt.close()
