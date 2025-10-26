import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
# Calculate overlap stats across seeds
overlap_ratios = []
zero_overlap_ratios = []

for seed in range(10):
    # Load valid pairs for this seed
    df = pd.read_csv(f'out/valid_pairs_stats_seed{seed}.csv')
    
    # Calculate overlap ratios for this seed
    df['overlap_ratio'] = df['overlap_features'] / df[['table1_features', 'table2_features']].min(axis=1)
    
    # Calculate zero overlap ratio
    zero_overlap_ratio = (df['overlap_features'] == 0).mean()
    zero_overlap_ratios.append(zero_overlap_ratio)
    
    # Store all overlap ratios
    overlap_ratios.extend(df['overlap_ratio'].tolist())

# Print zero overlap statistics
print("Metric & Mean & Std & Min & Max \\\\")
print("\\hline")
print(f"Zero overlap ratio & {np.mean(zero_overlap_ratios):.3f} & {np.std(zero_overlap_ratios):.3f} & {np.min(zero_overlap_ratios):.3f} & {np.max(zero_overlap_ratios):.3f} \\\\")

plt.rcParams['font.size'] = 20

os.makedirs('fig', exist_ok=True)
plt.figure(figsize=(6, 5))
plt.hist(overlap_ratios, bins=50, edgecolor='black', density=False)
plt.xlabel(r'Overlap Ratio')
plt.ylabel(r'Frequency')
plt.title(rf'Overlap Ratios (Mean: {np.mean(overlap_ratios):.3f})')
plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.savefig('fig/overlap_distribution.png', bbox_inches='tight', dpi=600)
plt.close()

"""
Metric & Mean & Std & Min & Max \\
\hline
Zero overlap ratio & 0.254 & 0.013 & 0.237 & 0.277 \\
"""