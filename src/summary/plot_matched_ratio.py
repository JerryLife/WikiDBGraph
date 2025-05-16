import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_data(data_path):
    """Load and prepare the matched ratio data"""
    df = pd.read_csv(data_path, header=0)
    df['ratio'] = df['overlapped_records'] / df['total_records1']
    return df

def plot_ratio_histogram(df):
    """Plot histogram of matched record ratios"""
    plt.rcParams['font.size'] = 20
    plt.figure(figsize=(6, 5))
    plt.hist(df['ratio'], bins=50, edgecolor='black', density=False)
    plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.title(r'Matched Ratio (Mean: {:.3f})'.format(df['ratio'].mean()))
    plt.xlabel(r'Matched Ratio')
    plt.ylabel(r'Frequency')

    plt.savefig('fig/matched_ratio_hist.png', bbox_inches='tight', dpi=600)
    plt.close()

def get_zero_match_stats(df):
    """Calculate statistics for records with zero matches"""
    isoverlap = df['overlapped_records'] == 0
    stats = { 
        'mean': isoverlap.mean(),
        'std': isoverlap.std(), 
        'count': isoverlap.sum(),
        'total': len(df['total_records1']),
        'ratio': isoverlap.sum() / len(df['total_records1'])
    }
    return stats

def get_precise_match_stats(df):
    """Calculate statistics for records with precise matches"""
    isprecise = df['overlapped_records'] == df['total_records1']
    stats = { 
        'mean': isprecise.mean(),
        'std': isprecise.std(), 
        'count': isprecise.sum(),
        'total': len(df['total_records1']),
        'ratio': isprecise.sum() / len(df['total_records1'])
    }
    return stats

if __name__ == "__main__":
    # Load data and create plots
    df = load_data('out/matched_ratio_1000.csv')
    plot_ratio_histogram(df)
    
    # Get and print statistics
    zero_match_stats = get_zero_match_stats(df)
    precise_match_stats = get_precise_match_stats(df)
    
    print("\nMetric & Mean & Std & Count & Total & Ratio \\\\")
    print("\\hline")
    print(f"Zero matches & {zero_match_stats['mean']:.1f} & {zero_match_stats['std']:.1f} & {zero_match_stats['count']} & {zero_match_stats['total']} & {zero_match_stats['ratio']:.1f} \\\\")
    print(f"Precise matches & {precise_match_stats['mean']:.1f} & {precise_match_stats['std']:.1f} & {precise_match_stats['count']} & {precise_match_stats['total']} & {precise_match_stats['ratio']:.1f} \\\\")


"""
Metric & Mean & Std & Count & Total & Ratio \\
\hline
Zero matches & 1.0 & 0.2 & 202253 & 212642 & 1.0 \\
Precise matches & 0.0 & 0.0 & 349 & 212642 & 0.0 \\
"""