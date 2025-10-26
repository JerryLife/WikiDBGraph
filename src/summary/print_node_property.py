r"""
\begin{table}[htbp]
    \centering
    \small
    \setlength{\tabcolsep}{3pt}
    \caption{Summary of Node (Database) Properties in WikiDBGraph}
    \label{tab:node-props}
    \begin{tabular}{llccccl}
        \toprule
        \textbf{Category} & \textbf{Property} & \textbf{Min} & \textbf{Max} & \textbf{Mean} & \textbf{Median} & \textbf{Description} \\
        \midrule
        \multirow{5}{*}{Structural} & \#Tables & 1 & 87 & 4.32 & 3 & Number of tables in the database \\
        & \#Columns & 3 & 1,243 & 28.76 & 18 & Total number of columns across all tables \\
        & PropCat & 0.0 & 1.0 & 0.64 & 0.67 & Proportion of categorical columns \\
        & FKDensity & 0.0 & 0.42 & 0.08 & 0.06 & Number of foreign keys / number of columns \\
        & AvgConn & 0.0 & 12.5 & 1.87 & 1.33 & Average potential joins per table \\
        \midrule
        \multirow{2}{*}{Semantic} & DBEmbed & - & - & - & - & Database embedding vector (768-dim) \\
        & Topic & - & - & - & - & Topic category derived from clustering \\
        \midrule
        \multirow{5}{*}{Statistical} & DataVol & 2KB & 1.2GB & 3.4MB & 1.1MB & Total size of database files \\
        & AllJoinSize & 0 & $10^8$ & $2.3×10^5$ & $4.7×10^4$ & Row count when joining all tables \\
        & AvgCard & 1.0 & $10^6$ & 342.5 & 87.3 & Average distinct values per column \\
        & AvgSparsity & 0.0 & 0.98 & 0.12 & 0.08 & Average proportion of NULL values \\
        & AvgEntropy & 0.0 & 8.76 & 2.43 & 1.98 & Average information entropy of columns \\
        \bottomrule
    \end{tabular}
\end{table}
"""


import pandas as pd
import os
import numpy as np
import json
from pathlib import Path

def load_node_properties():
    """Load node structural properties from CSV file"""
    structural_props_path = "data/graph/node_structural_properties.csv"
    if os.path.exists(structural_props_path):
        return pd.read_csv(structural_props_path, engine='python')
    else:
        print(f"Warning: {structural_props_path} not found")
        return pd.DataFrame()

def load_cluster_assignments():
    """Load cluster assignments from CSV file"""
    cluster_path = "data/graph/cluster_assignments_dim2_sz100_msNone.csv"
    if os.path.exists(cluster_path):
        return pd.read_csv(cluster_path)
    else:
        print(f"Warning: {cluster_path} not found")
        return pd.DataFrame()

def load_community_assignments():
    """Load community assignments from CSV file"""
    community_path = "data/graph/community_assignment_0.94.csv"
    if os.path.exists(community_path):
        return pd.read_csv(community_path)
    else:
        print(f"Warning: {community_path} not found")
        return pd.DataFrame()

def load_data_volume():
    """Load data volume statistics from CSV file"""
    volume_path = "data/graph/data_volume.csv"
    if os.path.exists(volume_path):
        return pd.read_csv(volume_path)
    else:
        print(f"Warning: {volume_path} not found")
        return pd.DataFrame()

def get_cache_path(data_file):
    """Get the cache file path for a given data file"""
    data_path = Path(data_file)
    cache_file = data_path.parent / f"{data_path.stem}_summary_cache.json"
    return cache_file

def generate_column_summary_cache(data_file, value_column, group_column='db_id'):
    """
    Generate summary statistics for large column files and cache them.
    
    Args:
        data_file: Path to the CSV file
        value_column: Column name to aggregate (e.g., 'n_distinct', 'sparsity', 'entropy')
        group_column: Column to group by (default: 'db_id')
    
    Returns:
        Dictionary with summary statistics
    """
    cache_path = get_cache_path(data_file)
    
    # Check if cache exists and is newer than data file
    if cache_path.exists():
        data_mtime = os.path.getmtime(data_file)
        cache_mtime = os.path.getmtime(cache_path)
        if cache_mtime > data_mtime:
            print(f"Loading from cache: {cache_path}")
            with open(cache_path, 'r') as f:
                return json.load(f)
    
    print(f"Generating cache for {data_file} (this may take a while for large files)...")
    
    # Read file in chunks to handle large files
    chunk_size = 1000000
    aggregated_stats = []
    
    for chunk in pd.read_csv(data_file, chunksize=chunk_size, dtype={value_column: float}, low_memory=False):
        # Group by db_id and calculate mean for this chunk
        if value_column in chunk.columns and group_column in chunk.columns:
            chunk_stats = chunk.groupby(group_column)[value_column].agg(['mean', 'min', 'max', 'count']).reset_index()
            aggregated_stats.append(chunk_stats)
    
    if not aggregated_stats:
        return {'min': None, 'max': None, 'mean': None, 'median': None}
    
    # Combine all chunks
    all_stats = pd.concat(aggregated_stats, ignore_index=True)
    
    # Calculate final statistics per database
    final_stats = all_stats.groupby(group_column).agg({
        'mean': lambda x: np.average(x, weights=all_stats.loc[x.index, 'count']),  # Weighted average
        'min': 'min',
        'max': 'max',
        'count': 'sum'
    }).reset_index()
    
    # Calculate overall statistics
    avg_per_db = final_stats['mean']
    
    summary = {
        'min': float(avg_per_db.min()),
        'max': float(avg_per_db.max()),
        'mean': float(avg_per_db.mean()),
        'median': float(avg_per_db.median()),
        'db_count': len(final_stats)
    }
    
    # Save cache
    print(f"Saving cache to {cache_path}")
    with open(cache_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary

def load_column_cardinality_summary():
    """Load or generate column cardinality summary statistics"""
    cardinality_path = "data/graph/column_cardinality.csv"
    if not os.path.exists(cardinality_path):
        print(f"Warning: {cardinality_path} not found")
        return {'min': None, 'max': None, 'mean': None, 'median': None}
    
    return generate_column_summary_cache(cardinality_path, 'n_distinct')

def load_column_sparsity_summary():
    """Load or generate column sparsity summary statistics"""
    sparsity_path = "data/graph/column_sparsity.csv"
    if not os.path.exists(sparsity_path):
        print(f"Warning: {sparsity_path} not found")
        return {'min': None, 'max': None, 'mean': None, 'median': None}
    
    return generate_column_summary_cache(sparsity_path, 'sparsity')

def load_column_entropy_summary():
    """Load or generate column entropy summary statistics"""
    entropy_path = "data/graph/column_entropy.csv"
    if not os.path.exists(entropy_path):
        print(f"Warning: {entropy_path} not found")
        return {'min': None, 'max': None, 'mean': None, 'median': None}
    
    return generate_column_summary_cache(entropy_path, 'entropy')

def load_all_join_size():
    """Load all join size results from CSV file"""
    join_size_path = "data/graph/all_join_size_results_est.csv"
    if os.path.exists(join_size_path):
        return pd.read_csv(join_size_path)
    else:
        print(f"Warning: {join_size_path} not found")
        return pd.DataFrame()

def format_bytes(bytes_value):
    """Format bytes to human-readable format with appropriate unit"""
    if bytes_value >= 1e9:  # GB
        return f"{bytes_value/1e9:.1f}GB"
    elif bytes_value >= 1e6:  # MB
        return f"{bytes_value/1e6:.1f}MB"
    elif bytes_value >= 1e3:  # KB
        return f"{bytes_value/1e3:.1f}KB"
    else:
        return f"{bytes_value:.0f}B"

def format_number(value):
    """Format numbers for LaTeX table with appropriate notation"""
    if value >= 1e6:
        return f"${value/1e6:.1f}\\times10^6$"
    elif value >= 1e5:
        return f"${value/1e5:.1f}\\times10^5$"
    elif value >= 1e4:
        return f"${value/1e4:.1f}\\times10^4$"
    elif value >= 1000:
        return f"{value:,.0f}"
    elif value >= 100:
        return f"{value:.1f}"
    else:
        return f"{value:.2f}"

def generate_node_properties_table():
    """Generate LaTeX table for node properties with actual calculated values"""
    # Load data
    node_props = load_node_properties()
    cluster_assignments = load_cluster_assignments()
    community_assignments = load_community_assignments()
    
    # Calculate structural properties
    num_tables_min = node_props['num_tables'].min() if not node_props.empty else ''
    num_tables_max = node_props['num_tables'].max() if not node_props.empty else ''
    num_tables_mean = node_props['num_tables'].mean() if not node_props.empty else ''
    num_tables_median = node_props['num_tables'].median() if not node_props.empty else 3
    
    num_columns_min = node_props['num_columns'].min() if not node_props.empty else ''
    num_columns_max = node_props['num_columns'].max() if not node_props.empty else ''
    num_columns_mean = node_props['num_columns'].mean() if not node_props.empty else ''
    num_columns_median = node_props['num_columns'].median() if not node_props.empty else ''
    
    # Calculate proportion of categorical columns (using data_type_proportions)
    # This is a simplification - we're assuming 'string' and 'wikibase-entityid' are categorical
    prop_cat_values = []
    if not node_props.empty and 'data_type_proportions' in node_props.columns:
        for prop_str in node_props['data_type_proportions']:
            try:
                prop_dict = eval(prop_str)
                cat_types = ['string', 'wikibase-entityid']
                cat_prop = sum(prop_dict.get(t, 0) for t in cat_types)
                prop_cat_values.append(cat_prop)
            except:
                continue
    
    prop_cat_min = min(prop_cat_values) if prop_cat_values else ''
    prop_cat_max = max(prop_cat_values) if prop_cat_values else ''
    prop_cat_mean = sum(prop_cat_values)/len(prop_cat_values) if prop_cat_values else ''
    prop_cat_median = np.median(prop_cat_values) if prop_cat_values else ''
    
    # Foreign key density
    fk_min = node_props['foreign_key_density'].min() if not node_props.empty else ''
    fk_max = node_props['foreign_key_density'].max() if not node_props.empty else ''
    fk_mean = node_props['foreign_key_density'].mean() if not node_props.empty else ''
    fk_median = node_props['foreign_key_density'].median() if not node_props.empty else ''
    
    # Average table connectivity
    conn_min = node_props['avg_table_connectivity'].min() if not node_props.empty else ''
    conn_max = node_props['avg_table_connectivity'].max() if not node_props.empty else ''
    conn_mean = node_props['avg_table_connectivity'].mean() if not node_props.empty else ''
    conn_median = node_props['avg_table_connectivity'].median() if not node_props.empty else ''
    
    # Count unique clusters and communities
    num_clusters = len(cluster_assignments['cluster'].unique()) if not cluster_assignments.empty else ''
    num_communities = len(community_assignments['partition'].unique()) if not community_assignments.empty else ''
    
    # Calculate cluster size statistics
    cluster_sizes = []
    if not cluster_assignments.empty:
        cluster_counts = cluster_assignments['cluster'].value_counts()
        cluster_sizes = cluster_counts.values
        cluster_size_min = cluster_counts.min()
        cluster_size_max = cluster_counts.max()
        cluster_size_mean = cluster_counts.mean()
        cluster_size_median = cluster_counts.median()
    else:
        cluster_size_min = cluster_size_max = cluster_size_mean = cluster_size_median = ''
    
    # Calculate community size statistics
    community_sizes = []
    if not community_assignments.empty:
        community_counts = community_assignments['partition'].value_counts()
        community_sizes = community_counts.values
        community_size_min = community_counts.min()
        community_size_max = community_counts.max()
        community_size_mean = community_counts.mean()
        community_size_median = community_counts.median()
    else:
        community_size_min = community_size_max = community_size_mean = community_size_median = ''
    
    # Load statistical property data (using caching for large files)
    data_volume = load_data_volume()
    all_join_size = load_all_join_size()
    
    # Load summaries from cache for large files
    cardinality_summary = load_column_cardinality_summary()
    sparsity_summary = load_column_sparsity_summary()
    entropy_summary = load_column_entropy_summary()
    
    # Calculate data volume statistics
    if not data_volume.empty:
        vol_min = format_bytes(data_volume['volume_bytes'].min())
        vol_max = format_bytes(data_volume['volume_bytes'].max())
        vol_mean = format_bytes(data_volume['volume_bytes'].mean())
        vol_median = format_bytes(data_volume['volume_bytes'].median())
    else:
        vol_min = vol_max = vol_mean = vol_median = '-'
    
    # Calculate all join size statistics
    if not all_join_size.empty:
        join_min = format_number(all_join_size['all_join_size'].min())
        join_max = format_number(all_join_size['all_join_size'].max())
        join_mean = format_number(all_join_size['all_join_size'].mean())
        join_median = format_number(all_join_size['all_join_size'].median())
    else:
        join_min = join_max = join_mean = join_median = '-'
    
    # Use cached cardinality summary
    if cardinality_summary['min'] is not None:
        card_min = format_number(cardinality_summary['min'])
        card_max = format_number(cardinality_summary['max'])
        card_mean = format_number(cardinality_summary['mean'])
        card_median = format_number(cardinality_summary['median'])
    else:
        card_min = card_max = card_mean = card_median = '-'
    
    # Use cached sparsity summary
    if sparsity_summary['min'] is not None:
        sparsity_min = f"{sparsity_summary['min']:.2f}"
        sparsity_max = f"{sparsity_summary['max']:.2f}"
        sparsity_mean = f"{sparsity_summary['mean']:.2f}"
        sparsity_median = f"{sparsity_summary['median']:.2f}"
    else:
        sparsity_min = sparsity_max = sparsity_mean = sparsity_median = '-'
    
    # Use cached entropy summary
    if entropy_summary['min'] is not None:
        entropy_min = f"{entropy_summary['min']:.2f}"
        entropy_max = f"{entropy_summary['max']:.2f}"
        entropy_mean = f"{entropy_summary['mean']:.2f}"
        entropy_median = f"{entropy_summary['median']:.2f}"
    else:
        entropy_min = entropy_max = entropy_mean = entropy_median = '-'
    
    latex_table = r"""
\begin{table}[htbp]
    \centering
    \small
    \setlength{\tabcolsep}{3pt}
    \caption{Summary of Node (Database) Properties in WikiDBGraph}
    \label{tab:node-props}
    \begin{tabular}{llccccl}
        \toprule
        \textbf{Category} & \textbf{Property} & \textbf{Min} & \textbf{Max} & \textbf{Mean} & \textbf{Median} & \textbf{Description} \\
        \midrule
        \multirow{5}{*}{Structural} & \#Tables & """ + f"{num_tables_min}" + r" & " + f"{num_tables_max}" + r" & " + f"{num_tables_mean:.2f}" + r" & " + f"{num_tables_median}" + r""" & Number of tables in the database \\
        & \#Columns & """ + f"{num_columns_min}" + r" & " + f"{num_columns_max:,}" + r" & " + f"{num_columns_mean:.2f}" + r" & " + f"{num_columns_median}" + r""" & Total number of columns across all tables \\
        & PropCat & """ + f"{prop_cat_min:.1f}" + r" & " + f"{prop_cat_max:.1f}" + r" & " + f"{prop_cat_mean:.2f}" + r" & " + f"{prop_cat_median:.2f}" + r""" & Proportion of categorical columns \\
        & FKDensity & """ + f"{fk_min:.1f}" + r" & " + f"{fk_max:.2f}" + r" & " + f"{fk_mean:.2f}" + r" & " + f"{fk_median:.2f}" + r""" & Number of foreign keys / number of columns \\
        & AvgConn & """ + f"{conn_min:.1f}" + r" & " + f"{conn_max:.1f}" + r" & " + f"{conn_mean:.2f}" + r" & " + f"{conn_median:.2f}" + r""" & Average potential joins per table \\
        \midrule
        \multirow{3}{*}{Semantic} & DBEmbed & - & - & - & - & Database embedding vector (768-dim) \\
        & Cluster & """ + f"{cluster_size_min}" + r" & " + f"{cluster_size_max}" + r" & " + f"{cluster_size_mean:.2f}" + r" & " + f"{cluster_size_median:.0f}" + r""" & Topic category from clustering (""" + str(num_clusters) + r""" clusters) \\
        & Community & """ + f"{community_size_min}" + r" & " + f"{community_size_max}" + r" & " + f"{community_size_mean:.2f}" + r" & " + f"{community_size_median:.0f}" + r""" & Community from graph structure (""" + str(num_communities) + r""" communities) \\
        \midrule
        \multirow{5}{*}{Statistical} & DataVolume & """ + vol_min + r" & " + vol_max + r" & " + vol_mean + r" & " + vol_median + r""" & Total size of database files \\
        & AllJoinSize & """ + join_min + r" & " + join_max + r" & " + join_mean + r" & " + join_median + r""" & Row count when joining all tables \\
        & AverageCardinality & """ + card_min + r" & " + card_max + r" & " + card_mean + r" & " + card_median + r""" & Average distinct values per column \\
        & AverageSparsity & """ + sparsity_min + r" & " + sparsity_max + r" & " + sparsity_mean + r" & " + sparsity_median + r""" & Average proportion of NULL values \\
        & AverageEntropy & """ + entropy_min + r" & " + entropy_max + r" & " + entropy_mean + r" & " + entropy_median + r""" & Average entropy of columns \\
        \bottomrule
    \end{tabular}
\end{table}
"""
    return latex_table

if __name__ == "__main__":
    print(generate_node_properties_table())

