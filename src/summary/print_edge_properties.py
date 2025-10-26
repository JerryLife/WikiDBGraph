r"""
\begin{table}[htbp]
    \centering
    \small
    \setlength{\tabcolsep}{3pt}
    \caption{Summary of Edge (Database Relationship) Properties in WikiDBGraph}
    \label{tab:edge-props}
    \begin{tabular}{llccccl}
        \toprule
        \textbf{Category} & \textbf{Property} & \textbf{Min} & \textbf{Max} & \textbf{Mean} & \textbf{Median} & \textbf{Description} \\
        \midrule
        \multirow{4}{*}{Structural} & JaccTable & - & - & - & - & Jaccard index of table name sets \\
        & JaccCol & - & - & - & - & Jaccard index of column name sets \\
        & JaccType & - & - & - & - & Jaccard index of data type sets \\
        & GraphSim & - & - & - & - & Similarity of internal graph structures \\
        \midrule
        \multirow{2}{*}{Semantic} & EmbedSim & - & - & - & - & Cosine similarity of database embeddings \\
        & SimConf & - & - & - & - & Confidence score of similarity prediction \\
        \midrule
        \multirow{2}{*}{Statistical} & DistDiv & - & - & - & - & KL divergence of shared column distributions \\
        & OverlapRatio & - & - & - & - & Ratio of overlapping values in shared columns \\
        \bottomrule
    \end{tabular}
\end{table}
"""

import pandas as pd
import os
import numpy as np

def load_edge_properties():
    """Load edge structural properties from CSV file"""
    structural_props_path = "data/graph/edge_structural_properties_GED_0.94.csv"
    if os.path.exists(structural_props_path):
        return pd.read_csv(structural_props_path)
    else:
        print(f"Warning: {structural_props_path} not found")
        return pd.DataFrame()

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

def generate_edge_properties_table():
    """Generate LaTeX table for edge properties with actual calculated values"""
    # Load data
    edge_props = load_edge_properties()
    
    # Calculate structural properties
    jacc_table_min = edge_props['jaccard_table_names'].min() if not edge_props.empty else '-'
    jacc_table_max = edge_props['jaccard_table_names'].max() if not edge_props.empty else '-'
    jacc_table_mean = edge_props['jaccard_table_names'].mean() if not edge_props.empty else '-'
    jacc_table_median = edge_props['jaccard_table_names'].median() if not edge_props.empty else '-'
    
    jacc_col_min = edge_props['jaccard_columns'].min() if not edge_props.empty else '-'
    jacc_col_max = edge_props['jaccard_columns'].max() if not edge_props.empty else '-'
    jacc_col_mean = edge_props['jaccard_columns'].mean() if not edge_props.empty else '-'
    jacc_col_median = edge_props['jaccard_columns'].median() if not edge_props.empty else '-'
    
    jacc_type_min = edge_props['jaccard_data_types'].min() if not edge_props.empty else '-'
    jacc_type_max = edge_props['jaccard_data_types'].max() if not edge_props.empty else '-'
    jacc_type_mean = edge_props['jaccard_data_types'].mean() if not edge_props.empty else '-'
    jacc_type_median = edge_props['jaccard_data_types'].median() if not edge_props.empty else '-'
    
    hellinger_min = edge_props['hellinger_distance_data_types'].min() if not edge_props.empty else '-'
    hellinger_max = edge_props['hellinger_distance_data_types'].max() if not edge_props.empty else '-'
    hellinger_mean = edge_props['hellinger_distance_data_types'].mean() if not edge_props.empty else '-'
    hellinger_median = edge_props['hellinger_distance_data_types'].median() if not edge_props.empty else '-'
    
    ged_min = edge_props['graph_edit_distance'].min() if not edge_props.empty else '-'
    ged_max = edge_props['graph_edit_distance'].max() if not edge_props.empty else '-'
    ged_mean = edge_props['graph_edit_distance'].mean() if not edge_props.empty else '-'
    ged_median = edge_props['graph_edit_distance'].median() if not edge_props.empty else '-'
    
    common_tables_min = edge_props['common_tables'].min() if not edge_props.empty else '-'
    common_tables_max = edge_props['common_tables'].max() if not edge_props.empty else '-'
    common_tables_mean = edge_props['common_tables'].mean() if not edge_props.empty else '-'
    common_tables_median = edge_props['common_tables'].median() if not edge_props.empty else '-'
    
    common_columns_min = edge_props['common_columns'].min() if not edge_props.empty else '-'
    common_columns_max = edge_props['common_columns'].max() if not edge_props.empty else '-'
    common_columns_mean = edge_props['common_columns'].mean() if not edge_props.empty else '-'
    common_columns_median = edge_props['common_columns'].median() if not edge_props.empty else '-'
    
    common_types_min = edge_props['common_data_types'].min() if not edge_props.empty else '-'
    common_types_max = edge_props['common_data_types'].max() if not edge_props.empty else '-'
    common_types_mean = edge_props['common_data_types'].mean() if not edge_props.empty else '-'
    common_types_median = edge_props['common_data_types'].median() if not edge_props.empty else '-'
    
    latex_table = r"""
\begin{table}[htbp]
    \centering
    \small
    \setlength{\tabcolsep}{3pt}
    \caption{Summary of Edge (Database Relationship) Properties in WikiDBGraph}
    \label{tab:edge-props}
    \begin{tabular}{llccccl}
        \toprule
        \textbf{Category} & \textbf{Property} & \textbf{Min} & \textbf{Max} & \textbf{Mean} & \textbf{Median} & \textbf{Description} \\
        \midrule
        \multirow{8}{*}{Structural} & JaccTable & """ + f"{jacc_table_min:.2f}" + r" & " + f"{jacc_table_max:.2f}" + r" & " + f"{jacc_table_mean:.2f}" + r" & " + f"{jacc_table_median:.2f}" + r""" & Jaccard index of table name sets \\
        & JaccCol & """ + f"{jacc_col_min:.2f}" + r" & " + f"{jacc_col_max:.2f}" + r" & " + f"{jacc_col_mean:.2f}" + r" & " + f"{jacc_col_median:.2f}" + r""" & Jaccard index of column name sets \\
        & JaccType & """ + f"{jacc_type_min:.2f}" + r" & " + f"{jacc_type_max:.2f}" + r" & " + f"{jacc_type_mean:.2f}" + r" & " + f"{jacc_type_median:.2f}" + r""" & Jaccard index of data type sets \\
        & HellingerDist & """ + f"{hellinger_min:.2f}" + r" & " + f"{hellinger_max:.2f}" + r" & " + f"{hellinger_mean:.2f}" + r" & " + f"{hellinger_median:.2f}" + r""" & Hellinger distance of data type distributions \\
        & GED & """ + f"{ged_min:.2f}" + r" & " + f"{ged_max:.2f}" + r" & " + f"{ged_mean:.2f}" + r" & " + f"{ged_median:.2f}" + r""" & Graph edit distance between schema structures \\
        & CommonTables & """ + f"{common_tables_min}" + r" & " + f"{common_tables_max}" + r" & " + f"{common_tables_mean:.2f}" + r" & " + f"{common_tables_median:.2f}" + r""" & Number of common tables between schemas \\
        & CommonCols & """ + f"{common_columns_min}" + r" & " + f"{common_columns_max}" + r" & " + f"{common_columns_mean:.2f}" + r" & " + f"{common_columns_median:.2f}" + r""" & Number of common columns between schemas \\
        & CommonTypes & """ + f"{common_types_min}" + r" & " + f"{common_types_max}" + r" & " + f"{common_types_mean:.2f}" + r" & " + f"{common_types_median:.2f}" + r""" & Number of common data types between schemas \\
        \midrule
        \multirow{2}{*}{Semantic} & EmbedSim & - & - & - & - & Cosine similarity of database embeddings \\
        & SimConf & - & - & - & - & Confidence score of similarity prediction \\
        \midrule
        \multirow{2}{*}{Statistical} & DistDiv & - & - & - & - & KL divergence of shared column distributions \\
        & OverlapRatio & - & - & - & - & Ratio of overlapping values in shared columns \\
        \bottomrule
    \end{tabular}
\end{table}
"""
    return latex_table

if __name__ == "__main__":
    print(generate_edge_properties_table())
