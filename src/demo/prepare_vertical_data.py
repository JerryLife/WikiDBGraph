import pandas as pd
import sqlite3
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Helper function to extract base ID (e.g., TcCLB.xxxxxx from TcCLB.xxxxxx.yyy)
def get_base_gene_id(gene_id_val):
    if pd.isna(gene_id_val):
        return np.nan 
    parts = str(gene_id_val).split('.')
    if len(parts) >= 2:
        return '.'.join(parts[:2])
    return str(gene_id_val)

def load_database_00381(output_dir_for_raw):
    db_dir = "data/unziplink/00381-TrypanosomaCruziOrthologs1"
    db_file_path = os.path.join(db_dir, "database.db")
    if not os.path.exists(db_file_path):
        print(f"[Error] Database file not found for DB 00381: {db_file_path}")
        return pd.DataFrame()
    with sqlite3.connect(db_file_path) as conn:
        query = "SELECT * FROM GeneOrthologsAnnotations" # This table contains GeneId
        df = pd.read_sql_query(query, conn)
    print(f"Shape of DB 00381 (raw from GeneOrthologsAnnotations): {df.shape}")
    os.makedirs(output_dir_for_raw, exist_ok=True)
    df.to_csv(os.path.join(output_dir_for_raw, "00381_raw_GeneOrthologsAnnotations.csv"), index=False)
    return df

def load_database_48804(output_dir_for_raw):
    """
    Load database 48804 (Ortholog_Lpg1l_Genomic_Data)
    MODIFIED: Focusing on Ortholog_Lpg1l_Protein_Annotations table (which has 147 rows).
    This table contains 'Gene_Db_Id' which will be used for linking.
    """
    db_dir = "data/unziplink/48804-Ortholog_Lpg1l_Genomic_Data"
    db_file_path = os.path.join(db_dir, "database.db")
    if not os.path.exists(db_file_path):
        print(f"[Error] Database file not found for DB 48804: {db_file_path}")
        return pd.DataFrame()
    with sqlite3.connect(db_file_path) as conn:
        # MODIFIED: Querying the protein table
        query = "SELECT * FROM Ortholog_Lpg1l_Protein_Annotations"
        df = pd.read_sql_query(query, conn)
    print(f"Shape of DB 48804 (raw from Ortholog_Lpg1l_Protein_Annotations): {df.shape}") # Should be 147 rows initially
    os.makedirs(output_dir_for_raw, exist_ok=True)
    # MODIFIED: Reflecting the source table in the raw CSV name
    df.to_csv(os.path.join(output_dir_for_raw, "48804_raw_Ortholog_Lpg1l_Protein_Annotations.csv"), index=False)
    return df

def process_dataframe_for_vfl(df_orig, original_link_col, prefix, common_id_col_name="Common_Gene_ID"):
    """
    Helper function to process a dataframe before merging for VFL.
    Keeps all rows after Common_Gene_ID generation (no deduplication on Common_Gene_ID).
    Handles NaNs in object/string columns by converting to string "nan" before LabelEncoding.
    """
    if df_orig.empty:
        print(f"[Warning] Input dataframe for prefix '{prefix}' is empty during processing.")
        return pd.DataFrame(), []

    df = df_orig.copy()

    if original_link_col not in df.columns:
        print(f"[Error] Linking column '{original_link_col}' not found in DataFrame for prefix '{prefix}'. Available: {df.columns.tolist()}")
        return pd.DataFrame(), []
    df[common_id_col_name] = df[original_link_col].apply(get_base_gene_id)
    
    # Only drop rows if the Common_Gene_ID itself is NaN (i.e., original link_col was NaN)
    initial_rows = len(df)
    df.dropna(subset=[common_id_col_name], inplace=True)
    dropped_rows = initial_rows - len(df)
    if dropped_rows > 0:
        print(f"[Info] Dropped {dropped_rows} rows from {prefix.strip('_')} data due to NaN Common_Gene_ID.")

    df = df.reset_index(drop=True) 
    
    print(f"Shape of DataFrame (source: {prefix.strip('_')}) after {common_id_col_name} creation (duplicates on {common_id_col_name} KEPT): {df.shape}")

    df.columns = [prefix + col if col != common_id_col_name else col for col in df.columns]
    final_cols = list(df.columns)

    for col in df.columns:
        if col == common_id_col_name: # Common_Gene_ID might be object if it contains non-numeric parts
            if df[col].dtype in ['object', 'string', 'category']:
                 # If Common_Gene_ID needs encoding (e.g. if it can have string values like 'LPG1L')
                df[col] = df[col].astype(str) # Ensure it's string for LE
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
            continue # Skip further processing for the join key itself if already handled
        
        if pd.api.types.is_numeric_dtype(df[col]):
            pass # Numerical NaNs will be preserved, XGBoost handles them
        elif df[col].dtype in ['object', 'string', 'category']:
            df[col] = df[col].astype(str) # Converts NaNs to "nan", empty strings to ""
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col]) # "nan" and "" become distinct categories
            
    return df, final_cols

def create_merged_dataset(
    df_primary_orig,
    df_secondary_orig,
    primary_orig_link_col,
    secondary_orig_link_col,
    primary_col_prefix,
    secondary_col_prefix,
    output_dir_primary_party,
    output_dir_secondary_party
):
    if df_primary_orig.empty:
        print("[Error] Primary input dataframe (DB 48804) is empty. Skipping merge.")
        return None, None
    
    df_primary, df_primary_final_cols = process_dataframe_for_vfl(
        df_primary_orig, primary_orig_link_col, primary_col_prefix
    )
    
    if not df_secondary_orig.empty:
        df_secondary, df_secondary_final_cols = process_dataframe_for_vfl(
            df_secondary_orig, secondary_orig_link_col, secondary_col_prefix
        )
    else:
        print("[Info] Secondary dataframe (DB 00381) is empty. Proceeding with primary data only for merge step.")
        df_secondary = pd.DataFrame(columns=['Common_Gene_ID']) 
        df_secondary_final_cols = ['Common_Gene_ID']

    if df_primary.empty :
        print("[Error] Primary dataframe (DB 48804) became empty after processing (e.g. all link keys were NaN). Skipping merge.")
        return None, None
        
    print(f"Columns in processed primary DataFrame ({primary_col_prefix.strip('_')}) before merge: {df_primary_final_cols}")
    if 'Common_Gene_ID' in df_secondary.columns:
        print(f"Columns in processed secondary DataFrame ({secondary_col_prefix.strip('_')}) before merge: {df_secondary_final_cols}")

    if 'Common_Gene_ID' in df_secondary.columns and len(df_secondary_final_cols) > 1 :
        merged_df = pd.merge(df_primary, df_secondary, on='Common_Gene_ID', how='left')
    else:
        print("[Info] Secondary dataframe has no features to merge beyond Common_Gene_ID. Merged_df will be based on primary.")
        merged_df = df_primary.copy()
        # Add columns from secondary as NaNs if they were expected but secondary df was empty of features
        if not df_secondary_orig.empty: # Check original secondary df
            expected_secondary_cols_after_prefix = [secondary_col_prefix + c if c != 'Common_Gene_ID' else c for c in df_secondary_orig.columns]
            expected_secondary_cols_after_prefix.append('Common_Gene_ID') # Common_Gene_ID would have been added
            expected_secondary_cols_after_prefix = list(dict.fromkeys(expected_secondary_cols_after_prefix)) # unique

            for col_name_in_secondary_schema in expected_secondary_cols_after_prefix:
                # Use the final prefixed name from df_secondary_final_cols if it exists
                # This part is tricky if df_secondary became just Common_Gene_ID
                # For simplicity, ensure placeholder columns if secondary had features
                # df_secondary_final_cols from an empty features df_secondary only has Common_Gene_ID
                if len(df_secondary_final_cols) <=1: # only Common_Gene_ID
                     temp_sec_cols_to_add = [c for c in expected_secondary_cols_after_prefix if c != 'Common_Gene_ID' and c not in merged_df.columns]
                     for col_to_add in temp_sec_cols_to_add:
                         merged_df[col_to_add] = np.nan
                else: # df_secondary was processed and had features
                     for col in df_secondary_final_cols:
                        if col != 'Common_Gene_ID' and col not in merged_df.columns:
                           merged_df[col] = np.nan


    print(f"Shape of the fully merged dataset before train/test split: {merged_df.shape}")
    if merged_df.empty:
        print("[Error] Merged dataframe is empty.")
        return None, None

    merged_df_train, merged_df_test = train_test_split(merged_df, test_size=0.2, shuffle=True, random_state=0)

    df_primary_party_aligned_train = merged_df_train[df_primary_final_cols]
    df_primary_party_aligned_test = merged_df_test[df_primary_final_cols]

    valid_secondary_cols_in_merge = [col for col in df_secondary_final_cols if col in merged_df_train.columns]
    df_secondary_party_aligned_train = merged_df_train[valid_secondary_cols_in_merge]
    df_secondary_party_aligned_test = merged_df_test[valid_secondary_cols_in_merge]

    os.makedirs(output_dir_primary_party, exist_ok=True)
    df_primary_party_aligned_train.to_csv(os.path.join(output_dir_primary_party, f"{primary_col_prefix}aligned_train.csv"), index=False)
    df_primary_party_aligned_test.to_csv(os.path.join(output_dir_primary_party, f"{primary_col_prefix}aligned_test.csv"), index=False)

    os.makedirs(output_dir_secondary_party, exist_ok=True)
    df_secondary_party_aligned_train.to_csv(os.path.join(output_dir_secondary_party, f"{secondary_col_prefix}aligned_train.csv"), index=False)
    df_secondary_party_aligned_test.to_csv(os.path.join(output_dir_secondary_party, f"{secondary_col_prefix}aligned_test.csv"), index=False)

    print(f"Shape of the merged train dataset: {merged_df_train.shape}")
    print(f"Shape of the merged test dataset: {merged_df_test.shape}")
    print(f"Shape of {primary_col_prefix}aligned_train: {df_primary_party_aligned_train.shape}")
    if not df_secondary_party_aligned_train.empty and len(df_secondary_party_aligned_train.columns) > 1:
        print(f"Shape of {secondary_col_prefix}aligned_train: {df_secondary_party_aligned_train.shape}")
    else:
        print(f"Secondary party aligned train data ({secondary_col_prefix}) is empty or effectively contains only Common_Gene_ID.")

    return merged_df_train, merged_df_test

if __name__ == "__main__":
    # Using a new version in directory name to signify changes
    base_output_dir = "data/clean/" 
    
    dir_00381_raw = os.path.join(base_output_dir, "00381", "raw")
    dir_48804_raw = os.path.join(base_output_dir, "48804", "raw")
    
    dir_00381_aligned = os.path.join(base_output_dir, "00381", "aligned") 
    dir_48804_aligned = os.path.join(base_output_dir, "48804", "aligned") 

    df_48804_loaded = load_database_48804(dir_48804_raw) 
    df_00381_loaded = load_database_00381(dir_00381_raw) 

    if not df_48804_loaded.empty:
        print("\nStarting merged dataset creation with DB 48804 (Protein Annotations) as primary...")
        print("Duplicates on Common_Gene_ID within each source DB (after base ID mapping) will be KEPT.")
        print("NaNs in object/string columns will be converted to string 'nan' and label encoded.")
        
        create_merged_dataset(
            df_primary_orig=df_48804_loaded,      
            df_secondary_orig=df_00381_loaded,    
            primary_orig_link_col='Gene_Db_Id', # This column exists in Ortholog_Lpg1l_Protein_Annotations    
            secondary_orig_link_col='GeneId',       
            primary_col_prefix='lpg_', # From LPG1L proteins (DB 48804)          
            secondary_col_prefix='tc_', # From T. cruzi general orthologs (DB 00381)          
            output_dir_primary_party=dir_48804_aligned, 
            output_dir_secondary_party=dir_00381_aligned
        )
        
        print("\nChecking a processed column from the primary table (DB 48804 - Protein data), e.g., 'lpg_Protein_Type':")
        try:
            temp_df_primary_check = pd.read_csv(os.path.join(dir_48804_aligned, "lpg_aligned_train.csv"))
            check_col = 'lpg_Protein_Type' # Example column from Ortholog_Lpg1l_Protein_Annotations
            if check_col in temp_df_primary_check.columns:
                 print(f"Number of unique categories in {check_col}: {temp_df_primary_check[check_col].nunique()}")
                 print(f"Sample values for {check_col} (first 5 unique): {temp_df_primary_check[check_col].unique()[:5]}")
                 print(f"Value counts for {check_col} (top 5): \n{temp_df_primary_check[check_col].value_counts().nlargest(5)}")
            else:
                print(f"Column '{check_col}' not found in saved aligned primary data: {temp_df_primary_check.columns.tolist()}")
        except FileNotFoundError:
            print(f"Saved aligned file 'lpg_aligned_train.csv' not found for checking '{check_col}'.")
        except Exception as e:
            print(f"Could not check '{check_col}': {e}")
    else:
        print("\nSkipping merge: Primary dataframe (DB 48804) is empty after loading.")