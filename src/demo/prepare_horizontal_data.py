import pandas as pd
import sqlite3
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np


# Label mapping for gene classification
LABEL_MAP = {
    "protein-coding gene": 1,
    "pseudogene": 0
}

# Mapping for database 02799 (Trypanosoma_Cruzi_Orthologs_Db_30)
DB_02799_MAPPING = {
    # Columns from DB1 Main Table: Gene_Orthologs
    # (Conceptual Keys match actual column names from Gene_Orthologs)
    'Gene_Id': 'gene_id',
    'Gene_Description': 'gene_description',
    'Entity_Type': 'entity_type',  # Harmonized to generic entity_type
    'Species': 'species',
    'Chromosome_Id': 'chromosome_id',
    'Strand_Orientation': 'strand_orientation',
    'Gene_Db_Id': 'gene_db_id',
    'Genomic_Start_Position': 'genomic_start_position',
    'Genomic_End_Position': 'genomic_end_position',
    'Gene_Type': 'gene_type',
    'Encoded_Protein': 'encoded_protein',
    'Ortholog_Id': 'ortholog_id',  # Represents the ID of the ortholog, FK in this table

    # Columns from DB1 Details Table: Gene_Orthologs_Details
    # (Conceptual Keys are prefixed with "Details_")
    'Details_Gene_Label': 'details_gene_label', # Corresponds to Gene_Orthologs_Details.Gene_Label
    'Details_Gene_Description': 'details_gene_description', # Corresponds to Gene_Orthologs_Details.Gene_Description
    'Details_Entity_Type': 'details_gene_type', # Corresponds to Gene_Orthologs_Details.Entity_Type
    'Details_Host_Taxon': 'details_host_taxon', # Corresponds to Gene_Orthologs_Details.Host_Taxon
    'Details_Chromosome_Location': 'details_chromosome_location', # Corresponds to Gene_Orthologs_Details.Chromosome_Location
    'Details_Strand_Orientation': 'details_strand_orientation', # Corresponds to Gene_Orthologs_Details.Strand_Orientation
    'Details_Gene_Db_Id': 'details_gene_db_id', # Corresponds to Gene_Orthologs_Details.Gene_Db_Id
    'Details_Genomic_Start_Position': 'details_genomic_start_position', # Corresponds to Gene_Orthologs_Details.Genomic_Start_Position
    'Details_Genomic_End_Position': 'details_genomic_end_position', # Corresponds to Gene_Orthologs_Details.Genomic_End_Position
    'Details_Gene_Classification': 'details_gene_classification', # Corresponds to Gene_Orthologs_Details.Gene_Classification
    'Details_Encoded_Protein': 'details_encoded_protein', # Corresponds to Gene_Orthologs_Details.Encoded_Protein
    'Details_Orthologous_Gene': 'details_orthologous_gene_fk' # Corresponds to Gene_Orthologs_Details.Orthologous_Gene (FK)
}

# Mapping for database 79665 (TrypanosomaCruziOrthologs225)
DB_79665_MAPPING = {
    # Columns from DB2 Details Table: GeneOrthologs
    # (Conceptual Keys match actual column names from GeneOrthologs)
    # These should map to "details_*" harmonized names.
    'GeneLabel': 'details_gene_label', # Corresponds to GeneOrthologs.GeneLabel
    'GeneDescription': 'details_gene_description', # Corresponds to GeneOrthologs.GeneDescription
    'GeneType': 'details_gene_type', # Corresponds to GeneOrthologs.GeneType (Note: DB1 Details doesn't have this exact field)
    'HostTaxon': 'details_host_taxon', # Corresponds to GeneOrthologs.HostTaxon
    'ChromosomeLocation': 'details_chromosome_location', # Corresponds to GeneOrthologs.ChromosomeLocation
    'StrandOrientation': 'details_strand_orientation', # Corresponds to GeneOrthologs.StrandOrientation
    'GeneDbId': 'details_gene_db_id', # Corresponds to GeneOrthologs.GeneDbId
    'GenomicStartPosition': 'details_genomic_start_position', # Corresponds to GeneOrthologs.GenomicStartPosition
    'GenomicEndPosition': 'details_genomic_end_position', # Corresponds to GeneOrthologs.GenomicEndPosition
    'GeneClassification': 'details_gene_classification', # Corresponds to GeneOrthologs.GeneClassification
    'EncodedProtein': 'details_encoded_protein', # Corresponds to GeneOrthologs.EncodedProtein
    'OrthologousGene': 'details_orthologous_gene_fk', # Corresponds to GeneOrthologs.OrthologousGene (FK)

    # Columns from DB2 Main Table: TrypanosomaCruziGeneOrthologs
    # (Conceptual Keys are prefixed with "Details_" in your example, mapping to actual columns)
    # These should map to "gene_*" harmonized names.
    'Details_GeneId': 'gene_id', # Corresponds to TrypanosomaCruziGeneOrthologs.GeneId
    'Details_GeneDescription': 'gene_description', # Corresponds to TrypanosomaCruziGeneOrthologs.GeneDescription
    'Details_EntityType': 'entity_type', # Corresponds to TrypanosomaCruziGeneOrthologs.EntityType
    'Details_Species': 'species', # Corresponds to TrypanosomaCruziGeneOrthologs.Species
    'Details_ChromosomeId': 'chromosome_id', # Corresponds to TrypanosomaCruziGeneOrthologs.ChromosomeId
    'Details_StrandOrientation': 'strand_orientation', # Corresponds to TrypanosomaCruziGeneOrthologs.StrandOrientation
    'Details_GeneDbId': 'gene_db_id', # Corresponds to TrypanosomaCruziGeneOrthologs.GeneDbId
    'Details_GenomicStartPosition': 'genomic_start_position', # Corresponds to TrypanosomaCruziGeneOrthologs.GenomicStartPosition
    'Details_GenomicEndPosition': 'genomic_end_position', # Corresponds to TrypanosomaCruziGeneOrthologs.GenomicEndPosition
    'Details_GeneType': 'gene_type', # Corresponds to TrypanosomaCruziGeneOrthologs.GeneType
    # This key was 'Details_OrthologId' in your example for the Main table's FK.
    # The actual column is 'OrthologId' in TrypanosomaCruziGeneOrthologs.
    # And 'EncodedProtein' is also a column.
    # Assuming your conceptual 'Details_OrthologId' maps to TrypanosomaCruziGeneOrthologs.OrthologId
    'Details_OrthologId': 'ortholog_id', # Corresponds to TrypanosomaCruziGeneOrthologs.OrthologId
    # Assuming your conceptual 'Details_EncodedProtein' maps to TrypanosomaCruziGeneOrthologs.EncodedProtein
    'Details_EncodedProtein': 'encoded_protein' # Corresponds to TrypanosomaCruziGeneOrthologs.EncodedProtein
}

def load_database_02799():
    """
    Load the database with id 02799
    """
    database_dir = os.path.join("data", "unziplink", "02799-Trypanosoma_Cruzi_Orthologs_Db_30")
    database_path = os.path.join(database_dir, "database.db")

    if not os.path.exists(database_path):
        raise FileNotFoundError(f"Database file not found at {database_path}")

    with sqlite3.connect(database_path) as conn:
        # Get schema information
        cursor = conn.cursor()
        
        # Query the tables using a LEFT JOIN
        query = """
        SELECT 
            go.Gene_Id, 
            go.Gene_Description,
            go.Entity_Type,
            go.Species,
            go.Chromosome_Id,
            go.Strand_Orientation,
            go.Gene_Db_Id,
            go.Genomic_Start_Position,
            go.Genomic_End_Position,
            go.Gene_Type,
            go.Encoded_Protein,
            go.Ortholog_Id,
            god.Gene_Label as Details_Gene_Label,
            god.Gene_Description as Details_Gene_Description,
            god.Entity_Type as Details_Entity_Type,
            god.Host_Taxon as Details_Host_Taxon,
            god.Chromosome_Location as Details_Chromosome_Location,
            god.Strand_Orientation as Details_Strand_Orientation,
            god.Gene_Db_Id as Details_Gene_Db_Id,
            god.Genomic_Start_Position as Details_Genomic_Start_Position,
            god.Genomic_End_Position as Details_Genomic_End_Position,
            god.Gene_Classification as Details_Gene_Classification,
            god.Encoded_Protein as Details_Encoded_Protein,
            god.Orthologous_Gene as Details_Orthologous_Gene
        FROM Gene_Orthologs go
        LEFT JOIN Gene_Orthologs_Details god
        ON go.Gene_Id = god.Orthologous_Gene
        """
        
        # Execute the query and convert to pandas DataFrame
        joined_data = pd.read_sql_query(query, conn)
        
        # Remove duplicate columns from the join - no need for this since we're explicitly naming columns
        # columns_to_drop = [col for col in joined_data.columns 
        #                  if col.startswith('Orthologous_Gene') and col != 'Orthologous_Gene']
        # data_matrix = joined_data.drop(columns=columns_to_drop, errors='ignore')
        data_matrix = joined_data
        
        # Rename columns to standardized format
        renamed_columns = {}
        for col in data_matrix.columns:
            if col in DB_02799_MAPPING:
                renamed_columns[col] = DB_02799_MAPPING[col]
        
        standardized_df = data_matrix.rename(columns=renamed_columns)

        # rename columns to standardized format
        standardized_df = standardized_df[sorted(DB_02799_MAPPING.values())]
        
        # Encode all other categorical features
        categorical_columns = standardized_df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col != 'gene_classification':  # Skip if already mapped
                le = LabelEncoder()
                # Convert to string representation if the column contains arrays/lists
                if standardized_df[col].apply(lambda x: isinstance(x, (list, tuple, np.ndarray))).any():
                    standardized_df[col] = standardized_df[col].apply(lambda x: str(x))
                standardized_df[col] = le.fit_transform(standardized_df[col].astype(str))
        
        # Convert numeric columns to appropriate types
        numeric_columns = standardized_df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_columns:
            standardized_df[col] = pd.to_numeric(standardized_df[col], errors='coerce')
        
        # Split into train and test sets
        train_df, test_df = train_test_split(standardized_df, test_size=0.2, random_state=0)
        
        # Create directory if it doesn't exist
        os.makedirs("data/clean/02799", exist_ok=True)
        
        # Save the datasets with standardized column names
        train_df.to_csv("data/clean/02799/train_data.csv", index=False)
        test_df.to_csv("data/clean/02799/test_data.csv", index=False)
        
        print(f"Training data shape: {train_df.shape}")
        print(f"Testing data shape: {test_df.shape}")


def load_database_79665():
    """
    Load the database with id 79665
    """
    database_dir = os.path.join("data", "unziplink", "79665-TrypanosomaCruziOrthologs225")
    database_path = os.path.join(database_dir, "database.db")

    if not os.path.exists(database_path):
        raise FileNotFoundError(f"Database file not found at {database_path}")

    with sqlite3.connect(database_path) as conn:
        # Get schema information
        cursor = conn.cursor()

        # Query the tables using a LEFT JOIN
        query = """
        SELECT 
            g.GeneLabel,
            g.GeneDescription,
            g.GeneType,
            g.HostTaxon,
            g.ChromosomeLocation,
            g.StrandOrientation,
            g.GeneDbId,
            g.GenomicStartPosition,
            g.GenomicEndPosition,
            g.GeneClassification,
            g.EncodedProtein,
            g.OrthologousGene,
            t.GeneId as Details_GeneId,
            t.GeneDescription as Details_GeneDescription,
            t.EntityType as Details_EntityType,
            t.Species as Details_Species,
            t.ChromosomeId as Details_ChromosomeId,
            t.StrandOrientation as Details_StrandOrientation,
            t.GeneDbId as Details_GeneDbId,
            t.GenomicStartPosition as Details_GenomicStartPosition,
            t.GenomicEndPosition as Details_GenomicEndPosition,
            t.GeneType as Details_GeneType,
            t.OrthologId as Details_OrthologId,
            t.EncodedProtein as Details_EncodedProtein
        FROM GeneOrthologs g
        LEFT JOIN TrypanosomaCruziGeneOrthologs t
        ON g.OrthologousGene = t.GeneId
        """
        
        # Execute the query and convert to pandas DataFrame
        joined_data = pd.read_sql_query(query, conn)
        
        # Remove duplicate columns from the join - no need for this since we're explicitly naming columns
        # columns_to_drop = [col for col in joined_data.columns 
        #                  if col.startswith('OrthologousGene') and col != 'OrthologousGene']
        # data_matrix = joined_data.drop(columns=columns_to_drop, errors='ignore')
        data_matrix = joined_data
        
        # Rename columns to standardized format
        renamed_columns = {}
        for col in data_matrix.columns:
            if col in DB_79665_MAPPING:
                renamed_columns[col] = DB_79665_MAPPING[col]
        
        standardized_df = data_matrix.rename(columns=renamed_columns)

        # rename columns to standardized format
        standardized_df = standardized_df[sorted(DB_79665_MAPPING.values())]
        
        # Encode all other categorical features
        categorical_columns = standardized_df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col != 'gene_classification':  # Skip if already mapped
                le = LabelEncoder()
                # Convert to string representation if the column contains arrays/lists
                if standardized_df[col].apply(lambda x: isinstance(x, (list, tuple, np.ndarray))).any():
                    standardized_df[col] = standardized_df[col].apply(lambda x: str(x))
                standardized_df[col] = le.fit_transform(standardized_df[col].astype(str))
        
        # Convert numeric columns to appropriate types
        numeric_columns = standardized_df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_columns:
            standardized_df[col] = pd.to_numeric(standardized_df[col], errors='coerce')
        
        # Split into train and test sets
        train_df, test_df = train_test_split(standardized_df, test_size=0.2, random_state=0)
        
        # Create directory if it doesn't exist
        os.makedirs("data/clean/79665", exist_ok=True)
        
        # Save the datasets with standardized column names
        train_df.to_csv("data/clean/79665/train_data.csv", index=False)
        test_df.to_csv("data/clean/79665/test_data.csv", index=False)
        
        print(f"Training data shape: {train_df.shape}")
        print(f"Testing data shape: {test_df.shape}")


def prepare_horizontal_data_with_unified_encoding():
    """
    Prepare horizontal federated learning data with unified label encoding
    to ensure label consistency across clients.
    Follows the exact same logic as load_database_02799 and load_database_79665
    but with unified categorical encoding.
    """
    print("=== PREPARING HORIZONTAL DATA WITH UNIFIED LABEL ENCODING ===")
    
    # Step 1: Load and process data from both databases using the exact same logic
    print("Step 1: Loading and processing data from both databases...")
    
    # ==================== PROCESS DATABASE 02799 ====================
    database_dir_02799 = os.path.join("data", "unziplink", "02799-Trypanosoma_Cruzi_Orthologs_Db_30")
    database_path_02799 = os.path.join(database_dir_02799, "database.db")

    if not os.path.exists(database_path_02799):
        raise FileNotFoundError(f"Database file not found at {database_path_02799}")

    with sqlite3.connect(database_path_02799) as conn_02799:
        # Use the exact same query as load_database_02799
        query_02799 = """
        SELECT 
            go.Gene_Id, 
            go.Gene_Description,
            go.Entity_Type,
            go.Species,
            go.Chromosome_Id,
            go.Strand_Orientation,
            go.Gene_Db_Id,
            go.Genomic_Start_Position,
            go.Genomic_End_Position,
            go.Gene_Type,
            go.Encoded_Protein,
            go.Ortholog_Id,
            god.Gene_Label as Details_Gene_Label,
            god.Gene_Description as Details_Gene_Description,
            god.Entity_Type as Details_Entity_Type,
            god.Host_Taxon as Details_Host_Taxon,
            god.Chromosome_Location as Details_Chromosome_Location,
            god.Strand_Orientation as Details_Strand_Orientation,
            god.Gene_Db_Id as Details_Gene_Db_Id,
            god.Genomic_Start_Position as Details_Genomic_Start_Position,
            god.Genomic_End_Position as Details_Genomic_End_Position,
            god.Gene_Classification as Details_Gene_Classification,
            god.Encoded_Protein as Details_Encoded_Protein,
            god.Orthologous_Gene as Details_Orthologous_Gene
        FROM Gene_Orthologs go
        LEFT JOIN Gene_Orthologs_Details god
        ON go.Gene_Id = god.Orthologous_Gene
        """
        
        joined_data_02799 = pd.read_sql_query(query_02799, conn_02799)
        
        # Rename columns to standardized format - exact same logic
        renamed_columns_02799 = {}
        for col in joined_data_02799.columns:
            if col in DB_02799_MAPPING:
                renamed_columns_02799[col] = DB_02799_MAPPING[col]
        
        standardized_df_02799 = joined_data_02799.rename(columns=renamed_columns_02799)
        standardized_df_02799 = standardized_df_02799[sorted(DB_02799_MAPPING.values())]

    # ==================== PROCESS DATABASE 79665 ====================
    database_dir_79665 = os.path.join("data", "unziplink", "79665-TrypanosomaCruziOrthologs225")
    database_path_79665 = os.path.join(database_dir_79665, "database.db")

    if not os.path.exists(database_path_79665):
        raise FileNotFoundError(f"Database file not found at {database_path_79665}")

    with sqlite3.connect(database_path_79665) as conn_79665:
        # Use the exact same query as load_database_79665
        query_79665 = """
        SELECT 
            g.GeneLabel,
            g.GeneDescription,
            g.GeneType,
            g.HostTaxon,
            g.ChromosomeLocation,
            g.StrandOrientation,
            g.GeneDbId,
            g.GenomicStartPosition,
            g.GenomicEndPosition,
            g.GeneClassification,
            g.EncodedProtein,
            g.OrthologousGene,
            t.GeneId as Details_GeneId,
            t.GeneDescription as Details_GeneDescription,
            t.EntityType as Details_EntityType,
            t.Species as Details_Species,
            t.ChromosomeId as Details_ChromosomeId,
            t.StrandOrientation as Details_StrandOrientation,
            t.GeneDbId as Details_GeneDbId,
            t.GenomicStartPosition as Details_GenomicStartPosition,
            t.GenomicEndPosition as Details_GenomicEndPosition,
            t.GeneType as Details_GeneType,
            t.OrthologId as Details_OrthologId,
            t.EncodedProtein as Details_EncodedProtein
        FROM GeneOrthologs g
        LEFT JOIN TrypanosomaCruziGeneOrthologs t
        ON g.OrthologousGene = t.GeneId
        """
        
        joined_data_79665 = pd.read_sql_query(query_79665, conn_79665)
        
        # Rename columns to standardized format - exact same logic
        renamed_columns_79665 = {}
        for col in joined_data_79665.columns:
            if col in DB_79665_MAPPING:
                renamed_columns_79665[col] = DB_79665_MAPPING[col]
        
        standardized_df_79665 = joined_data_79665.rename(columns=renamed_columns_79665)
        standardized_df_79665 = standardized_df_79665[sorted(DB_79665_MAPPING.values())]

    print(f"Loaded DB 02799: {standardized_df_02799.shape}")
    print(f"Loaded DB 79665: {standardized_df_79665.shape}")

    # Step 2: Create unified label encoders for categorical columns
    print("\nStep 2: Creating unified encoders for categorical columns...")
    
    # Get categorical columns from both datasets
    categorical_columns_02799 = standardized_df_02799.select_dtypes(include=['object']).columns
    categorical_columns_79665 = standardized_df_79665.select_dtypes(include=['object']).columns
    all_categorical_columns = list(set(categorical_columns_02799) | set(categorical_columns_79665))
    
    print(f"Found categorical columns: {all_categorical_columns}")
    
    # Create unified encoders for each categorical column
    unified_encoders = {}
    
    for col in all_categorical_columns:
        if col != 'gene_classification':  # Skip if already mapped
            print(f"  Creating unified encoder for column: {col}")
            
            # Collect all unique values from both datasets
            values_02799 = []
            values_79665 = []
            
            if col in standardized_df_02799.columns:
                values_02799 = standardized_df_02799[col].fillna('__MISSING__').astype(str).tolist()
                # Convert to string representation if the column contains arrays/lists
                if standardized_df_02799[col].apply(lambda x: isinstance(x, (list, tuple, np.ndarray))).any():
                    values_02799 = [str(x) for x in values_02799]
            
            if col in standardized_df_79665.columns:
                values_79665 = standardized_df_79665[col].fillna('__MISSING__').astype(str).tolist()
                # Convert to string representation if the column contains arrays/lists
                if standardized_df_79665[col].apply(lambda x: isinstance(x, (list, tuple, np.ndarray))).any():
                    values_79665 = [str(x) for x in values_79665]
            
            # Create unified vocabulary
            all_unique_values = sorted(list(set(values_02799 + values_79665)))
            
            # Create and fit unified encoder
            encoder = LabelEncoder()
            encoder.fit(all_unique_values)
            unified_encoders[col] = encoder
            
            print(f"    Found {len(all_unique_values)} unique values for {col}")

    # Step 3: Apply unified encoders to both datasets (following original logic)
    print("\nStep 3: Applying unified encoders to both datasets...")
    
    # Process DB 02799 with unified encoders
    for col in categorical_columns_02799:
        if col != 'gene_classification' and col in unified_encoders:
            # Convert to string representation if the column contains arrays/lists
            if standardized_df_02799[col].apply(lambda x: isinstance(x, (list, tuple, np.ndarray))).any():
                standardized_df_02799[col] = standardized_df_02799[col].apply(lambda x: str(x))
            
            # Apply unified encoder
            standardized_df_02799[col] = unified_encoders[col].transform(standardized_df_02799[col].fillna('__MISSING__').astype(str))
    
    # Process DB 79665 with unified encoders
    for col in categorical_columns_79665:
        if col != 'gene_classification' and col in unified_encoders:
            # Convert to string representation if the column contains arrays/lists
            if standardized_df_79665[col].apply(lambda x: isinstance(x, (list, tuple, np.ndarray))).any():
                standardized_df_79665[col] = standardized_df_79665[col].apply(lambda x: str(x))
            
            # Apply unified encoder
            standardized_df_79665[col] = unified_encoders[col].transform(standardized_df_79665[col].fillna('__MISSING__').astype(str))
    
    # Convert numeric columns to appropriate types (following original logic)
    numeric_columns_02799 = standardized_df_02799.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_columns_02799:
        standardized_df_02799[col] = pd.to_numeric(standardized_df_02799[col], errors='coerce')
    
    numeric_columns_79665 = standardized_df_79665.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_columns_79665:
        standardized_df_79665[col] = pd.to_numeric(standardized_df_79665[col], errors='coerce')

    # Step 4: Verify label alignment for target column
    print("\nStep 4: Verifying label alignment...")
    target_col = 'details_encoded_protein'
    
    labels_02799 = sorted(standardized_df_02799[target_col].unique())
    labels_79665 = sorted(standardized_df_79665[target_col].unique())
    all_labels = sorted(list(set(labels_02799 + labels_79665)))
    
    print(f"Client 02799 labels: {labels_02799}")
    print(f"Client 79665 labels: {labels_79665}")
    print(f"All unified labels: {all_labels}")
    print(f"✅ Label alignment verified - both clients use the same encoding!")
    
    # Show the distribution per client
    print(f"\nLabel distributions:")
    print(f"Client 02799:")
    for label in labels_02799:
        count = (standardized_df_02799[target_col] == label).sum()
        print(f"  Class {label}: {count} samples")
    
    print(f"Client 79665:")
    for label in labels_79665:
        count = (standardized_df_79665[target_col] == label).sum()
        print(f"  Class {label}: {count} samples")

    # Step 5: Split and save data (following original logic exactly)
    print("\nStep 5: Splitting and saving data...")
    
    # Split DB 02799 (exact same logic as original)
    train_02799, test_02799 = train_test_split(standardized_df_02799, test_size=0.2, random_state=0)
    os.makedirs("data/clean/02799", exist_ok=True)
    train_02799.to_csv("data/clean/02799/train_data.csv", index=False)
    test_02799.to_csv("data/clean/02799/test_data.csv", index=False)
    
    # Split DB 79665 (exact same logic as original)
    train_79665, test_79665 = train_test_split(standardized_df_79665, test_size=0.2, random_state=0)
    os.makedirs("data/clean/79665", exist_ok=True)
    train_79665.to_csv("data/clean/79665/train_data.csv", index=False)
    test_79665.to_csv("data/clean/79665/test_data.csv", index=False)
    
    print(f"✅ Saved training data:")
    print(f"  - DB 02799: {train_02799.shape} train, {test_02799.shape} test")
    print(f"  - DB 79665: {train_79665.shape} train, {test_79665.shape} test")
    
    # Save encoders for future use
    import pickle
    os.makedirs("data/encoders", exist_ok=True)
    with open("data/encoders/unified_label_encoders.pkl", "wb") as f:
        pickle.dump(unified_encoders, f)
    print(f"✅ Saved unified encoders to data/encoders/unified_label_encoders.pkl")
    
    print("\n=== HORIZONTAL DATA PREPARATION WITH UNIFIED ENCODING COMPLETE ===")
    return unified_encoders


if __name__ == "__main__":
    # Use the new unified encoding approach
    try:
        encoders = prepare_horizontal_data_with_unified_encoding()
        print("✅ Successfully prepared horizontal data with unified label encoding!")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Falling back to original method...")
        # Fallback to original approach if databases not found
        load_database_02799()
        load_database_79665()
