import os
import json
import sqlite3
import glob
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import attrs
import cattrs
from tqdm import tqdm
import argparse


@attrs.define
class ForeignKey:
    column_name: str
    reference_column_name: str
    reference_table_name: str


@attrs.define
class Column:
    column_name: str
    alternative_column_names: list[str]
    data_type: str
    wikidata_property_id: str | None


@attrs.define
class Table:
    table_name: str
    alternative_table_names: list[str]
    file_name: str
    columns: list[Column]
    foreign_keys: list[ForeignKey]


@attrs.define
class Schema:
    database_name: str
    alternative_database_names: list[str]
    wikidata_property_id: str
    wikidata_property_label: str
    wikidata_topic_item_id: str
    wikidata_topic_item_label: str
    tables: list[Table]


def split_into_chunks(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def build_table_graph(schema: Schema):
    graph = defaultdict(set)
    for table in schema.tables:
        for fk in table.foreign_keys:
            graph[table.table_name].add(fk.reference_table_name)
            graph[fk.reference_table_name].add(table.table_name)
    return graph


def find_connected_components(graph):
    visited = set()
    components = []

    def dfs(node, component):
        visited.add(node)
        component.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor, component)

    for node in graph:
        if node not in visited:
            comp = []
            dfs(node, comp)
            components.append(comp)
    return components


def build_join_sql(schema: Schema, tables_subset):
    table_map = {table.table_name: table for table in schema.tables}
    joins = []
    used_tables = set(tables_subset)

    for table_name in tables_subset:
        table = table_map[table_name]
        for fk in table.foreign_keys:
            if fk.reference_table_name in tables_subset:
                joins.append((table.table_name, fk.column_name, fk.reference_table_name, fk.reference_column_name))

    if not joins:
        return None

    base_table = joins[0][0]
    sql = f'SELECT COUNT(*) FROM "{base_table}"'
    joined = set([base_table])

    for t1, col1, t2, col2 in joins:
        if t2 in joined and t1 not in joined:
            sql += f' JOIN "{t1}" ON "{t1}"."{col1}" = "{t2}"."{col2}"'
            joined.add(t1)
        elif t1 in joined and t2 not in joined:
            sql += f' JOIN "{t2}" ON "{t1}"."{col1}" = "{t2}"."{col2}"'
            joined.add(t2)

    return sql


def build_and_count(db_path):
    try:
        with open(f"{db_path}/schema.json", "r", encoding="utf-8") as file:
            schema = cattrs.structure(json.load(file), Schema)

        sqlite_path = f":memory:"

        with sqlite3.connect(sqlite_path) as conn:
            for table in schema.tables:
                ct_columns = [f'"{col.column_name}"' for col in table.columns]
                fk_constraints = [
                    f'FOREIGN KEY("{fk.column_name}") REFERENCES "{fk.reference_table_name}"("{fk.reference_column_name}")'
                    for fk in table.foreign_keys
                ]
                create_sql = f'CREATE TABLE "{table.table_name}" ( {", ".join(ct_columns + fk_constraints)} );'
                conn.execute(create_sql)
                conn.commit()

                df = pd.read_csv(f"{db_path}/tables/{table.file_name}")
                try:
                    df.to_sql(table.table_name, conn, if_exists="append", index=False)
                except OverflowError:
                    conn.execute(f'DELETE FROM "{table.table_name}"')
                    df = df.map(str)
                    df.to_sql(table.table_name, conn, if_exists="append", index=False)

            # compute AllJoinSize by connected components
            graph = build_table_graph(schema)
            components = find_connected_components(graph)
            total = 0
            max_join_tables = 60

            for component in components:
                chunked_results = []
                for chunk in split_into_chunks(component, max_join_tables):
                    join_sql = build_join_sql(schema, chunk)
                    if join_sql:
                        try:
                            count = conn.execute(join_sql).fetchone()[0]
                            chunked_results.append(count)
                        except Exception as e:
                            print(f"[Join Error on chunk] {db_path}: {e}")
                            continue
                if chunked_results:
                    total += sum(chunked_results)
            return total if total > 0 else None
    except Exception as e:
        print(f"[ERROR] {db_path}: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch-size", type=int, default=100)
    parser.add_argument("-t", "--num-threads", type=int, default=64)
    args = parser.parse_args()

    db_dirs = sorted(glob.glob("data/unzip/*"))
    results = []
    batch_size = args.batch_size
    output_file = "data/graph/all_join_size_results.csv"
    temp_output_dir = "data/graph/tmp/"

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Create temp directory if it doesn't exist
    if not os.path.exists(temp_output_dir):
        os.makedirs(temp_output_dir)
        
    # Process databases in batches with multiple threads
    total_dbs = len(db_dirs)
    total_batches = (total_dbs + batch_size - 1) // batch_size
    
    def process_batch(batch_id):
        thread_results = []
        start_idx = batch_id * batch_size
        end_idx = min(start_idx + batch_size, total_dbs)
        batch_dbs = db_dirs[start_idx:end_idx]
        
        for db_dir in batch_dbs:
            count = build_and_count(db_dir)
            if count is not None:
                db_id = os.path.basename(db_dir).split()[0]
                thread_results.append({"db_id": db_id, "all_join_size": count})
        
        # Save to temp file
        temp_df = pd.DataFrame(thread_results)
        temp_df.to_csv(f"{temp_output_dir}/all_join_size_results.{batch_id}.csv", index=False)
        
        return thread_results
    
    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = [executor.submit(process_batch, i) for i in range(total_batches)]
        
        # Use tqdm to show progress as futures complete
        for future in tqdm(as_completed(futures), total=total_batches, desc="Processing databases"):
            results.extend(future.result())
    
    # Final save
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)


    print(f"\n======== AllJoinSize Stats (Connected Components) ========")
    print(f"Results saved to: {output_file}")
    print(f"Total databases processed: {len(df)}")
    print(f"Min: {df['all_join_size'].min()}")
    print(f"Max: {df['all_join_size'].max()}")
    print(f"Mean: {df['all_join_size'].mean()}")
    print(f"Median: {df['all_join_size'].median()}")