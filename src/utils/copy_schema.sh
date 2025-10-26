#!/bin/bash

# Create schema directory if it doesn't exist
mkdir -p data/schema

# Find all schema.json files under data/unzip and copy them
for dir in data/unzip/*; do
    # Extract folder name from path
    folder_name=$(basename "$dir")
    # Replace space with underscore in folder name
    folder_name=$(echo "$folder_name" | sed 's/ /_/g')
    # Copy schema.json to data/schema with modified folder name
    # # Skip if schema already exists
    if [ -f "data/schema/${folder_name}.json" ]; then
        echo "Skipping ${folder_name} because it already exists"
        continue
    fi
    cp "$dir/schema.json" "data/schema/${folder_name}.json"
    echo "Copied '$dir/schema.json' to 'data/schema/${folder_name}.json'"
done
