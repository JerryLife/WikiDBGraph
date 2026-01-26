#!/bin/bash

# Download script for GitTables 1M dataset from Zenodo
# Source: https://zenodo.org/records/6517052
# Supports parallel downloads for faster execution

set -e

# Define output directory and parallel jobs
OUTPUT_DIR="${1:-data/GitTables}"
PARALLEL_JOBS="${2:-4}"  # Default to 4 parallel downloads

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "=== GitTables 1M Download Script ==="
echo "Output directory: $OUTPUT_DIR"
echo "Parallel downloads: $PARALLEL_JOBS"
echo ""

# Base URL for downloads
BASE_URL="https://zenodo.org/records/6517052/files"

# List of all files to download
FILES=(
    "abstraction_tables_licensed.zip"
    "allegro_con_spirito_tables_licensed.zip"
    "attrition_rate_tables_licensed.zip"
    "beats_per_minute_tables_licensed.zip"
    "beauty_sleep_tables_licensed.zip"
    "bits_per_second_tables_licensed.zip"
    "cardiac_output_tables_licensed.zip"
    "cease_tables_licensed.zip"
    "centripetal_acceleration_tables_licensed.zip"
    "channel_capacity_tables_licensed.zip"
    "clotting_time_tables_licensed.zip"
    "command_processing_overhead_time_tables_licensed.zip"
    "count_per_minute_tables_licensed.zip"
    "crime_rate_tables_licensed.zip"
    "data_rate_tables_licensed.zip"
    "dead_air_tables_licensed.zip"
    "dogwatch_tables_licensed.zip"
    "dose_rate_tables_licensed.zip"
    "dwarf_tables_licensed.zip"
    "entr'acte_tables_licensed.zip"
    "episcopate_tables_licensed.zip"
    "erythrocyte_sedimentation_rate_tables_licensed.zip"
    "escape_velocity_tables_licensed.zip"
    "fertile_period_tables_licensed.zip"
    "graveyard_watch_tables_licensed.zip"
    "growth_rate_tables_licensed.zip"
    "half_life_tables_licensed.zip"
    "halftime_tables_licensed.zip"
    "heterotroph_tables_licensed.zip"
    "hypervelocity_tables_licensed.zip"
    "id_tables_licensed.zip"
    "in_time_tables_licensed.zip"
    "incubation_period_tables_licensed.zip"
    "indiction_tables_licensed.zip"
    "inflation_rate_tables_licensed.zip"
    "interim_tables_licensed.zip"
    "kilohertz_tables_licensed.zip"
    "kilometers_per_hour_tables_licensed.zip"
    "lapse_tables_licensed.zip"
    "last_gasp_tables_licensed.zip"
    "latent_period_tables_licensed.zip"
    "lead_time_tables_licensed.zip"
    "living_thing_tables_licensed.zip"
    "lunitidal_interval_tables_licensed.zip"
    "meno_mosso_tables_licensed.zip"
    "menstrual_cycle_tables_licensed.zip"
    "metabolic_rate_tables_licensed.zip"
    "miles_per_hour_tables_licensed.zip"
    "multistage_tables_licensed.zip"
    "musth_tables_licensed.zip"
    "neonatal_mortality_tables_licensed.zip"
    "object_tables_licensed.zip"
    "orbit_period_tables_licensed.zip"
    "organism_tables_licensed.zip"
    "parent_tables_licensed.zip"
    "peacetime_tables_licensed.zip"
    "peculiar_velocity_tables_licensed.zip"
    "physical_entity_tables_licensed.zip"
    "processing_time_tables_licensed.zip"
    "question_time_tables_licensed.zip"
    "quick_time_tables_licensed.zip"
    "radial_pulse_tables_licensed.zip"
    "radial_velocity_tables_licensed.zip"
    "rainy_day_tables_licensed.zip"
    "rate_of_return_tables_licensed.zip"
    "reaction_time_tables_licensed.zip"
    "real_time_tables_licensed.zip"
    "relaxation_time_tables_licensed.zip"
    "respiratory_rate_tables_licensed.zip"
    "return_on_invested_capital_tables_licensed.zip"
    "revolutions_per_minute_tables_licensed.zip"
    "rotational_latency_tables_licensed.zip"
    "running_time_tables_licensed.zip"
    "safe_period_tables_licensed.zip"
    "sampling_frequency_tables_licensed.zip"
    "sampling_rate_tables_licensed.zip"
    "secretory_phase_tables_licensed.zip"
    "seek_time_tables_licensed.zip"
    "shiva_tables_licensed.zip"
    "show_time_tables_licensed.zip"
    "solar_constant_tables_licensed.zip"
    "speed_of_light_tables_licensed.zip"
    "split_shift_tables_licensed.zip"
    "steerageway_tables_licensed.zip"
    "stopping_point_tables_licensed.zip"
    "terminal_velocity_tables_licensed.zip"
    "terminus_ad_quem_tables_licensed.zip"
    "then_tables_licensed.zip"
    "thing_tables_licensed.zip"
    "time-out_tables_licensed.zip"
    "time_interval_tables_licensed.zip"
    "time_slot_tables_licensed.zip"
    "track-to-track_seek_time_tables_licensed.zip"
    "usance_tables_licensed.zip"
    "wartime_tables_licensed.zip"
    "whole_tables_licensed.zip"
)

TOTAL_FILES=${#FILES[@]}
echo "Total files to download: $TOTAL_FILES"
echo ""

# Export variables for use in subshell
export OUTPUT_DIR BASE_URL

# Function to download a single file
download_file() {
    local FILE="$1"
    local OUTPUT_PATH="$OUTPUT_DIR/$FILE"
    
    # Skip if file already exists
    if [ -f "$OUTPUT_PATH" ]; then
        echo "[SKIP] $FILE (already exists)"
        return 0
    fi
    
    # URL encode the filename for special characters (like apostrophe)
    local ENCODED_FILE
    ENCODED_FILE=$(python3 -c "import urllib.parse, sys; print(urllib.parse.quote(sys.argv[1]))" "$FILE")
    local DOWNLOAD_URL="$BASE_URL/$ENCODED_FILE?download=1"
    
    echo "[DOWNLOADING] $FILE"
    
    # Download with wget, resume support, and retries
    if wget -c -q --show-progress --retry-connrefused --tries=5 --timeout=30 \
        -O "$OUTPUT_PATH" "$DOWNLOAD_URL" 2>&1; then
        echo "[OK] $FILE"
        return 0
    else
        echo "[FAILED] $FILE"
        # Remove partial download
        rm -f "$OUTPUT_PATH"
        return 1
    fi
}

export -f download_file

# Use GNU parallel if available, otherwise fall back to xargs
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel with $PARALLEL_JOBS jobs..."
    echo ""
    printf '%s\n' "${FILES[@]}" | parallel -j "$PARALLEL_JOBS" download_file {}
else
    echo "Using xargs with $PARALLEL_JOBS parallel processes..."
    echo "(Install GNU parallel for better progress tracking)"
    echo ""
    printf '%s\0' "${FILES[@]}" | xargs -0 -P "$PARALLEL_JOBS" -I {} bash -c 'download_file "$@"' _ {}
fi

# Count results
DOWNLOADED=$(find "$OUTPUT_DIR" -name "*.zip" -type f | wc -l)
FAILED=$((TOTAL_FILES - DOWNLOADED))

echo ""
echo "=== Download Complete ==="
echo "Successfully downloaded: $DOWNLOADED files"
if [ $FAILED -gt 0 ]; then
    echo "Failed/Skipped downloads: $FAILED files"
fi

echo ""
echo "Files are saved in: $OUTPUT_DIR"
echo ""

# Decompress all zip files
echo "=== Decompressing Files ==="

# Function to decompress a single file
decompress_file() {
    local FILE="$1"
    local ZIP_PATH="$OUTPUT_DIR/$FILE"
    local EXTRACT_DIR="$OUTPUT_DIR/${FILE%.zip}"
    
    # Skip if already extracted
    if [ -d "$EXTRACT_DIR" ]; then
        echo "[SKIP] $FILE (already extracted)"
        return 0
    fi
    
    if [ ! -f "$ZIP_PATH" ]; then
        echo "[SKIP] $FILE (zip file not found)"
        return 0
    fi
    
    echo "[EXTRACTING] $FILE"
    
    if unzip -q "$ZIP_PATH" -d "$EXTRACT_DIR" 2>&1; then
        echo "[OK] $FILE -> ${FILE%.zip}/"
        return 0
    else
        echo "[FAILED] $FILE"
        return 1
    fi
}

export -f decompress_file

# Use GNU parallel if available, otherwise fall back to xargs
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel with $PARALLEL_JOBS jobs for extraction..."
    echo ""
    printf '%s\n' "${FILES[@]}" | parallel -j "$PARALLEL_JOBS" decompress_file {}
else
    echo "Using xargs with $PARALLEL_JOBS parallel processes for extraction..."
    echo ""
    printf '%s\0' "${FILES[@]}" | xargs -0 -P "$PARALLEL_JOBS" -I {} bash -c 'decompress_file "$@"' _ {}
fi

# Count extraction results
EXTRACTED=$(find "$OUTPUT_DIR" -maxdepth 1 -type d ! -name "$(basename "$OUTPUT_DIR")" | wc -l)

echo ""
echo "=== Extraction Complete ==="
echo "Successfully extracted: $EXTRACTED directories"
echo ""
echo "Files are saved in: $OUTPUT_DIR"
