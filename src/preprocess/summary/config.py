"""
Centralized plot configuration for consistent styling across all figures.

This configuration is used for academic paper figures (top CS conferences).
All plot scripts should import from this file for consistency.
"""

# =============================================================================
# Font Sizes (Large for academic papers)
# =============================================================================
FONTSIZE = {
    "title": 22,
    "axis_label": 18,
    "tick_label": 16,
    "legend": 16,
    "annotation": 14,
}

# =============================================================================
# Line Properties
# =============================================================================
LINEWIDTH = {
    "main": 2.5,
    "baseline": 2.0,
    "diagonal": 1.5,
}

MARKERSIZE = {
    "main": 10,
    "secondary": 9,
}

# =============================================================================
# Colors - Semantically organized
# =============================================================================
COLORS = {
    # Primary metrics
    "auc": "#2E86AB",           # Blue - primary metric
    "f1": "#E94F37",            # Red/Orange - secondary metric
    "accuracy": "#28A745",      # Green
    
    # Model comparison
    "finetuned": "#9d4f8e",     # Purple - finetuned model
    "original": "#4f9d8e",      # Teal - original/baseline model
    
    # Baselines
    "baseline": "#666666",      # Gray - baseline reference
    "baseline_secondary": "#999999",
    
    # Ablation studies - distinct colors
    "ablation_1": "#2196F3",    # Blue
    "ablation_2": "#28A745",    # Green
    "ablation_3": "#FF5722",    # Deep Orange
    "ablation_4": "#9C27B0",    # Purple
}

# =============================================================================
# Markers
# =============================================================================
MARKERS = {
    "primary": "o",      # Circle
    "secondary": "s",    # Square
    "tertiary": "^",     # Triangle up
    "quaternary": "D",   # Diamond
}

# =============================================================================
# Plot Style
# =============================================================================
STYLE = "seaborn-v0_8-whitegrid"

GRID = {
    "alpha": 0.4,
    "linestyle": "--",
}

# Error bar configuration
ERRORBAR = {
    "capsize": 6,
    "capthick": 2,
}

# Figure DPI for saving
DPI = 300
