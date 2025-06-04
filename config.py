"""
config.py
Central configuration for required columns and color palettes used in the analysis and visualizations.
- REQUIRED_COLUMNS: List of columns expected in the input data
- PALETTE_*: Color palette names for consistent plotting
"""

# List of required columns for the analysis
REQUIRED_COLUMNS = [
    'YEAR', 'MONTH', 'VALUE', 'QUANTITY', 'LOCATION TYPE', 'TYPE OF LOSS', 'PROPERTY CATEGORY',
    'S.RACE', 'S.GENDER', 'V.RACE', 'V.GENDER', 'VICTIM TYPE', 'WEAPON', 'DESCRIPTION', 'CHARGE TYPE'
]

# Example color palettes for plots (Seaborn/Matplotlib)
PALETTE_CREST = 'crest'
PALETTE_HUSL = 'husl'
PALETTE_VIRIDIS = 'viridis'
PALETTE_MAGMA = 'magma'
PALETTE_CUBEHELIX = 'cubehelix'
PALETTE_TAB10 = 'tab10' 