#!/usr/bin/env python3
"""
Script to add visualization save paths to complete_analysis.ipynb
"""

import json
import re

# Load the notebook
with open('complete_analysis.ipynb', 'r') as f:
    nb = json.load(f)

# Mapping of visualization search patterns to output paths
visualization_map = {
    'Distribution of Numerical Features': 'results/figures/univariate/01_numerical_distributions.png',
    'Box Plots - Numerical Features (Outlier Detection)': 'results/figures/univariate/02_boxplots_outlier_detection.png',
    'Distribution of Key Categorical Features': 'results/figures/univariate/03_categorical_distributions.png',
    'Severity Distribution (Percentage)': 'results/figures/univariate/04_severity_distribution.png',
    'POI Features - Presence in Accident Locations': 'results/figures/univariate/05_poi_features_prevalence.png',
    'Total Accidents by Hour': 'results/figures/bivariate/01_accidents_by_hour.png',
    'Accidents by Hour and Severity Level': 'results/figures/bivariate/02_severity_by_hour.png',
    'Total Accidents by Day of Week': 'results/figures/bivariate/03_accidents_by_day_of_week.png',
    'Average Accident Severity by Day': 'results/figures/bivariate/04_severity_by_day_of_week.png',
    "Road Infrastructure Features by Severity Level": 'results/figures/bivariate/05_poi_vs_severity.png',
    'Correlation Heatmap - Numerical Features': 'results/figures/bivariate/06_correlation_heatmap.png',
    'Severity Distribution by Weather Condition': 'results/figures/bivariate/07_weather_vs_severity.png',
    'Top 10 States by Accident Count': 'results/figures/bivariate/08_geographic_analysis.png',
}

added_imports = False

for cell in nb['cells']:
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        
        # Convert source to string
        if isinstance(source, list):
            source_str = ''.join(source)
        else:
            source_str = source
        
        # Add imports to first code cell if not already added
        if not added_imports and any(key in source_str for key in visualization_map.keys()):
            # Check if import os already exists
            if 'import os' not in source_str and 'os.makedirs' not in source_str:
                import_lines = "import os\nos.makedirs('results/figures/univariate', exist_ok=True)\nos.makedirs('results/figures/bivariate', exist_ok=True)\n\n"
                if isinstance(cell['source'], list):
                    cell['source'].insert(0, import_lines)
                else:
                    cell['source'] = import_lines + cell['source']
                added_imports = True
        
        # Add savefig commands before plt.show()
        for viz_name, save_path in visualization_map.items():
            if viz_name in source_str and 'plt.show()' in source_str:
                # Replace plt.show() with savefig + show
                if isinstance(cell['source'], list):
                    new_source = []
                    for line in cell['source']:
                        if line.strip() == 'plt.show()':
                            new_source.append(f"plt.savefig('{save_path}', dpi=300, bbox_inches='tight')\n")
                        new_source.append(line)
                    cell['source'] = new_source
                else:
                    source_str = cell['source'].replace(
                        'plt.show()',
                        f"plt.savefig('{save_path}', dpi=300, bbox_inches='tight')\nplt.show()"
                    )
                    cell['source'] = source_str

# Save the modified notebook
with open('data_wrangling.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("✓ Successfully updated data_wrangling.ipynb!")
print("✓ Univariate visualizations will be saved to: results/figures/univariate/")
print("✓ Bivariate visualizations will be saved to: results/figures/bivariate/")
print("\nVisualization files that will be created:")
print("  Univariate:")
print("    - 01_numerical_distributions.png")
print("    - 02_boxplots_outlier_detection.png")
print("    - 03_categorical_distributions.png")
print("    - 04_severity_distribution.png")
print("    - 05_poi_features_prevalence.png")
print("  Bivariate:")
print("    - 01_accidents_by_hour.png")
print("    - 02_severity_by_hour.png")
print("    - 03_accidents_by_day_of_week.png")
print("    - 04_severity_by_day_of_week.png")
print("    - 05_poi_vs_severity.png")
print("    - 06_correlation_heatmap.png")
print("    - 07_weather_vs_severity.png")
print("    - 08_geographic_analysis.png")
