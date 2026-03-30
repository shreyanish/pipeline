import pandas as pd
import numpy as np

# Load the existing batch results
df = pd.read_csv('batch_evaluation_results.csv')

# Split the 'Algorithm' column into 'Region' and 'Method'
# The format is region_name_METHOD (e.g., upper_medial_forehead_POS)
def split_algo(name):
    parts = name.rsplit('_', 1)
    return parts[0], parts[1]

df[['Region', 'Method']] = df['Algorithm'].apply(lambda x: pd.Series(split_algo(x)))

# 1. Top 5 Performing Regions (across all methods) based on SNR
region_snr = df.groupby('Region')['SNR'].mean().sort_values(ascending=False)
top_5_regions = region_snr.head(5)
bot_5_regions = region_snr.tail(5)

# 2. Top 3 Algorithms (across all regions)
method_snr = df.groupby('Method')['SNR'].mean().sort_values(ascending=False)

# 3. Specific Comparison: Skin vs Face logic
# Face Regions: eyes, nasal tip, temporal lobes
face_list = ['right_eye', 'left_eye', 'nasal_tip', 'right_temporal_lobe', 'left_temporal_lobe']
skin_df = df[~df['Region'].isin(face_list)]
face_df = df[df['Region'].isin(face_list)]

skin_avg_snr = skin_df['SNR'].mean()
face_avg_snr = face_df['SNR'].mean()

print("--- REGION ANALYSIS (SNR) ---")
print(f"Top 5 Regions:\n{top_5_regions.to_string()}\n")
print(f"Bottom 5 Regions:\n{bot_5_regions.to_string()}\n")

print("--- ALGORITHM ANALYSIS (SNR) ---")
print(f"Ranked Methods:\n{method_snr.to_string()}\n")

print("--- SKIN VS FACE CONTRAST ---")
print(f"Skin ROI Avg SNR: {skin_avg_snr:.4f}")
print(f"Face ROI Avg SNR: {face_avg_snr:.4f}")
print(f"Improvement: {skin_avg_snr - face_avg_snr:.4f} dB\n")

# 4. Variance Analysis (Signal Quality)
region_var = df.groupby('Region')['Variance'].mean().sort_values(ascending=True)
print("--- CLEANEST REGIONS (LOWEST VARIANCE) ---")
print(region_var.head(5).to_string())
