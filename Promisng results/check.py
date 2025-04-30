import pandas as pd
import os

# Load the exported features file

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path1 = os.path.join(script_dir, "features.csv")
df = pd.read_csv(csv_path1)

# Show the first few rows
print("=== Data Preview ===")
print(df.head())

# Summary statistics
print("\n=== Descriptive Statistics ===")
print(df.describe())

# Columns for counts and compositions
amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
count_cols = [f"count_{aa}" for aa in amino_acids]
comp_cols  = [f"comp_{aa}"  for aa in amino_acids]

# Check: sum of counts == seq_length
df["sum_counts"] = df[count_cols].sum(axis=1)
mismatch_counts = df[df["sum_counts"] != df["seq_length"]]
print(f"\nRows where sum(counts) != seq_length: {len(mismatch_counts)}")

# Check: sum of compositions ≈ 1
df["sum_comps"] = df[comp_cols].sum(axis=1)
mismatch_comps = df[~df["sum_comps"].between(0.999, 1.001)]
print(f"Rows where sum(comps) not ≈ 1: {len(mismatch_comps)}")

# Check for missing values
missing = df.isnull().sum()
print("\n=== Missing Values per Column ===")
print(missing[missing > 0])

# Class distribution
print("\n=== Class Distribution ===")
print(df["class"].value_counts())

