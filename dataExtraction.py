import pandas as pd
import os


script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path1 = os.path.join(script_dir, "g_data.csv")
csv_path2 = os.path.join(script_dir, "n-data.csv")
g_data = pd.read_csv(csv_path1)  
n_data = pd.read_csv(csv_path2)  


num_proteins_g = len(g_data)
num_proteins_n = len(n_data)
total_proteins = num_proteins_g + num_proteins_n
print(f"Total number of proteins: {total_proteins}")
print(f"\nNumber of proteins in g_data: {num_proteins_g}")
print(f"Number of proteins in n_data: {num_proteins_n}")

labels_g = g_data["Fold1"].unique()
labels_n = n_data["Fold1"].unique()

num_labels_g = len(labels_g)
num_labels_n = len(labels_n)

print(f"\nNumber of unique labels in g_data: {num_labels_g}")
print(f"Number of unique labels in n-data: {num_labels_n}")


proteins_per_class_g = g_data.groupby("Fold1").size()
proteins_per_class_n = n_data.groupby("Fold1").size()

print("\nProteins per class in gram-positive dataset:")
print(proteins_per_class_g)

print("\nProteins per class in gram-negative dataset:")
print(proteins_per_class_n)




def calculate_lengths(data, seq_col):
    data['Length'] = data[seq_col].apply(lambda x: len(str(x)))
    return data.groupby("Fold1")['Length'].mean()

def calculate_max_min_lengths(data, seq_col):
    data['Length'] = data[seq_col].apply(lambda x: len(str(x)))
    max_length = data.groupby("Fold1")['Length'].max()
    min_length = data.groupby("Fold1")['Length'].min()
    return max_length, min_length



# For g_data, sequence is in column index 3
avg_length_g = calculate_lengths(g_data, g_data.columns[3])
max_length_g, min_length_g = calculate_max_min_lengths(g_data, g_data.columns[3])

# For n_data, sequence is in column index 5
avg_length_n = calculate_lengths(n_data, n_data.columns[5])
max_length_n, min_length_n = calculate_max_min_lengths(n_data, n_data.columns[5])


print("\nAverage length of proteins per class in gram-negative dataset:")
print(avg_length_n)

print("\nAverage length of proteins per class in gram-positive dataset:")
print(avg_length_g)

print("\nMax length of proteins per class in gram-negative dataset:")
print(max_length_n)
print("\nMin length of proteins per class in gram-negative dataset:")
print(min_length_n)
print("\nMax length of proteins per class in gram-positive dataset:")
print(max_length_g)
print("\nMin length of proteins per class in gram-positive dataset:")
print(min_length_g) 

