import pickle
import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from collections import defaultdict

# --- Configuration ---
FEATURES_FILE = 'image_features.pkl'
INDEX_FILE = 'image_index.faiss'
METADATA_PATH = 'styles.csv'
NUM_TEST_QUERIES = 200  # How many random images to test with
K = 5                   # For Precision@k, we will check top 5 results

# --- Load Data ---
print("Loading pre-computed data...")
try:
    with open(FEATURES_FILE, 'rb') as f:
        data = pickle.load(f)
    features = data['features']
    img_paths = data['paths']
    
    index = faiss.read_index(INDEX_FILE)
    
    df_meta = pd.read_csv(METADATA_PATH, on_bad_lines='skip')
    df_meta['id'] = df_meta['id'].astype(str)
    
    # Create a mapping from image path to its subCategory for quick lookups
    path_to_id = {path: os.path.basename(path).split('.')[0] for path in img_paths}
    id_to_subcategory = pd.Series(df_meta.subCategory.values, index=df_meta.id).to_dict()
    path_to_subcategory = {path: id_to_subcategory.get(path_to_id[path]) for path in img_paths}
    
    # --- NEW: Pre-calculate totals for Recall ---
    print("Pre-calculating category totals for recall metric...")
    subcategory_totals = defaultdict(int)
    for path in img_paths:
        subcat = path_to_subcategory.get(path)
        if subcat:
            subcategory_totals[subcat] += 1
            
except FileNotFoundError:
    print("Error: Data files not found. Please run feature_extractor.py first.")
    exit()
except Exception as e:
    print(f"An error occurred: {e}")
    exit()

print("Data loaded successfully.")

# --- Evaluation ---
print(f"Starting evaluation with {NUM_TEST_QUERIES} random test queries...")

# Select random indices for our query images
test_indices = np.random.choice(len(img_paths), NUM_TEST_QUERIES, replace=False)

precision_at_k_scores = []
precision_at_1_scores = []
reciprocal_ranks = []
recall_at_k_scores = [] # NEW list for recall scores

for i in tqdm(test_indices, desc="Evaluating Model"):
    query_path = img_paths[i]
    query_feature = features[i:i+1]
    
    true_subcategory = path_to_subcategory.get(query_path)
    
    if true_subcategory is None:
        continue
        
    distances, indices = index.search(query_feature, K + 1)
    result_indices = indices[0][1:]
    
    # --- Calculate Metrics for this Query ---
    correct_predictions_at_k = 0
    first_match_rank = 0
    
    for rank, result_idx in enumerate(result_indices):
        result_path = img_paths[result_idx]
        result_subcategory = path_to_subcategory.get(result_path)
        if result_subcategory == true_subcategory:
            correct_predictions_at_k += 1
            if first_match_rank == 0:
                first_match_rank = rank + 1
    
    # Precision@1
    first_result_path = img_paths[result_indices[0]]
    if path_to_subcategory.get(first_result_path) == true_subcategory:
        precision_at_1_scores.append(1.0)
    else:
        precision_at_1_scores.append(0.0)

    # Precision@K
    precision_at_k = correct_predictions_at_k / K
    precision_at_k_scores.append(precision_at_k)
    
    # Reciprocal Rank
    if first_match_rank > 0:
        reciprocal_ranks.append(1.0 / first_match_rank)
    else:
        reciprocal_ranks.append(0.0)

    # --- NEW: Calculate Recall@K ---
    total_relevant_items = subcategory_totals.get(true_subcategory, 0)
    if total_relevant_items > 0:
        recall_at_k = correct_predictions_at_k / total_relevant_items
        recall_at_k_scores.append(recall_at_k)

# --- Calculate and Display Final Averages ---
if precision_at_k_scores:
    mean_precision_at_1 = np.mean(precision_at_1_scores)
    mean_precision_at_k = np.mean(precision_at_k_scores)
    mean_reciprocal_rank = np.mean(reciprocal_ranks)
    mean_recall_at_k = np.mean(recall_at_k_scores) # NEW
    
    print("\n--- Evaluation Results ---")
    print(f"Tested with {len(precision_at_k_scores)} random queries.")
    print("-" * 50)
    print(f"Precision@1:       {mean_precision_at_1 * 100:.2f}%")
    print(f"Precision@{K}:       {mean_precision_at_k * 100:.2f}%")
    print(f"Recall@{K} (Sensitivity): {mean_recall_at_k * 100:.2f}%") # NEW
    print(f"Mean Reciprocal Rank (MRR): {mean_reciprocal_rank:.3f}")
    print("-" * 50)
else:
    print("Evaluation could not be completed. No valid queries were processed.")