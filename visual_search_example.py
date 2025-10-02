import pickle
import faiss
import numpy as np
import random
import os
from PIL import Image
import matplotlib.pyplot as plt

# --- Configuration ---
FEATURES_FILE = 'image_features.pkl'
INDEX_FILE = 'image_index.faiss'
K = 5 # Number of similar items to retrieve

# --- Load Pre-computed Data ---
print("Loading data...")
try:
    with open(FEATURES_FILE, 'rb') as f:
        data = pickle.load(f)
    features = data['features']
    img_paths = data['paths']
    index = faiss.read_index(INDEX_FILE)
except FileNotFoundError:
    print("Error: Data files not found. Please run feature_extractor.py first.")
    exit()

# --- Perform a Random Search ---
# 1. Pick a random image from our dataset
random_index = random.randint(0, len(img_paths) - 1)
query_path = img_paths[random_index]
query_feature = features[random_index:random_index+1]

print(f"Querying with image: {os.path.basename(query_path)}")

# 2. Perform the search
distances, indices = index.search(query_feature, K + 1)
result_indices = indices[0][1:] # Exclude the first result (which is the query itself)
result_paths = [img_paths[i] for i in result_indices]

# --- Display the Results ---
# 3. Use Matplotlib to show the images
plt.style.use('dark_background')
fig, axes = plt.subplots(1, K + 1, figsize=(10, 5))

# Display query image
query_image = Image.open(query_path)
axes[0].imshow(query_image)
axes[0].set_title("Query Image", fontsize=12)
axes[0].axis('off')

# Display result images
for i, result_path in enumerate(result_paths):
    result_image = Image.open(result_path)
    axes[i+1].imshow(result_image)
    axes[i+1].set_title(f"Result #{i+1}", fontsize=12)
    axes[i+1].axis('off')

plt.suptitle("Visual Search Results", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make room for suptitle
plt.show()