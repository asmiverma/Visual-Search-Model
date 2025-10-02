import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
import numpy as np
import pandas as pd
from PIL import Image
import faiss
import pickle
from tqdm import tqdm
import os

# --- Section 1: Configuration ---
# Define key parameters for our project.
IMG_SIZE = 224
BATCH_SIZE = 64 # Process 64 images at a time for efficiency
DATASET_PATH = 'images'
METADATA_PATH = 'styles.csv'
FEATURES_FILE = 'image_features.pkl' # File to save extracted features
INDEX_FILE = 'image_index.faiss'   # File to save the search index

# --- Section 2: Load the Pre-trained Model (The "AI Brain") ---
print("Loading pre-trained MobileNetV2 model...")
# We use MobileNetV2, a model trained on millions of images (ImageNet dataset).
# It's already an expert at understanding features in images.
# `include_top=False`: We don't need the final classification layer that says "cat" or "dog".
# We want the layer just before that, which is a rich numerical summary of the image.
# `pooling='avg'`: This adds a layer that averages all the features into a single vector.
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling='avg')
model = tf.keras.Model(inputs=base_model.input, outputs=base_model.output)
print("Model loaded successfully.")

# --- Section 3: Prepare the Image List ---
# We need a clean list of all image files we want to process.
try:
    df = pd.read_csv(METADATA_PATH, on_bad_lines='skip')
    # Ensure product IDs are strings to match filenames (e.g., '15970' matches '15970.jpg')
    df['id'] = df['id'].astype(str)
    all_image_ids = df['id'].tolist()
    
    # Create the full path for each image
    all_image_paths = [os.path.join(DATASET_PATH, f"{img_id}.jpg") for img_id in all_image_ids]
    
    # Data Cleaning: Keep only the paths for images that actually exist in our folder.
    image_paths = [path for path in all_image_paths if os.path.exists(path)]
    print(f"Found {len(image_paths)} existing images from the metadata file.")

except FileNotFoundError:
    print(f"Error: Metadata file not found at {METADATA_PATH}")
    exit()

if not image_paths:
    print("No images found. Please check your folder structure. Exiting.")
    exit()

# --- Section 4: The Feature Extraction Function ---
def extract_features_batch(img_paths, model):
    """Takes a list of image paths and returns their feature vectors."""
    batch_images = []
    valid_paths_in_batch = []
    for img_path in img_paths:
        try:
            # Step 1: Load image and resize it to the required 224x224 pixels.
            img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
            # Step 2: Convert image to a NumPy array (a grid of numbers).
            img_array = image.img_to_array(img)
            batch_images.append(img_array)
            valid_paths_in_batch.append(img_path)
        except Exception as e:
            # This is a robustness step. If an image is corrupted, we skip it.
            print(f"Warning: Could not process image {img_path}. Skipping.")
            
    # If the batch is empty after skipping corrupted images, return nothing.
    if not batch_images:
        return None, []

    # Step 3: Stack all individual image arrays into a single "batch" array.
    batch_images = np.stack(batch_images, axis=0)
    
    # Step 4: Pre-process the batch. This normalizes pixel values (e.g., from 0-255 to -1 to 1).
    # This is critical, as the model expects images in this normalized format.
    preprocessed_batch = preprocess_input(batch_images)
    
    # Step 5: The core step. Feed the batch to the model and get the feature vectors.
    features = model.predict(preprocessed_batch, verbose=0)
    return features, valid_paths_in_batch

# --- Section 5: Main Loop to Process All Images ---
print("Starting feature extraction...")
all_features = []
valid_image_paths = []

# Process images in batches for massive speed improvement.
# tqdm creates a smart progress bar.
for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="Extracting features"):
    batch_paths = image_paths[i:i+BATCH_SIZE]
    # Call our function to get features for the current batch.
    batch_features, successful_paths = extract_features_batch(batch_paths, model)
    
    if batch_features is not None:
        all_features.append(batch_features)
        valid_image_paths.extend(successful_paths)

# --- Section 6: Saving the Features and Building the Index ---
if all_features:
    # Combine the list of batch features into one single, large NumPy array.
    features_array = np.vstack(all_features)
    
    # Save the features and their corresponding paths together in one file using pickle.
    # This is crucial so we can later map a search result back to its filename.
    print(f"\nSaving features and paths for {len(valid_image_paths)} images...")
    with open(FEATURES_FILE, 'wb') as f:
        pickle.dump({'features': features_array, 'paths': valid_image_paths}, f)
    print("Features saved.")

    # Now, build the search index using FAISS.
    print("Building FAISS index for fast searching...")
    # Get the dimensionality of our vectors (for MobileNetV2, this is 1280).
    d = features_array.shape[1]
    
    # We use 'IndexFlatL2', which performs an exact, brute-force search.
    # It's extremely fast for millions of vectors and guarantees accuracy.
    index = faiss.IndexFlatL2(d)
    
    # Add all our feature vectors to the index.
    index.add(features_array)
    
    # Save the ready-to-use index to a file.
    print(f"Saving FAISS index to {INDEX_FILE}...")
    faiss.write_index(index, INDEX_FILE)
    print("FAISS index saved.")
else:
    print("No features were extracted. Something went wrong.")

print("\n--- Feature Extraction and Indexing Complete ---")