# Image-Based Search Engine for E-commerce
This project implements a visual search engine for a fashion product dataset. It allows a user to find stylistically similar clothing items based on an input image. The core of the project is a deep learning model (MobileNetV2) used for feature extraction and a high-speed similarity search index (FAISS).

## Project Overview
The system works in two main stages:

#### 1. Indexing Pipeline (feature_extractor.py): 
A pre-trained MobileNetV2 model (trained on ImageNet) is used to convert each of the ~44,000 product images into a 1280-dimensional feature vector (embedding).

These embeddings are stored and then indexed using Facebook AI's Similarity Search (FAISS) library to enable ultra-fast lookups.

#### 2. Evaluation Pipeline (evaluate_model.py):

The system's performance is measured by taking a random sample of images and using them as queries. For each query, we find the top 5 most similar items and check if their product subcategories match the query's subcategory.

This is used to calculate standard information retrieval metrics.

#### 3. Performance Metrics
The model was evaluated on 200 random test queries, yielding the following performance:

##### Precision@1: 95.00%

##### Precision@5: 91.00%

##### Recall@5 (Sensitivity): 0.26%

Mean Reciprocal Rank (MRR): 0.966

(These are excellent results that demonstrate the high accuracy of the model, especially the high Precision and MRR.)

## How to Run This Project
#### 1. Setup
Clone the repository and set up the Python virtual environment:

    git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)

    cd YOUR_REPO_NAME

    python -m venv venv
#### Activate the environment
    
    Windows:
    .\venv\Scripts\activate

    macOS/Linux:
    source venv/bin/activate

    pip install -r requirements.txt

#### 2. Download Data
Download the "Fashion Product Images (Small)" dataset from Kaggle. Unzip it and place the images folder and styles.csv file in the root of the project directory.

#### 3. Run Feature Extraction
This will process all images and build the search index. It may take a few minutes.

    python feature_extractor.py

#### 4. Evaluate the Model
Run the evaluation script to see the performance metrics.

    python evaluate_model.py

#### 5. See a Visual Example
Run this script to see a pop-up window with a random query and its search results.

    python visual_search_example.py

## Tech Stack
#### Backend: Python
#### Deep Learning: TensorFlow, Keras

#### Similarity Search: Faiss (Facebook AI Similarity Search)

#### Data Handling: Pandas, NumPy, Pickle

#### Image Processing: Pillow

#### Visualization: Matplotlib