# Image-Based Search Engine for E-commerce

## Project Overview

This project is an image-based product retrieval system for fashion e-commerce. Given a query image, it returns visually similar clothing items by converting each product image into a dense feature embedding with MobileNetV2 and searching those embeddings with FAISS.

The repository demonstrates an end-to-end machine learning workflow that is directly relevant to production search and recommendation systems: large-scale image preprocessing, transfer learning, vector indexing, nearest-neighbor retrieval, and retrieval evaluation.

For recruiters, the value of this project is that it shows practical ML application beyond classification. Instead of predicting a label, the system solves a business problem common in online retail: helping users discover similar products when text search is incomplete or when shoppers start from an image.

## Problem Statement

Traditional e-commerce search depends heavily on structured metadata and keyword matching. That approach breaks down when:

- Product titles are inconsistent or incomplete.
- Users want visually similar items rather than exact text matches.
- Style, silhouette, and visual attributes matter more than keywords.

This project addresses that gap with visual search. A shopper can start from a product image and retrieve similar items based on learned visual features, which is useful for product discovery, catalog exploration, and recommendation workflows.

## System Architecture

The system follows a straightforward retrieval architecture:

1. Load product metadata from `styles.csv` and match records to available images.
2. Preprocess each image and pass it through a pre-trained MobileNetV2 backbone.
3. Extract a 1280-dimensional embedding for every valid image.
4. Store embeddings and image paths in a serialized artifact.
5. Build a FAISS index for fast nearest-neighbor search.
6. Retrieve the top `K` closest products for a query image.
7. Evaluate retrieval quality using metadata-driven relevance labels.

### Current Scripts

- `feature_extractor.py`: builds embeddings and the FAISS index.
- `evaluate_model.py`: samples query images and computes retrieval metrics.
- `visual_search_example.py`: displays one random query and its top matches.

## Machine Learning Pipeline

### 1. Dataset Preparation

The project uses the Fashion Product Images (Small) dataset, which contains roughly 44,000 catalog images and structured metadata in `styles.csv`.

The preparation logic in `feature_extractor.py`:

- Reads product metadata with pandas.
- Converts product IDs to strings so they map cleanly to image filenames.
- Builds image paths in the `images/` directory.
- Filters out metadata rows whose corresponding image files are missing.

This is a useful data-engineering step because it ensures the downstream embedding pipeline only indexes images that actually exist.

### 2. Feature Extraction

Each image is resized to `224 x 224`, converted to a NumPy array, and normalized with MobileNetV2 preprocessing. Images are processed in batches for throughput.

The output of the CNN backbone is a compact embedding vector that captures visual information such as shape, texture, and color patterns. These embeddings become the representation used for retrieval.

### 3. Embedding Generation

The repository uses the output of MobileNetV2 with `include_top=False` and `pooling='avg'`. This removes the final ImageNet classifier and keeps the global feature representation, which is better suited for similarity search than closed-set classification logits.

### 4. Similarity Indexing

After embeddings are generated, the project builds a FAISS `IndexFlatL2` index. This supports exact nearest-neighbor search using Euclidean distance over the embedding vectors.

### 5. Retrieval

For a query image, the system retrieves the closest items in vector space and returns the top results after excluding the query item itself.

### 6. Evaluation

The evaluation script uses product `subCategory` as the relevance label. For each sampled query image, retrieved items are considered relevant when they share the same subcategory as the query.

This produces a clear, reproducible offline evaluation setup for comparing retrieval quality.

## Dataset

**Dataset:** Fashion Product Images (Small)

Typical repository inputs:

- `styles.csv`: product-level metadata.
- `images/`: catalog images named by product ID.

Why this dataset is a good fit:

- It is large enough to demonstrate batch inference and indexing at realistic scale.
- It includes structured metadata that can be reused for offline retrieval evaluation.
- It reflects a real e-commerce use case with diverse fashion categories.

### Optional EDA

An optional notebook is included at `notebooks/exploratory_data_analysis.ipynb` to inspect:

- dataset size and image coverage,
- missing values,
- category distributions,
- available metadata columns,
- top product segments by frequency.

## Model and Feature Extraction

### Why MobileNetV2

MobileNetV2 is a strong choice for this project because it offers a practical balance of accuracy, inference speed, and deployment efficiency. For portfolio projects, it is especially appropriate because it demonstrates good model selection judgment rather than brute-force model size.

Benefits in this use case:

- Pre-trained on ImageNet, so it transfers well to generic visual feature extraction.
- Lightweight enough for batch processing on modest hardware.
- Widely used and easy to explain in production-oriented ML discussions.

### Transfer Learning Approach

This project uses MobileNetV2 as a fixed feature extractor rather than fine-tuning it on the fashion dataset. That is a valid and pragmatic retrieval baseline because:

- it reduces training cost,
- it avoids the need for labeled retraining,
- it produces strong semantic image embeddings quickly.

### What the Embeddings Represent

The embedding vector is a numerical summary of each image. Similar products should map to nearby points in feature space, making nearest-neighbor retrieval possible.

## Similarity Search

FAISS is a library designed for efficient similarity search over dense vectors. In this project it is used to index all product embeddings and return the nearest matches for a query vector.

Why FAISS is used here:

- It is the industry-standard tool for vector retrieval workloads.
- It scales better than naive pairwise distance computation.
- It cleanly separates representation learning from retrieval infrastructure.

The current implementation uses `IndexFlatL2`, which is an exact search index. That is a sensible baseline because it prioritizes correctness and keeps the retrieval behavior easy to reason about before moving to approximate indexing strategies.

## Evaluation Metrics

The repository currently reports the following metrics from `evaluate_model.py` on 200 random queries:

- Precision@1: `95.00%`
- Precision@5: `91.00%`
- Recall@5: `0.26%`
- Mean Reciprocal Rank (MRR): `0.966`

### How to Interpret Them

**Precision@K** measures how many of the top `K` retrieved items are relevant.

- High Precision@1 means the first result is usually in the same subcategory as the query.
- High Precision@5 means the top results are consistently category-aligned.

**Recall@K** measures how much of the relevant set was recovered in the top `K` results.

- In this project, Recall@5 is low because each product subcategory can contain many relevant items, while only 5 results are retrieved.
- That makes low recall expected even when top-ranked results are good.

**Mean Reciprocal Rank (MRR)** measures how early the first relevant result appears.

- An MRR close to `1.0` means the system usually places a relevant result at or near rank 1.

### Practical Reading of the Results

These results suggest the system is strong at returning immediately useful near-neighbor results, which is often the most important behavior for user-facing visual search. They do not imply full catalog coverage of all relevant items, and the current evaluation should be treated as a solid baseline rather than a final benchmark.

## Project Structure

The repository is organized as follows:

```text
.
|-- evaluate_model.py
|-- feature_extractor.py
|-- notebooks/
|   |-- exploratory_data_analysis.ipynb
|-- visual_search_example.py
|-- requirements.txt
|-- README.md
```

What each directory is for:

- `notebooks/`: EDA, experiments, and analysis.
- `feature_extractor.py`: extracts image embeddings and builds the FAISS index.
- `evaluate_model.py`: evaluates retrieval quality with ranking metrics.
- `visual_search_example.py`: displays a sample query and retrieved results.
- `requirements.txt`: lists Python dependencies.
- `README.md`: documents the project, workflow, and usage.

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/asmiverma/Visual-Search-Model.git
cd Visual-Search-Model
```

### 2. Create a Virtual Environment

**Windows**

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

**macOS / Linux**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download and Place the Dataset

Download the Fashion Product Images (Small) dataset from Kaggle and place the files in the project root so the structure looks like this:

```text
.
|-- images/
|-- styles.csv
|-- feature_extractor.py
|-- evaluate_model.py
|-- visual_search_example.py
```

### 5. Build the Embeddings and Search Index

```bash
python feature_extractor.py
```

This creates:

- `image_features.pkl`
- `image_index.faiss`

### 6. Run Offline Evaluation

```bash
python evaluate_model.py
```

### 7. Visualize Example Retrieval Results

```bash
python visual_search_example.py
```

### 8. Optional: Open the EDA Notebook

```bash
jupyter notebook notebooks/exploratory_data_analysis.ipynb
```

## Example Results

The demo script displays one query image and the top retrieved matches side by side. A typical successful retrieval pattern looks like this:

```text
Query Image | Result 1 | Result 2 | Result 3 | Result 4 | Result 5
```

In practice, the strongest results tend to preserve:

- product subcategory,
- overall silhouette,
- dominant color palette,
- visual texture or print style.

For a portfolio version of this repository, the best next step is to save one or two representative retrieval screenshots from `visual_search_example.py` and include them directly in this section.

## Tech Stack

- Python
- TensorFlow / Keras
- MobileNetV2
- FAISS
- Pandas
- NumPy
- Pillow
- Matplotlib
- tqdm

## Future Improvements

- Fine-tune the backbone on fashion-specific similarity objectives.
- Replace exact search with approximate FAISS indices for larger catalogs.
- Add category-aware filtering and hybrid metadata + image retrieval.
- Package the pipeline as a small API or Streamlit app.
- Add experiment tracking and configuration management.
- Expand evaluation with stratified sampling and qualitative error analysis.

## Recruiter Signals

This project demonstrates:

- applied transfer learning for representation learning,
- vector search with FAISS,
- batch inference over a dataset of roughly 44,000 images,
- practical retrieval evaluation with ranking metrics,
- clear linkage between machine learning output and business value in e-commerce.
