# Synnapse-PS-1
## Problem Approach

### 1) Problem Statement
The objective of this project is **Vision-Based Inventory Item Re-Identification**: given a **query image** of an inventory item, the system must retrieve the **Top-K most visually similar items/images** from a database **purely based on visual similarity**, without relying on barcodes/metadata.  
To solve this, we build a **deep metric learning** pipeline that learns an embedding space where:
- Images of the **same item ID** are close together
- Images of **different item IDs** are far apart  
This embedding space is later used for **similarity scoring and retrieval** (open-set setting).
### 2) Dataset Loading (Stanford Online Products)
We load the Stanford Online Products (SOP) dataset and verify its directory structure to ensure:
- The dataset is downloaded correctly
- Expected annotation files and image folders exist
- Image paths referenced in annotations are valid on disk
### 3) Train/Test Construction Using `Ebay_train.txt` and `Ebay_test.txt`
The SOP dataset provides official splits via:
- `Ebay_train.txt`
- `Ebay_test.txt`
We parse these files (skipping the header) and construct two metadata tables (DataFrames) containing:
- `path`: relative image path
- `item_id`: inventory item identifier (instance label)
This produces:
- A training metadata table derived from `Ebay_train.txt`
- A testing metadata table derived from `Ebay_test.txt`
We also validate that:
- All image paths exist on disk
- The annotation-to-file mapping is consistent
### 4) EDA & Integrity Checks
- **Total number of images** in train and test
- **Number of unique item IDs** in train and test
- **Missing / invalid images** (paths that do not exist)
- **Train vs Test item overlap check**
#### Open-Set Retrieval Requirement (No Overlapping Item IDs)
Open-set retrieval requires that **item IDs in test are unseen during training**.  
So we explicitly verify that:
- `train_item_ids ∩ test_item_ids = ∅`
This ensures the model is evaluated on **new identities** and must generalize via learned visual similarity rather than memorization.
### 5) Label Mapping for Cross-Entropy Training
In addition to metric learning, we also use **Cross-Entropy Loss**, which requires labels to be:
- Integer encoded
- Contiguous in the range `[0, N-1]`
So we create a mapping:
- `item_id → label_index` where `label_index ∈ {0, 1, 2, ..., N-1}`
We also validate that:
- Labels are contiguous
- No missing indices exist
This enables a stable classification head during training.
### 6) PyTorch Dataset Construction
We convert the metadata tables into a PyTorch-compatible dataset.
Each sample returns:
- **Training mode (`is_train=True`)**
  - `(image_tensor, label_int)`
- **Test mode (`is_train=False`)**
  - `(image_tensor, item_id, image_path)`
This design supports:
- Metric + classification training on train set
- Retrieval-style evaluation on test set
### 7) Preprocessing & Data Augmentation (Transforms)
To satisfy the PS requirement of robustness to **viewpoint changes, lighting variations, cluttered backgrounds, and minor occlusions**, we use different transforms for train and test:
#### Training Transforms (Augmentation + Normalization)
- **RandomResizedCrop**
  - Encourages robustness to scale changes and partial views
  - Helps handle background clutter by forcing the model to focus on the object region
- **RandomHorizontalFlip**
  - Improves viewpoint invariance (left-right variations)
- **ColorJitter**
  - Improves robustness to lighting changes and color shifts
- **RandomErasing**
  - Simulates minor occlusions by randomly masking regions
  - Encourages learning from multiple discriminative parts
- **ToTensor + Normalization**
  - Converts images to tensors and normalizes using ImageNet statistics for stable training
#### Test Transforms (Deterministic)
- Resize / Center crop (deterministic)
- ToTensor + Normalization
No augmentation is applied at test time to ensure fair and consistent evaluation.
### 8) Training Objective: Metric Learning + Cross-Entropy (Hybrid Loss)
We train the model using a **joint objective**:
1. **Metric Learning Loss (Triplet Loss with Hard Mining)**
   - Directly optimizes the embedding space for retrieval
   - Ensures same-ID embeddings are closer than different-ID embeddings with a margin
2. **Classification Loss (Cross-Entropy)**
   - Applied using a classifier head on top of backbone features
   - Acts as a stabilizer/regularizer and improves discriminative learning
   
The overall optimization objective is:
`L_total = L_triplet + λ * L_CE`
Where:
- `L_triplet` enforces embedding separation
- `L_CE` is the classification loss
- `λ` balances the two objectives
### 9) Why PK Sampler is Required (PK Batch Sampling)
Metric learning requires **positives and negatives inside the same batch**.  
A random batch can contain many item IDs with only one sample, making triplet formation weak or impossible.
So we use a **PKBatchSampler**, which guarantees that each batch contains:
- **P unique item IDs**
- **K images per item ID**
Therefore, batch size = **P × K**
This ensures:
- **Multiple positives** per anchor (same item ID)
- **Many negatives** from other item IDs
- Efficient online hard mining within the batch
This batch construction significantly improves triplet learning quality and convergence for retrieval tasks.

In the next section, we describe the different model architectures explored (ResNet50, Swin Transformer, and DINOv2) and justify the final choice.

## Model Architectures
To determine the most effective backbone for Inventory Item Re-Identification, we experimented with three architectures:
1. **ResNet50 (CNN-based backbone)**
2. **Swin Transformer (Hierarchical Vision Transformer)**
3. **DINOv2 (Self-Supervised Vision Transformer)**
All three models follow a common high-level structure:
Backbone → Feature Extraction → 512-D Projection Head → L2 Normalization → (Classifier Head during training)
The embedding head and training pipeline remain consistent across models to ensure fair comparison.
### ResNet50 (CNN-Based Backbone)
#### Architecture Overview
ResNet50 is a deep Convolutional Neural Network using residual connections to enable stable training of deep architectures.
- Pretrained on ImageNet
- Final fully connected layer removed
- Global Average Pooling output used as feature vector
- Custom 512-dimensional embedding layer added
- L2 normalization applied to embeddings
- Additional classifier head for Cross-Entropy training
#### Dimensions
- Backbone output: 2048-d
- Projection layer: 2048 → 512
- Embedding dimension: 512
- Classifier output: `num_train_classes`
#### Fine-Tuning Strategy
- Early layers frozen initially
- Only top residual blocks and custom heads trained
- Differential learning rates:
  - Higher LR for embedding & classifier heads
  - Lower LR for backbone layers
- BatchNorm layers' running stats frozen for stability
#### Optimizer Setup
- AdamW optimizer
- Higher LR for heads (e.g., 3e-4)
- Lower LR for backbone (e.g., 3e-5)
#### Performance
ResNet50 provided strong baseline retrieval performance but struggled slightly with fine-grained visual similarity compared to transformer-based models.

### Swin Transformer (Hierarchical Vision Transformer)
#### Architecture Overview
Swin Transformer is a hierarchical Vision Transformer that uses shifted window attention for efficient local-global feature modeling.
- Pretrained ImageNet backbone via `timm`
- Default classification head removed
- Feature output projected to 512-d embedding
- L2 normalization applied
- Classifier head added for Cross-Entropy
#### Dimensions
- Backbone feature dimension: 768-d
- Projection layer → 512-d embedding
- Classifier → `num_train_classes`
#### Dynamic Unfreezing Strategy
Swin backbone is divided into stages. We:
1. Initially freeze entire backbone
2. Train only embedding + classifier heads
3. Gradually unfreeze last N backbone stages
4. Progressively increase trainable depth over epochs
Learning rate adjusted dynamically:
- Heads: higher LR
- Backbone stages: smaller LR
- LR reduced as more stages are unfrozen
This prevents catastrophic forgetting and stabilizes transformer fine-tuning.
#### Performance
Swin Transformer improved over ResNet in capturing global structure and fine-grained similarities but still did not outperform DINOv2 in open-set retrieval.

### DINOv2 (Self-Supervised Vision Transformer) — Final Model
#### Architecture Overview
DINOv2 is a powerful self-supervised Vision Transformer pretrained by Meta AI.
Key advantages:
- Learns strong semantic representations without labels
- Better generalization to unseen classes
- Excellent performance in retrieval tasks
Model structure:
- Pretrained DINOv2 ViT backbone
- Feature extractor output
- Linear projection to 512-d embedding
- L2 normalization
- Classifier head (training only)
#### Dimensions
- ViT feature output:384-d
- Projection head → 512-d embedding
- Classifier → `num_train_classes`
#### Progressive Unfreezing Strategy
Training is performed in stages:
1. Freeze entire backbone initially
2. Train embedding & classifier heads
3. Gradually unfreeze last transformer blocks
4. Increase number of trainable blocks across epochs
Benefits:
- Stable initial convergence
- Controlled adaptation to SOP dataset
- Prevents large destructive updates to pretrained weights
#### Optimizer Configuration
- AdamW optimizer
- Higher learning rate for projection & classifier heads
- Lower learning rate for backbone blocks
- Gradient clipping (`max_norm=1.0`) for stability
- Different weight decay for backbone vs heads
### Loss & Training Consistency Across Models
All three architectures use:
- **Batch-Hard Triplet Loss**
- **Cross-Entropy Loss**
- **PK Batch Sampler**
- L2-normalized 512-d embeddings
- Cosine similarity for evaluation
This ensures fair comparison between backbones.
### Performance Comparison
| Model        | Top-1 | Top-5 (Primary) | Top-10 |
|-------------|-------|-----------------|--------|
| ResNet50    | 67.28%   | 79.03%             | 82.64%    |
| Swin        | 74.10%   | 84.56%             | 87.47%    |
| DINOv2      | 75.04%   | 85.73%             | 88.63%    |

### Final Model Selection
DINOv2 achieved the highest Top-5 retrieval accuracy (primary metric), demonstrating:
- Stronger semantic feature representation
- Better open-set generalization
- Superior embedding separability
Therefore, DINOv2 was selected as the final backbone for Module B retrieval and API deployment.
### Checkpointing Strategy
To ensure best model preservation:
- Model evaluated after every epoch
- Top-5 accuracy monitored (primary metric)
- Best-performing weights saved as `best_model.pt`
- Additionally:
  - Epoch-level checkpoints saved
  - Only latest few checkpoints retained to save storage
This guarantees:
- Recovery from interruptions
- Preservation of peak-performing model
- Clean deployment-ready weights

The final DINOv2-based embedding model is used to generate gallery embeddings, build the FAISS index, and power the similarity-based retrieval pipeline.

We now describe the similarity metric used.
## Similarity Metric Used
After training, each image in the dataset is represented as a **512-dimensional L2-normalized embedding vector**.  
Similarity between images is computed directly in this learned feature space.
### Cosine Similarity
We use **Cosine Similarity** to measure how close two embeddings are.
Since all embeddings are L2-normalized, cosine similarity reduces to a simple dot product between vectors.
### Why Cosine Similarity?
Cosine similarity is chosen because:
- It is **scale-invariant** (independent of embedding magnitude)
- It works naturally with L2-normalized embeddings
- It aligns well with Triplet Loss training
- It is computationally efficient
- It is widely used in retrieval and metric learning systems
Because the embedding space is explicitly trained to cluster similar items together, cosine similarity directly reflects semantic similarity between inventory items.
### Retrieval
For a given query image:
1. Extract its 512-d normalized embedding.
2. Compute cosine similarity with all gallery embeddings.
3. Rank gallery items based on similarity score.
4. Return the Top-K most similar items.

A retrieval is considered correct if at least one image with the same `item_id` appears within the Top-K results.

We now describe the retrieval pipeline.
## Retrieval Pipeline (Open-Set Evaluation)
After training the embedding model, retrieval is performed entirely in the learned feature space. The evaluation strictly follows the open-set protocol defined in the problem statement.
### Step 1: Extract Embeddings for All Test Images
The model is first switched to evaluation mode:
- `model.eval()` ensures deterministic behavior (no dropout, stable normalization layers).
- `@torch.no_grad()` disables gradient computation for faster and memory-efficient inference.
Each test image is passed through the network to obtain a **512-dimensional L2-normalized embedding**.
At the end of this stage, we obtain:
- `all_emb` → tensor of shape `(N_test_images, 512)`
- `all_ids` → corresponding `item_id` for each embedding
This forms the complete embedding database for the test split.
### Step 2: Build Similarity Index (FAISS)
To enable efficient similarity search, we build a FAISS index:
- Embeddings are converted to `float32`
- L2 normalization is applied
- `faiss.IndexFlatIP` (Inner Product index) is used
Because embeddings are normalized, **inner product = cosine similarity**.
This allows fast and accurate nearest-neighbor retrieval.
### Step 3: Query–Gallery Construction (PS Style)
For each unique `item_id` in the test set:
1. If the item has fewer than 2 images, it is skipped (no positive available).
2. One image is randomly selected as the **query**.
3. All remaining test images (including other instances of the same item) form the **gallery**.
This simulates a real-world open-set retrieval setting:
- The model has never seen these item IDs during training.
- It must retrieve matching items purely based on learned visual similarity.
### Step 4: Similarity Search
For each query embedding:
1. Perform FAISS search to retrieve nearest neighbors.
2. Remove:
   - The query image itself
   - Any invalid indices
3. Obtain a ranked list of retrieved gallery item IDs.
### Step 5: Top-K Evaluation
For each query, we compute:
- Top-1 Accuracy
- Top-5 Accuracy (primary metric)
- Top-10 Accuracy
A retrieval is considered **correct** for a given K if:
> At least one image with the same `item_id` appears within the Top-K retrieved results.
This directly evaluates the quality of the learned embedding space.
### Step 6: Final Metrics
After iterating over all valid queries:

Top-K Accuracy = (Number of correct queries at K) / (Total number of queries)

The function returns:
- A dictionary containing Top-1, Top-5, and Top-10 accuracies
- The total number of evaluated queries

This pipeline directly measures how well the trained model structures the embedding space for similarity-based inventory re-identification.

We now discuss as to why this approach was used.
## Why This Approach Was Chosen
- The problem is a **retrieval task**, not closed-set classification, so we use **deep metric learning** to structure the embedding space instead of predicting fixed labels.
- A **hybrid loss (Triplet + Cross-Entropy)** is used:
  - Triplet Loss directly optimizes embedding distances for similarity.
  - Cross-Entropy stabilizes training and improves feature discrimination.
- **PK batch sampling** ensures meaningful positive and negative samples in every batch, enabling effective hard triplet mining.
- **Pretrained backbones with progressive unfreezing** preserve learned visual knowledge while adapting to the inventory dataset.
- **DINOv2** was selected as the final model because it provided the best open-set retrieval performance among the tested architectures.
- **Cosine similarity with FAISS** enables accurate and scalable nearest-neighbor retrieval in the learned embedding space.
## Complete Workflow For User (Detailed Step-by-Step)
This section explains the entire system end-to-end, from setting up the repository to running similarity-based retrieval.
### 1) Clone the Repository
```bash
git clone https://github.com/Prernatripathi7/Synnapse-PS-1.git
cd Synnapse-PS-1
```
This downloads the complete project including:
- Model reconstruction code (`src/build_model.py`)
- Module A – Feature Extraction
- Module B – Similarity & Retrieval
- Checkpoint download script
- Demo
- API code
### 2) Install Dependencies
```bash
pip install -r requirements.txt
```
This installs:
- **PyTorch** → model execution  
- **Torchvision** → transforms  
- **FAISS (CPU)** → fast similarity search  
- **Hugging Face Hub** → checkpoint download  
- **NumPy, Pillow** → embedding storage + image loading  
- **Matplotlib** → visualization  
- **FastAPI & Uvicorn** → API deployment  
### 3) Download the Trained Checkpoint (Hosted on Hugging Face)
```bash
python scripts/download_checkpoint.py
```
This script:
- Connects to the Hugging Face repository  
- Downloads `reid_model3_best.pth`  
- Saves it locally at:
```
models/reid_model3_best.pth
```
#### Why is the checkpoint hosted on Hugging Face?
The trained model weights are intentionally **not stored directly in the GitHub repository**. Instead, they are hosted on Hugging Face for the following reasons:
- GitHub repositories have size limitations for large binary files.
- Model checkpoints (.pth files) are typically hundreds of MBs.
- Hugging Face is optimized for hosting and versioning ML artifacts.
- It allows clean separation of **code (GitHub)** and **model weights (Hugging Face)**.
- This ensures reproducibility and keeps the repository lightweight.
At runtime, the checkpoint is dynamically downloaded and integrated into the system.
### 4) Rebuild the Model Architecture
The system reconstructs the inference architecture using:
`src/build_model.py`
Internally, this:
- Loads **DINOv2 (dinov2_vits14)** backbone using `torch.hub`
- Backbone feature dimension = **384**
- Adds projection layer:
  384 → 512 embedding dimension
- Prepares classifier head (used during training, safely ignored during retrieval)
- Returns inference-ready embedding model
This guarantees architectural consistency with the trained checkpoint.
### 5) Module A — Feature Extraction (Image → 512-d Embedding)
Implemented in:
`src/feature_extraction/encoder.py`
When executed, Module A:
1. Rebuilds the DINOv2-based model
2. Loads checkpoint weights
3. Cleans DataParallel prefixes (`"module."`)
4. Ignores classifier mismatch keys (not needed for retrieval)
5. Switches model to `eval()` mode
6. Runs inference under `torch.no_grad()`
7. Produces a **512-dimensional embedding vector**
8. Applies **L2 normalization**
Why L2 normalization?
Because cosine similarity = inner product when vectors are normalized.  
This enables efficient FAISS similarity search.
### 6) Build the Gallery Embedding Database (One-Time Step)
Before retrieval, gallery embeddings must be generated.
This is handled by:
`src/feature_extraction/build_gallery.py`
It:
- Iterates over all gallery images
- Extracts 512-d normalized embeddings
- Saves:
features/gallery_embeddings.npy
features/gallery_item_ids.npy
features/gallery_refs.npy
This creates a persistent embedding database for fast retrieval.
#### Example: Generating Gallery Embeddings
Run the following from the project root:
```python
import torch
from src.build_model import build_model
from src.feature_extraction.build_gallery import build_gallery_database
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def model_ctor():
    return build_model(
        variant="dinov2_vits14",
        emb_dim=512,
        num_classes=1,
        device=device
    )
gallery_emb, gallery_ids, gallery_refs = build_gallery_database(
    model_ctor=model_ctor,
    ckpt_path="models/reid_model3_best.pth",
    gallery_loader=test_loader,
    device=device,
    out_dir="features",
    normalize=True
)
```
### 7) Module B — Similarity Search (FAISS Retrieval)
Implemented in:
`src/similarity_scoring_and_retrieval/retriever.py`
When a query is provided:
1. Load gallery embeddings
2. Build FAISS index:
```python
faiss.IndexFlatIP(512)
```
3. Since embeddings are L2 normalized:
Inner Product = Cosine Similarity
4. Perform Top-K nearest neighbor search
5. Return:
   - Retrieved indices
   - Corresponding item IDs
   - Similarity scores
### 8) Run the Retrieval Demo (User-Level Execution)
This demo supports **any query image** (it does NOT need to belong to the dataset).  
The system will embed the query image using the trained DINOv2 model and retrieve the **Top-K most similar gallery images** using FAISS (cosine similarity via L2-normalized inner product).
#### Example Usage (Any Input Image Path)
```python
import torch
from src.build_model import build_model
from src.similarity_scoring_and_retrieval.demo import run_retrieval_demo_any_image
device = "cuda" if torch.cuda.is_available() else "cpu"
def model_ctor():
    return build_model(
        variant="dinov2_vits14",
        emb_dim=512,
        num_classes=1,
        device=device
    )
top_ids, scores, idxs = run_retrieval_demo_any_image(
    model_ctor=model_ctor,
    ckpt_path="models/reid_model3_best.pth",
    query_image_path="path/to/any/query.jpg",  # <-- user can give ANY image here
    k=5,
    device=device,
    emb_path="features/gallery_embeddings.npy",
    ids_path="features/gallery_item_ids.npy",
    refs_path="features/gallery_refs.npy",
    output_path="sample_output/sample_retrieval.png",
    extra=300
)
```
## Final System Pipeline
**User Flow:**
Clone → Install Dependencies → Download Checkpoint → Rebuild Model → Build Gallery → Run Retrieval
**System Flow:**
Image → DINOv2 Backbone → 512-d Embedding → L2 Normalize → FAISS Cosine Search → Top-K Results → Visualization
This modular design ensures:
- Clean architecture reconstruction
- Lightweight checkpoint management
- Efficient large-scale similarity retrieval
- Easy deployment and reproducibility

## Deployment
### Local Testing (Development)
The API can be tested locally at:
http://127.0.0.1:8000  (localhost)

This is only for development and debugging purposes.
### Precomputed Gallery Embeddings
The gallery contains ~60K images.  
Computing embeddings during API startup would be slow and memory-intensive.
Therefore:
- Gallery embeddings are **precomputed offline**
- Stored as:
  - `gallery_embeddings.npy` (N × 512)
  - `gallery_item_ids.npy`
  - `gallery_refs.npy`
- Uploaded to a **Hugging Face Model Repository**
- Not stored in GitHub (to keep repo lightweight)
### Assets Hosted on Hugging Face
The following files are hosted externally:
- `models/reid_model3_best.pth` → Trained DINOv2 checkpoint
- `features/gallery_embeddings.npy`
- `features/gallery_item_ids.npy`
- `features/gallery_refs.npy`
During deployment, these are automatically downloaded if not present locally.
###  Deployment Workflow
When the API receives its first `/search` request:
1. Downloads model + gallery assets from Hugging Face (if needed)
2. Builds the DINOv2 model (`dinov2_vits14`)
3. Loads the trained checkpoint
4. Loads gallery embeddings into FAISS (cosine similarity)
5. Becomes ready to serve requests
6. Lazy initialization is used to avoid startup timeouts on cloud platforms.
### Retrieval Flow
1. User uploads **any image**
2. Image is resized + normalized
3. 512-D embedding is generated
4. FAISS performs cosine similarity search
5. Top-K similar gallery images are returned
Each result includes:
- `item_id`
- `score` (cosine similarity)
- `ref` (gallery image path)
### Screenshots of API  request and response
#### Request
![request](https://github.com/user-attachments/assets/ad2f5697-6d8d-441f-9c92-e510ddcdad07)
#### Response
![response](https://github.com/user-attachments/assets/5ba38d92-754c-4bee-aa2b-7c0237d2596b)

##  Cloud Deployment & Infrastructure Analysis (Render)

As a final step to test our pipeline in a real-world scenario, we attempted to deploy the FastAPI backend to Render's free tier. Our goal was to see how the application behaves in a live, public-facing environment. 

While the deployment proved that our code and API routing were correct, it also highlighted the heavy hardware requirements of running deep learning models. 

### What Worked: Successful Build and Routing
The deployment process itself was a success. Render was able to install all our dependencies from `requirements.txt`, launch the Uvicorn server, and set the service live at our `.onrender.com` URL in a few minutes.

<img width="1600" height="598" alt="image" src="https://github.com/user-attachments/assets/249cb610-41e1-4245-b618-f559de0a2619" />


We verified the API routing through the server logs. The diagnostic endpoints, including `GET /docs`, `GET /openapi.json`, and `GET /health`, all successfully returned HTTP `200 OK` status codes. As expected, navigating to the root URL (`/`) or `/favicon.ico` returned a `404 Not Found`, since we intentionally only defined the `/health` and `/search` routes for this API.

### The Challenge: Hardware Limits and Memory Crashes (OOM)
The main issue occurred when the application actually tried to run the deep learning components. The service triggered an automatic restart because it exceeded its allocated memory limit.

Render's free tier limits instances to just 512 MB of RAM. Our backend is designed to "lazy load" the DINOv2 model and the FAISS database into memory when the first `/search` request hits. Loading these heavy PyTorch tensors and massive `.npy` files requires far more than 512 MB, which immediately caused an Out-Of-Memory (OOM) crash by the host server.

Additionally, the free instance is designed to spin down after 15 minutes of inactivity. 

Our logs showed the server repeatedly shutting down and then undergoing a "cold start." This spin-down behavior makes the memory issue even worse, as the server has to try and reload the massive model weights from scratch every time it wakes up.

### Conclusion
This deployment attempt was highly valuable. It proved that our codebase is production-ready and the API routing is fully functional. However, it also taught us that hosting a persistent vector database and a transformer model requires dedicated hardware. To make this API stable and always available, it would need to be hosted on a higher-tier instance with at least 4GB to 8GB of RAM to comfortably handle the PyTorch weights and FAISS index.

