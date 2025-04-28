## Overview

This project implements an end-to-end pipeline for analyzing weekly Twitter activity around the U.S. election, deriving both network-based community clusters and semantic topic models, then visualizing the results via a Streamlit dashboard. At a high level:

1. **Data Ingestion & Preprocessing** (`dataloader.py`)  
2. **Weekly Stitching of Metrics & Text** (`datastitcher.py`)  
3. **Graph Embedding & Community Clustering** (`clustering.py`)  
4. **Semantic Topic Modeling & Alignment** (`topics.py`)  
5. **Interactive Visualization** (`viz.py`)

Each stage is a standalone Python script; together they produce:

- **`data/processed/uscelection/weeks/`**: per-week CSV of user metrics  
- **`data/processed/uscelection/weektxt/`**: per-week TSV of tweet texts  
- **`data/processed/clusters/`**: per-week community cluster outputs (embeddings + labels)  
- **`data/processed/topicmaps/`**: per-week topic-model JSON files  
- **Streamlit dashboard** to explore clusters and topics over time  

---

## Project Structure

```
.
├── dataloader.py       # Load & preprocess raw tweets + news headlines
├── datastitcher.py     # Aggregate and split data into weekly slices
├── clustering.py       # Train GraphSAGE for link-prediction, embed & k-means cluster
├── topics.py           # Run BERTopic per week, align to news sources
├── viz.py              # Streamlit app: interactive exploration
└── data/
    └── processed/
        ├── uscelection/
        │   ├── alltweets.tsv.gz
        │   ├── headlines.csv
        │   ├── weeks/
        │   └── weektxt/
        ├── clusters/
        └── topicmaps/
```

---

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-org/us-election-analysis.git
   cd us-election-analysis
   ```

2. **Create a Conda environment**  
   ```bash
   conda create -n uscelection python=3.8 -y
   conda activate uscelection
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   # requirements.txt should include:
   # pandas, numpy, scikit-learn, torch, torch-geometric, torch-scatter,
   # cuml, cupy, bertopic, sentence-transformers, streamlit, plotly, tqdm
   ```

4. **Configure GPU devices**  
   In `clustering.py` and `topics.py`, you can control which CUDA devices are used via:
   ```python
   os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'    # adjust as needed
   ```

---

## 1. Data Loading & Preprocessing (`dataloader.py`)

This script processes two data sources:

- **Raw Twitter JSON dumps** (per‐month files)  
  - Uses regex extraction to pull fields: `id`, `user`, `text`, `viewCount`, `likeCount`, `retweetCount`, `replyCount`, `quoteCount`, `lang`, `mentionedUsers`, `date`  
  - Normalizes counts to numeric types, concatenates into one DataFrame  
  - Outputs compressed TSV:  
    ```bash
    data/processed/uscelection/alltweets.tsv.gz
    ```

- **News Headlines CSVs**  
  - Reads each `<source>_YYYYMMDD.csv`, appends `source` column  
  - Converts `date` → pandas `datetime`, extracts `week` index (ISO week + offset)  
  - Outputs combined CSV:  
    ```bash
    data/processed/uscelection/headlines.csv
    ```

Key implementation notes:

- `extract(text, val, listed=False)` uses a compiled regex to robustly pull numeric fields from JSON strings.  
- Bulk concatenation via `pd.concat(dfs)` ensures efficient merging.

---

## 2. Weekly Stitching (`datastitcher.py`)

Splits the processed data into per‐week files for downstream analysis:

1. **Load** `alltweets.tsv.gz`  
2. **Parse** `mentionedUsers` lists with `ast.literal_eval`  
3. **Aggregate** tweet metrics by user & week:  
   ```python
   metric = (likeCount + retweetCount + replyCount + quoteCount) * log(viewCount + ε) + 1
   ```
4. **Rename** `mentionedUsers` → `mentionedUser` (count of mentions)  
5. **Write** per‐week CSVs:
   - **Metrics** → `data/processed/uscelection/weeks/week_<WEEK>.csv.gz`  
   - **Raw Texts** → `data/processed/uscelection/weektxt/week_<WEEK>.tsv.gz`  

By slicing at the week level, subsequent steps can parallelize weekly processing.

---

## 3. Graph Embedding & Clustering (`clustering.py`)

Implements a **link‐prediction**-trained GraphSAGE model, then clusters node embeddings.

1. **Data Preparation**  
   - Read `week_<WEEK>.csv.gz`, map string IDs → integer node indices  
   - Build `edge_index` and `edge_weight` tensors from user → mentionedUser with metric weights  
   - Construct node features:  
     ```python
     x = [1, in_degree(i), out_degree(i)]  for each node
     ```

2. **Weighted GraphSAGE**  
   ```python
   class WeightedSAGEConv(nn.Module):
       def forward(self, x, edge_index, edge_weight):
           # message: w_ij * x_j, aggregated by sum
           # normalization: divide by sqrt(deg_i * deg_j)
           # apply linear transform + activation
   class WeightedGraphSAGE(nn.Module):
       # stacks multiple WeightedSAGEConv layers
   ```
   - **Training**: binary cross‐entropy on dot‐product similarity
     - Positive edges ↔ existing mentions  
     - Negative edges via `torch_geometric.utils.negative_sampling`  
   - **Optimizer**: `Adam(lr=1e-3)`, 200 epochs

3. **Dimensionality Reduction**  
   - **First UMAP** (8D) → embeddings for clustering  
   - **Normalize** features  
   - **KMeans** (default 15 clusters) → cluster labels  
   - **Second UMAP** (2D) → for visualization, compute per‐cluster mean/std, filter outliers

4. **Outputs**  
   - Cleaned 2D embedding CSV → `data/processed/clusters/week_<WEEK>.csv` with columns:
     ```
     user_id, x, y, label (cluster ID)
     ```

---

## 4. Semantic Topic Modeling & Alignment (`topics.py`)

For each week’s raw tweets:

1. **Load** `weektxt/week_<WEEK>.tsv.gz`, clean text via `remove_urls`  
2. **Embed** with `SentenceTransformer('all-MiniLM-L6-v2')`  
3. **BERTopic** pipeline:
   - Custom **UMAP** (`n_components=5`) + **HDBSCAN** (`min_samples=10`)  
   - `KeyBERTInspired` representation model to extract keywords  
   - **Topic labels** generated (10 words each, joined with `_`)  
4. **Alignment** to news:
   - For each tweet embedding, find closest headline‐source embedding via cosine similarity (`closest_source`)  
   - Summarize alignments in a counter  

5. **Save** per‐week JSON → `data/processed/topicmaps/week_<WEEK>.json`  
   ```json
   {
     "<topic_id>": {
       "topics": ["keyword1", "keyword2", …],
       "alignments": {"cnn": 12, "nyt": 8, …}
     }, …
   }
   ```

---

## 5. Interactive Visualization (`viz.py`)

A Streamlit dashboard to explore both community clusters and topic maps:

- **Sidebar**: week navigation (prev/next, dropdown)  
- **Main view**:
  1. **Cluster Scatter**: Plotly scatter of nodes colored by cluster label  
  2. **Cluster Keywords**: multi-select clusters → display top keywords & cluster quality scores  
  3. **Topic Bar Charts**: show distribution of aligned news sources per topic  

Run locally:
```bash
streamlit run viz.py
```

---

## Deep Technical Details

- **Link-Prediction Training**  
  - Positive logits:  
    \[
      \text{pos\_dot} = z_i \cdot z_j,\quad (i,j)\in E
    \]
  - Negative sampling ratio = number of edges  
  - Loss:  
    \[
      \mathcal{L} = \mathrm{BCE}(\text{pos\_dot}, 1) + \mathrm{BCE}(\text{neg\_dot}, 0)
    \]

- **Weighted Aggregation**  
  \[
    m_i = \sum_{j\in \mathcal{N}(i)} w_{ij}\,x_j,\quad 
    \hat{x}_i = \sigma\big(W\, [x_i \,\|\, m_i]\big)
  \]
  Normalization by \(\sqrt{\deg(i)\,\deg(j)}\) ensures scale invariance.

- **Dimensionality Reduction & Clustering**  
  - **UMAP**: fast, GPU‐accelerated via cuML for high-dim embedding  
  - **KMeans**: cuML implementation for >10⁵ points  
  - **Outlier Filtering**: z-distance in 2D ≤3.0 to remove spurious embeddings

- **BERTopic Customization**  
  - Uses cuML UMAP & HDBSCAN for speed on large text corpora  
  - `generate_topic_labels(nr_words=10, separator='_' )` builds consistent labels  
  - Alignment of tweet topics ↔ news sources via embedding nearest-neighbor counts

---

## Usage Workflow

1. **Preprocess Data**  
   ```bash
   python dataloader.py
   ```
2. **Split Weekly Data**  
   ```bash
   python datastitcher.py
   ```
3. **Compute Graph Embeddings & Clusters**  
   ```bash
   python clustering.py
   ```
4. **Run Topic Modeling**  
   ```bash
   python topics.py
   ```
5. **Launch Dashboard**  
   ```bash
   streamlit run viz.py
   ```

---

## Dependencies

- **Core**: Python 3.8+, pandas, numpy, scikit-learn, tqdm  
- **Graph ML**: PyTorch, torch-geometric, torch-scatter  
- **GPU Accel.**: cuML, CuPy  
- **NLP**: bertopic, sentence-transformers  
- **Visualization**: streamlit, plotly  

Ensure your CUDA drivers and CUDA-enabled libraries are installed for GPU acceleration.

---

## Acknowledgments

- **BERTopic** by Maarten Grootendorst  
- **PyTorch Geometric** by Rusty Rossmann et al.  
- **cuML** from RAPIDS AI  

---

Feel free to open issues for bugs or feature requests!
