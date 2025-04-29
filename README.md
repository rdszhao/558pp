## overview

This project implements an end-to-end pipeline and dashboard for analyzing weekly Twitter activity around the 2024 U.S. election, utilizing both network-based graph-embedding clusters and semantic topic models, then visualizing the results via a Streamlit dashboard. At a high level:

1. **data ingestion & preprocessing** (`dataloader.py`)  
2. **weekly stitching of metrics & text** (`datastitcher.py`)  
3. **graph embedding & community clustering** (`clustering.py`)  
4. **semantic topic modeling & alignment** (`topics.py`)  
5. **interactive visualization** (`viz.py`)

Each stage is a standalone Python script; together they produce:

- **`data/processed/uscelection/weeks/`**: per-week CSV of user metrics  
- **`data/processed/uscelection/weektxt/`**: per-week TSV of tweet texts  
- **`data/processed/clusters/`**: per-week 2d cluster outputs (embeddings + labels)  
- **`data/processed/topicmaps/`**: per-week topic-model JSON files as well as alignments with various news outlets 
- **streamlit dashboard** to explore clusters and topics over time as well as graph visuals

---

## project structure

```
.
├── dataloader.py       # load & preprocess raw tweets + news headlines
├── datastitcher.py     # aggregate and split data into weekly slices
├── clustering.py       # train graphsage for link-prediction, embed & k-means cluster
├── topics.py           # run bertopic per week, align to news sources
├── viz.py              # streamlit app: interactive exploration
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

## 1. data loading & preprocessing (`dataloader.py`)

This script processes two data sources:

- **raw twitter json dumps** (per‐month files)  
  - Uses regex extraction to pull fields: `id`, `user`, `text`, `viewCount`, `likeCount`, `retweetCount`, `replyCount`, `quoteCount`, `mentionedUsers`, `date`  
  - Normalizes counts to numeric types, concatenates into one DataFrame
  - Outputs compressed TSV:  
    ```bash
    data/processed/uscelection/alltweets.tsv.gz
    ```

- **news headlines csvs**  
  - Reads each `<source>_YYYYMMDD.csv`, appends `source` column  
  - Converts `date` → pandas `datetime`, extracts `week` index (ISO week + offset)  
  - Outputs combined CSV:  
    ```bash
    data/processed/uscelection/headlines.csv
    ```
---

## 2. Weekly Stitching (`datastitcher.py`)

Splits the processed data into per‐week files for downstream analysis:

1. **load** `alltweets.tsv.gz`  
2. **parse** `mentionedusers` lists with `ast.literal_eval`  
3. **aggregate** tweet metrics by user & week:  
   ```python
   metric = (likeCount + retweetCount + replyCount + quoteCount) * log(viewCount + ε) + 1
   ```
4. **explode + rename** `mentionedUsers` → `mentionedUser` (multiple to one)  
5. **write** per‐week CSVs:
   - **metrics** → `data/processed/uscelection/weeks/week_<week>.csv.gz`  
   - **raw texts** → `data/processed/uscelection/weektxt/week_<week>.tsv.gz`  
---

## 3. graph embedding & clustering (`clustering.py`)

training a **link‐prediction**-trained GraphSAGE model then clustering the graph node embeddings.

1. **data preparation**  
   - Read `week_<WEEK>.csv.gz`, map string IDs → integer node indices  
   - build `edge_index` and `edge_weight` tensors from user → mentionedUser with metric weights  
   - construct node features:  
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
   - **training**: binary cross‐entropy on dot‐product similarity
     - positive edges ↔ existing mentions  
     - negative edges via `torch_geometric.utils.negative_sampling`  
   - **optimizer**: `Adam(lr=1e-3)`, 200 epochs

3. **dimensionality reduction**  
   - **first UMAP** (8D) → embeddings for clustering  
   - **normalize** features  
   - **kmeans** (15 clusters) → cluster labels  
   - **second UMAP** (2D) → for visualization, compute per‐cluster mean/std, filter outliers

4. **outputs**  
   - cleaned 2d embedding csv → `data/processed/clusters/week_<week>.csv` with columns:
     ```
     user_id, x, y, label (cluster ID)
     ```

---

## 4. topic modeling & alignment (`topics.py`)

For each week’s raw tweets:

1. **load** `weektxt/week_<week>.tsv.gz`, clean text via `remove_urls`  
2. **embed** with `SentenceTransformer('all-MiniLM-L6-v2')`  
3. **BERTopic** pipeline:
   - custom **UMAP** (`n_components=5`) + **HDBSCAN** (`min_samples=10`)  
   - `KeyBERTInspired` representation model to extract keywords  
   - **topic labels** generated (10 words each, joined with `_`)  
4. **alignment** to news:
   - For each tweet embedding, find closest headline‐source embedding via cosine similarity (`closest_source`)  
   - Summarize alignments in a counter  

5. **save** per‐week json → `data/processed/topicmaps/week_<week>.json`  
   ```json
   {
     "<topic_id>": {
       "topics": ["keywords1", "keywords2", …],
       "alignments": {"cnn": 1234, "nyt": 8425, …}
     }, …
   }
   ```

---

## 5. visualization dashboard (`viz.py`)

a streamlit dashboard to explore both community clusters and topic maps:

- **sidebar**: week navigation (prev/next, dropdown)  
- **main view**:
  1. **Cluster Scatter**: Plotly scatter of nodes colored by cluster label  
  2. **Cluster Keywords**: multi-select clusters → display top keywords & cluster quality scores  
- **graph view**:
  - selected representation of the actual graph

Run locally:
```bash
streamlit run viz.py
```
---

## dependencies
- **core**: python 3.8+, pandas, numpy, scikit-learn, tqdm  
- **graph ml**: pytorch, torch-geometric, torch-scatter  
- **gpu accel.**: cuml, cupy  
- **nlp**: bertopic, sentence-transformers  
- **visualization**: streamlit, plotly  