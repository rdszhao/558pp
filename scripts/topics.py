# %%
import os
import re
import json
import torch
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from cuml import UMAP
from cuml import HDBSCAN
from tqdm.notebook import tqdm

encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6'
device = torch.device('cuda')
txt_dir = 'data/weektxt'
embs_dir = 'data/clusters'
clustered = sorted([f"{embs_dir}/{filename}" for filename in os.listdir(embs_dir)])
txtfiles = sorted([f"{txt_dir}/{filename}" for filename in os.listdir(txt_dir)])
headline_df = pd.read_csv('data/headlines.csv')

def remove_urls(text: str) -> str:
    url_pattern = re.compile(r'https?://\S+|ftp://\S+|www\.\S+')
    return url_pattern.sub('', text)

def closest_source(emb, sources):
    sims = {
        src: cosine_similarity(emb.reshape(1,-1), s_emb.reshape(1,-1))[0,0]
        for src, s_emb in sources.items()
    }
    return max(sims, key=sims.get)

for emb_path, txt_path in tqdm(zip(clustered, txtfiles)):
	try:
		weeknum = int(emb_path.split('/')[-1].split('_')[1].split('.')[0])

		print(f"processing week {weeknum}...")
		txt_df = pd.read_csv(txt_path, sep='\t', compression='gzip')
		emb_df = pd.read_csv(emb_path).rename(columns={'user_id': 'user'})

		txt_df['user'] = txt_df['user'].apply(lambda x: format(x, '.0f'))
		txt_df['text'] = txt_df['text'].apply(remove_urls)
		emb_df['user'] = emb_df['user'].apply(lambda x: format(x, '.0f'))
		merged_df = pd.merge(emb_df, txt_df, on='user')

		week_df = headline_df[headline_df['week'] == weeknum].drop(columns=['week'])
		sources = {}
		for source in week_df['source'].unique():
			docs = week_df[week_df['source'] == source]['headline'].tolist()
			embs = encoder.encode(docs, show_progress_bar=False)
			sources[source] = embs.mean(axis=0)

		topicmap = {}
		for label in tqdm(merged_df['label'].unique()):
			docs = merged_df[merged_df['label'] == label]['text'].to_list()
			umap_model = UMAP(n_components=5, n_neighbors=15, min_dist=0.0)
			hdbscan_model = HDBSCAN(min_samples=10)
			topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, nr_topics=10)
			topics, probs = topic_model.fit_transform(docs)
			bert_embeddings = topic_model._extract_embeddings(docs)
			topic_labels = topic_model.generate_topic_labels(nr_words=10, topic_prefix=False, word_length=10, separator='_')
			topic_model.set_topic_labels(topic_labels)
			alignments = dict(Counter([closest_source(emb, sources) for emb in bert_embeddings]))
			topicmap[int(label)] = {
				'topics': topic_labels,
				'alignments': alignments,
			}

		with open(f"data/topicmaps/week_{weeknum}.json", 'w') as f:
			json.dump(topicmap, f, indent=4)
		print(f"week {weeknum} done!")
		os.remove(txt_path)
	except Exception as e:
		print(f"error processing {txt_path}: {e}")
		continue