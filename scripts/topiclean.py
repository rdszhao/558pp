# %%
import os
import json
import numpy as np
import pandas as pd
import nltk
from collections import Counter
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
# %%
nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()

def clean_representation(words) -> str:
    cleaned = []
    seen = set()
    for w in words:
        w_low = w.lower()
        if w_low in ENGLISH_STOP_WORDS or 'http' in w_low:
            continue
        lemma = lemmatizer.lemmatize(w_low)
        if lemma not in seen:
            seen.add(lemma)
            cleaned.append(lemma)
    return '_'.join(cleaned)
# %%
topicmaps_dir = '../data/processed/topicmaps'
topicfiles = ({int(filename.split('_')[-1].split('.')[0]): f"{topicmaps_dir}/{filename}" for filename in os.listdir(topicmaps_dir)})
topicfiles = dict(sorted(topicfiles.items(), key=lambda x: x[0]))
# %%
for weeknum, file in topicfiles.items():
    with open(file, 'r') as f:
        topics = json.load(f)
    for cluster in topics:
        topics[cluster]['clean_topics'] = list(set([clean_representation(tstring.split('_')) for tstring in topics[cluster]['topics']]))
    with open(file, 'w') as f:
        json.dump(topics, f, indent=4)
# %%
