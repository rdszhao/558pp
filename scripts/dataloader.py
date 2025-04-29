# %%
import os
import re
import json
import pandas as pd
from tqdm import tqdm

partsdir = '../data/usc-x-24-us-election'
processed_dir = '../data/processed/uscelection'
newsdir = '../data/thenews'
parts = [folder for folder in os.listdir(partsdir) if 'part' in folder]
prefiles = [f"{partsdir}/{part}/{fname}" for part in parts for fname in os.listdir(f"{partsdir}/{part}") if '.csv' in fname]
outdirs = [f"{processed_dir}/{part}/{fname.split('.')[0]}_processed.tsv.gz" for part in parts for fname in os.listdir(f"{partsdir}/{part}")]
filemap = dict(zip(outdirs, prefiles))
# %%
def extract(text, val, listed=False):
	try:
		pattern = re.compile(rf"'{val}': '(\d+)'")
		matches = pattern.findall(text)
		if len(matches) > 1 or listed:
			return list(set(matches))
		else:
			return matches[0]
	except:
		return None

attrs = ['id', 'user', 'text', 'viewCount', 'likeCount', 'retweetCount', 'replyCount', 'quoteCount', 'lang', 'mentionedUsers', 'date', 'in_reply_to_user_id_str']
dfs = []

for outdir, prefile in tqdm(filemap.items(), total=len(filemap)):
	try:
		df = pd.read_csv(prefile, compression='gzip', usecols=attrs)
		df = df[df['lang'] == 'en']
		df = df.drop(['lang'], axis=1)
		df['id'] = df['id'].astype(str)
		df['user'] = df['user'].apply(lambda x: extract(x, 'id_str'))
		df['likeCount'] = df['likeCount'].astype(int, errors='ignore')
		df['viewCount'] = df['viewCount'].apply(lambda x: extract(x, 'count')).astype(int, errors='ignore')
		df['date'] = pd.to_datetime(df['date'])
		df['year'] = df['date'].dt.year
		df['week'] = df['date'].dt.isocalendar().week
		df['week'] = df['week'] + (df['year'] - df['year'].min()) * 52
		df = df.drop('year', axis=1)
		df = df.dropna(subset=['viewCount'])
		df['mentionedUsers'] = df['mentionedUsers'].apply(lambda x: extract(x, 'id_str', listed=True))
		df['text'] = df['text'].astype(str).str.strip().replace(r'[^\w\s\t\-\.,;:!?()/\'"&]', '')
		dfs.append(df)
	except:
		print(f"{prefile} - error")
		continue

df = pd.concat(dfs)
df.to_csv(f"{processed_dir}/alltweets.tsv.gz", sep='\t', index=False)
# %%

files = [os.path.join(newsdir, filename) for filename in os.listdir(newsdir) if '.csv' in filename]
dfs = []
for file in files:
	site = file.split('/')[-1].split('_')[0]
	df = pd.read_csv(file)
	df['source'] = site
	df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
	df['year'] = df['date'].dt.year
	df['week'] = df['date'].dt.isocalendar().week
	df['week'] = df['week'] + (df['year'] - df['year'].min()) * 52
	df = df.drop(['year', 'date'], axis=1)
	dfs.append(df)

df = pd.concat(dfs)
df.to_csv('../data/processed/headlines.csv', index=False)