# %%
import os
import re
import pandas as pd
from decimal import Decimal
from tqdm import tqdm

partsdir = '../data/usc-x-24-us-election'
processed_dir = '../data/processed/uscelection'
parts = [folder for folder in os.listdir(partsdir) if 'part' in folder]
prefiles = [f"{partsdir}/{part}/{fname}" for part in parts for fname in os.listdir(f"{partsdir}/{part}") if '.csv' in fname]
outdirs = [f"{processed_dir}/{part}/{fname.split('.')[0]}_processed.tsv.gz" for part in parts for fname in os.listdir(f"{partsdir}/{part}")]
filemap = dict(zip(outdirs, prefiles))

# if os.path.exists(processed_dir):
# 	alr_processed = [folder for folder in os.listdir(processed_dir) if 'part' in folder]
# 	processed_files = [f"{processed_dir}/{part}/{fname}" for part in alr_processed for fname in os.listdir(f"{processed_dir}/{part}")]
# 	for file in processed_files:
# 		filemap.pop(file, None)

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

attrs = ['id', 'user', 'text', 'viewCount', 'likeCount', 'retweetCount', 'replyCount', 'quoteCount', 'lang', 'mentionedUsers', 'in_reply_to_user_id_str', 'date']

dfs = []
bigmentions = []

for outdir, prefile in tqdm(filemap.items(), total=len(filemap)):
	try:
		df = pd.read_csv(prefile, compression='gzip', usecols=attrs)

		df = df[df['lang'] == 'en']
		df = df.drop(['lang'], axis=1)

		df['text'] = df['text'].astype(str).str.strip().replace(r'[^\w\s\-\.,;:!?()/\'"&]', '')
		df['user'] = df['user'].apply(lambda x: extract(x, 'id_str'))
		df['likeCount'] = df['likeCount'].astype(int, errors='ignore')
		df['viewCount'] = df['viewCount'].apply(lambda x: extract(x, 'count')).astype(int, errors='ignore')
		df['in_reply_to_user_id_str'] = df['in_reply_to_user_id_str'].apply(lambda x: '{:f}'.format(Decimal(str(x)).normalize()))
		df['date'] = pd.to_datetime(df['date'])
		df = df.sort_values('date')
		df['week'] = df['date'].dt.isocalendar().week
		df['year'] = df['date'].dt.year
		df['week'] = df['week'] + (df['year'] - df['year'].min()) * 52
		df = df[df['year'] == 2024]
		df = df.drop('year', axis=1)
		df = df.dropna(subset=['viewCount'])
		dfs.append(df)
	except:
		print(f"{prefile} - error")
		continue
# %%
df = pd.concat(dfs)
# %%
for week in tqdm(df['week'].unique()):
	weekdf = df[df['week'] == week]
	weekdf['mentionedUsers'] = weekdf['mentionedUsers'].apply(lambda x: extract(x, 'id_str', listed=True))
	mentions_dict = weekdf[['id', 'mentionedUsers']].set_index('id').to_dict()['mentionedUsers']
	weekdf = weekdf.drop(['mentionedUsers'], axis=1)
	mentions_edges = [
		(sid, 'mentionedUser', mention) if mentions else (sid, 'mentionedUser', None)
		for sid, mentions in mentions_dict.items()
		for mention in mentions or [None]
	]
	mentions_df = pd.DataFrame(mentions_edges, columns=['node1', 'label', 'node2'])
	melted = weekdf.melt(id_vars=['id'])
	melted.columns = ['node1', 'label', 'node2']
	processed_df = pd.concat([melted, mentions_df])
	processed_df.to_csv(f"{processed_dir}/weeks/week_{week}.tsv.gz", sep='\t', index=False)
# %%
