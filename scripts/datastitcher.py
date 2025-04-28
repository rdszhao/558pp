# %%
import ast
from tqdm import tqdm
import numpy as np
import pandas as pd

processed_dir = '../data/processed/uscelection'
weeks_dir = '../data/processed/uscelection/weeks'
txt_dir = '../data/processed/uscelection/weektxt'

df = pd.read_csv(f"{processed_dir}/alltweets.tsv.gz", sep='\t', compression='gzip')
df['mentionedUsers'] = df['mentionedUsers'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

df['id'] = df['id'].astype(int, errors='ignore')
df['user'] = df['user'].astype(int, errors='ignore')

mentions = set()
for arr in df['mentionedUsers']:
	if isinstance(arr, list):
		for e in arr :
			mentions.add(int(e))

df = df[(df['mentionedUsers'].astype(bool)) | (df['user'].isin(mentions))]

cols = ['likeCount','retweetCount','replyCount','quoteCount','viewCount']
df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

explodf = df.explode('mentionedUsers')
aggdf = explodf.groupby(['week', 'user', 'mentionedUsers']).agg({
	'replyCount': 'sum',
	'retweetCount': 'sum',
	'quoteCount': 'sum',
	'likeCount': 'sum',
	'viewCount': 'sum',
}).reset_index()
aggdf['metric'] = \
	(aggdf['likeCount'] + aggdf['retweetCount'] + aggdf['replyCount'] + aggdf['quoteCount']) \
	* (aggdf['viewCount'] + 0.01).apply(np.log) + 1
aggdf = aggdf.rename(columns={'mentionedUsers': 'mentionedUser'})
aggdf['user'] = aggdf['user'].astype(str)
newdf = aggdf[['week', 'user', 'metric', 'mentionedUser']]

for week in tqdm(newdf['week'].unique()):
	weekdf = newdf[newdf['week'] == week]
	week_tdf = df[df['week'] == week][['user', 'text']]
	weekdf = weekdf.drop(['week'], axis=1)
	weekdf.to_csv(f"{weeks_dir}/week_{week}.csv.gz", index=False)
	week_tdf.to_csv(f"{txt_dir}/week_{week}.tsv.gz", sep='\t', index=False)