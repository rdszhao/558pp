# %%
import os
import pandas as pd
from tqdm import tqdm

partsdir = '../data/usc-x-24-us-election'
processed_dir = '../data/processed/uscelection'
parts = [folder for folder in os.listdir(partsdir) if 'part' in folder]
outdirs = [f"{processed_dir}/{part}/{fname.split('.')[0]}_processed.tsv.gz" for part in parts for fname in os.listdir(f"{partsdir}/{part}")]
# %%
dfs = []
file = outdirs[0]
df = pd.read_csv(file, compression='gzip', sep='\t')
df
# %%

dfs = []
for file in tqdm(outdirs):
	try:
		df = pd.read_csv(file, compression='gzip', sep='\t')
		dfs.append(df)
	except:
		print(f"{file} - error")
		continue

df = pd.concat(dfs)
# %%

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')
df['week'] = df['date'].dt.isocalendar().week
df['year'] = df['date'].dt.year
df['week'] = df['week'] + (df['year'] - df['year'].min()) * 52
df = df[df['year'] == 2024]
df = df.drop('year', axis=1)
df = df.dropna(subset=['viewCount'])

for week in tqdm(df['week'].unique()):
	df[df['week'] == week].to_csv(f"{processed_dir}/weeks/week_{week}.csv")
# %%
