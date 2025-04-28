# %%
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling, degree
from torch_scatter import scatter
import cupy as cp
from cuml import UMAP
from cuml import PCA
from cuml.cluster import KMeans
from tqdm import trange, tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6'
device = torch.device('cuda')
weeks_dir = 'data/weeks'
embs_dir = 'data/clusters'
files = sorted([f"{weeks_dir}/{filename}" for filename in os.listdir(weeks_dir)])

class WeightedSAGEConv(torch.nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.lin = torch.nn.Linear(2*in_c, out_c)

    def forward(self, x, edge_index, edge_weight):
        src, dst = edge_index
        m = x[dst] * edge_weight.view(-1,1)
        agg = scatter(m, src, dim=0, dim_size=x.size(0), reduce='mean')
        return self.lin(torch.cat([x, agg], dim=1))

class WeightedGraphSAGE(torch.nn.Module):
    def __init__(self, in_c, hidden_c, num_layers=2):
        super().__init__()
        channels = [in_c] + [hidden_c]*num_layers
        self.convs = torch.nn.ModuleList([
            WeightedSAGEConv(channels[i], channels[i+1])
            for i in range(num_layers)
        ])

    def forward(self, x, edge_index, edge_weight):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_weight))
        return x

for file in tqdm(files):
	try:
		print(f"processing {file}...")
		fname = file.split('/')[-1].split('.')[0]
		df = pd.read_csv(file)
		df['user'] = df['user'].apply(lambda x: format(x, '.0f'))
		df['mentionedUser'] = df['mentionedUser'].apply(lambda x: format(x, '.0f'))
		nodes    = pd.unique(df[['user', 'mentionedUser']].values.ravel())
		id_map   = {nid: i for i, nid in enumerate(nodes)}
		num_nodes = len(nodes)

		src = torch.tensor(df['user'].map(id_map).values, dtype=torch.long)
		dst = torch.tensor(df['mentionedUser'].map(id_map).values, dtype=torch.long)
		ew  = torch.tensor(df['metric'].values, dtype=torch.float)
		edge_index  = torch.stack([src, dst], dim=0)

		in_deg  = degree(dst, num_nodes=num_nodes).unsqueeze(1)
		out_deg = degree(src, num_nodes=num_nodes).unsqueeze(1)
		x = torch.cat([torch.ones(num_nodes,1), in_deg, out_deg], dim=1)
		data = Data(x=x, edge_index=edge_index, edge_weight=ew)
		data   = data.to(device)

		model = WeightedGraphSAGE(in_c=x.size(1), hidden_c=32, num_layers=2).to(device)
		optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

		for epoch in trange(1, 201):
			model.train()
			optimizer.zero_grad()

			z = model(data.x, data.edge_index, data.edge_weight)
			pos_dot = (z[data.edge_index[0]] * z[data.edge_index[1]]).sum(dim=1)
			neg_ei  = negative_sampling(
				data.edge_index, num_nodes=data.num_nodes,
				num_neg_samples=data.edge_index.size(1)
			)
			neg_dot = (z[neg_ei[0]] * z[neg_ei[1]]).sum(dim=1)

			loss = (
				F.binary_cross_entropy_with_logits(pos_dot, torch.ones_like(pos_dot)) +
				F.binary_cross_entropy_with_logits(neg_dot, torch.zeros_like(neg_dot))
			)
			loss.backward()
			optimizer.step()

		embeddings = z.detach().cpu()
		emb = embeddings.numpy()
		emb_gpu = cp.asarray(emb)
		
		umap1 = UMAP(
			n_components=8,
			n_neighbors=10,
			min_dist=0.5,
			metric='euclidean'
		)
		emb1 = umap1.fit_transform(emb_gpu)
		emb_norm = emb1 / np.linalg.norm(emb1, axis=1, keepdims=True)

		n_clusters = 15
		clusterer = KMeans(
			n_clusters=n_clusters,
			n_init='auto',
		)
		labels = clusterer.fit_predict(emb_norm) 

		umap2 = UMAP(
			n_components=2,
			n_neighbors=30,
			min_dist=0.1,
			metric='euclidean'
		)
		emb2_gpu = umap2.fit_transform(emb1)

		emb2 = emb2_gpu.get()
		labels = labels.get()
		inv_id_map = {v: k for k, v in id_map.items()}
		users_in_order = [inv_id_map[i] for i in range(len(inv_id_map))]
		df_umap = pd.DataFrame({
			'user_id': users_in_order,
			'x': emb2[:, 0],
			'y': emb2[:, 1],
			'label': labels
		})

		cluster_stats = (
			df_umap
			.groupby('label')[['x','y']]
			.agg(['mean','std'])
		)
		cluster_stats.columns = ['x_mean','x_std','y_mean','y_std']
		df2 = df_umap.join(cluster_stats, on='label')
		df2['z_x'] = (df2['x'] - df2['x_mean']) / df2['x_std']
		df2['z_y'] = (df2['y'] - df2['y_mean']) / df2['y_std']
		df2['z_dist'] = np.sqrt(df2['z_x']**2 + df2['z_y']**2)
		threshold = 3.0
		df_clean = df2[df2['z_dist'] <= threshold]
		df_clean = df_clean[['user_id', 'x', 'y', 'label']].to_csv(f"{embs_dir}/{fname}.csv", index=False)
		df_clean = pd.read_csv(f"{embs_dir}/{fname}.csv")
		os.remove(file)
		print(f"done processing {file}!")
	except Exception as e:
		print(f"error processing {file}: {e}")
		continue