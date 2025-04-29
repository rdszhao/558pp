# %%
import os
import json
import numpy as np
import pandas as pd
import datetime
import networkx as nx
from pyvis.network import Network
import streamlit as st
import tempfile
from collections import Counter
import plotly.express as px

clusters_dir = './data/processed/clusters'
topicmaps_dir = './data/processed/topicmaps'

topicfiles = ({int(filename.split('_')[-1].split('.')[0]): f"{topicmaps_dir}/{filename}" for filename in os.listdir(topicmaps_dir)})
topicfiles = dict(sorted(topicfiles.items(), key=lambda x: x[0]))
clusterfiles = ({int(filename.split('_')[-1].split('.')[0]): f"{clusters_dir}/{filename}" for filename in os.listdir(clusters_dir)})
clusterfiles = dict(sorted(clusterfiles.items(), key=lambda x: x[0]))

outlet_weights = {
    'democracynow' : -4.0,
    'thenation' : -5.0,
    'nyt' : -2.2,
    'cnn' : -1.3,
    'fox' : 3.9,
    'nypost' : 2.9,
    'dailywire' : 5.0,
    'breitbart' : 6.0
}
alpha = 0.8
def squash(z):
    return np.tanh(alpha * z)

def get_score(probs, weights=outlet_weights):
    z = {o: (probs[o] - source_stats[o][0]) / source_stats[o][1] for o in probs}
    s = {o: squash(z[o]) for o in probs}
    raw = sum(weights[o] * s[o] for o in probs)
    score = raw / sum(abs(weights[o]) for o in probs)
    return score

def compute_scores(weights=outlet_weights):
    newdict = weekdict
    for weeknum in weekdict:
        for cluster in weekdict[weeknum]:
            newdict[weeknum][cluster]['score'] = get_score(weekdict[weeknum][cluster]['probs'], weights=weights)
    return newdict

weekdict = {}
for weeknum, file in topicfiles.items():
    with open(file, 'r') as f:
        topics = json.load(f)
    for cluster, entry in topics.items():
        topics[cluster]['alignments'] = Counter(entry['alignments'])
        topics[cluster]['probs'] = dict(zip(entry['alignments'].keys(), np.array(list(entry['alignments'].values())) / entry['alignments'].total()))
    weekdict[weeknum] = topics

allprobs = [entries['probs'] for data in weekdict.values() for entries in data.values()]
probs_by_source = {outlet: [d.get(outlet, 0) for d in allprobs] for outlet in allprobs[0]}
source_stats = {source: (np.mean(probs), np.std(probs)) for source, probs in probs_by_source.items()}

for weeknum in weekdict:
    for cluster in weekdict[weeknum]:
        weekdict[weeknum][cluster]['score'] = get_score(weekdict[weeknum][cluster]['probs'])

scoredict = weekdict

# streamlit shit
st.set_page_config(layout='wide')
st.title('Twitter Clusters Over Time')

tab1, tab2 = st.tabs(['Knowledge Graph', 'Clusters Visualization'])

with tab1:
    if 'graph' not in st.session_state:
        #tweet_files = ['may_july_chunk_1_processed.tsv', 'aug_chunk_41_processed.tsv', 'september_chunk_41_processed.tsv', 'november_chunk_43_processed.tsv']
        
        tweets_df = pd.read_csv(f'./data/processed/kg_viz_data/aug_chunk_41_processed.tsv', delimiter='\t')
        metrics_df = pd.read_csv('./data/processed/kg_viz_data/alltweets.tsv', delimiter='\t')

        texts = tweets_df[tweets_df['label'] == 'text']
        replies = tweets_df[tweets_df['label'] == 'in_reply_to_user_id_str']
        replies['replied_to_user'] = replies['node2']

        merge = pd.merge(texts, replies[['node1','replied_to_user']], on='node1')
        merge = pd.merge(merge, metrics_df, left_on='node1', right_on='id')
        tweet_sample = merge.sample(500, random_state=1738)
        
        graph = nx.DiGraph()

        for idx, row in tweet_sample.iterrows():
            tweet_id = int(row['node1'])
            text = row['node2']
            replies = row['replyCount']
            retweets = row['retweetCount']
            likes = row['likeCount']
            views = row['viewCount']
            quotes = row['quoteCount']
            date = row['date']
            user_id = int(row['user'])

            tweet_title = f'Tweet: {text[:100]}\nLikes: {likes}\nRetweets: {retweets}\nViews: {views}\nReplies: {replies}'

            graph.add_node(user_id, label=f'User:{str(user_id)[-4:]}', color='green')
            graph.add_node(tweet_id, label=f'Tweet:{str(tweet_id)[-4:]}', title=tweet_title, color='skyblue')
            graph.add_edge(user_id, tweet_id, label='tweeted', color='skyblue')
            if row['mentionedUsers'] != '[]':
                if pd.notna(row['mentionedUsers']):
                    for mentioned_user in row['mentionedUsers'].strip("[]").split(', '):
                        mentioned_user = str(mentioned_user)
                        graph.add_node(mentioned_user, label=f'User:{str(mentioned_user)[-4:]}', color='green')
                        graph.add_edge(tweet_id, mentioned_user, label='mentioned', color='orange')
            if pd.notna(row['replied_to_user']):
                user_replied_to_id = str(row['replied_to_user'])
                graph.add_node(user_replied_to_id, label=f'User:{user_replied_to_id[-4:]}', color='green')
                graph.add_edge(tweet_id, user_replied_to_id, label='replied_to', color= 'purple')
                
        net = Network(directed=True)
        net.from_nx(graph)
        
        path = '/tmp/knowledge_graph.html'
        net.save_graph(path)
        with open(path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=800, scrolling=True)
        
with tab2:
    weeks = list(topicfiles.keys())
    if 'week' not in st.session_state:
        st.session_state.week = weeks[0]

    def go_prev():
        idx = weeks.index(st.session_state.week)
        if idx > 0:
            st.session_state.week = weeks[idx - 1]

    def go_next():
        idx = weeks.index(st.session_state.week)
        if idx < len(weeks) - 1:
            st.session_state.week = weeks[idx + 1]

    st.sidebar.title('Navigate Weeks')
    c1, c2 = st.sidebar.columns(2)
    with c1: st.button('← prev', on_click=go_prev)
    with c2: st.button('next →', on_click=go_next)
    selected = st.session_state.week

    for outlet, default in outlet_weights.items():
        key = f"weight_{outlet}"
        if key not in st.session_state:
            st.session_state[key] = default

    if st.sidebar.button("Reset weights to default"):
        for outlet, default in outlet_weights.items():
           st.session_state[f"weight_{outlet}"] = default

    st.sidebar.markdown("### Adjust Outlet Weights")
    new_weights = outlet_weights
    for outlet, default in new_weights.items():
        new_weights[outlet] = st.sidebar.slider(
            label=f"{outlet}",
            min_value=-10.0, max_value=10.0,
            value=float(default),
            step=0.1
        )
    scoredict = compute_scores(new_weights)

    year = 2024
    monday = datetime.date.fromisocalendar(year, selected, 1)
    sunday = datetime.date.fromisocalendar(year, selected, 7)
    st.sidebar.markdown(f"**week:** {selected}")

    def load_week_data(week):
        df = pd.read_csv(clusterfiles[week])
        mapping_score = {int(cl): data['score'] for cl, data in weekdict[week].items()}
        mapping_topics = {int(cl): data['topics'] for cl, data in weekdict[week].items()}
        df['score'] = df['label'].map(mapping_score)
        df['topics'] = df['label'].map(mapping_topics)
        return df

    df_week = load_week_data(selected)

    fig = px.scatter(
        df_week,
        x='x', y='y',
        color='score',
        color_continuous_scale='Portland',
        range_color=[-1,1],
        hover_data={'label':True, 'score':True},
        title=f"week of {monday:%b %d} — {sunday:%b %d, %y}"
    )
    fig.update_layout()
    fig.update_layout(
        xaxis_title='x',
        yaxis_title='y',
        height=800,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

    cluster_ids = sorted(weekdict[selected].keys(), key=lambda x: int(x))
    selected_clusters = st.multiselect(
        'select clusters to display keywords for:',
        options=cluster_ids,
        default=cluster_ids
    )

    if selected_clusters:
        st.subheader(f"Cluster Keywords — Week {selected}")
        for cl in selected_clusters:
            entry = scoredict[selected][cl]
            score = entry['score']
            keywords = entry['topics']
            st.markdown(f"**Cluster {cl} (score: {score:.3f})**")
            cols = st.columns(1)
            for idx, kw in enumerate(keywords):
                cols[idx % 1].write(f"- {kw}")
    else:
        st.info('No clusters selected.')

    st.markdown('---')
    st.markdown('Use the arrows to step through weeks.')
    st.markdown('Use the arrows to step through weeks.')
