import urllib.request, gzip
import pandas as pd
from ogb.nodeproppred import NodePropPredDataset

print("Downloading dataset...")
dataset = NodePropPredDataset(name="ogbn-arxiv", root="/tmp/dataset")
graph, labels = dataset[0]

print("Graph keys:", list(graph.keys()))
print("edge_index shape:", graph['edge_index'].shape)

print("Downloading titles...")
url = "https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz"
import os
os.makedirs("/tmp/dataset", exist_ok=True)
urllib.request.urlretrieve(url, "/tmp/dataset/titleabs.tsv.gz")

titles = pd.read_csv("/tmp/dataset/titleabs.tsv.gz", sep="\t", compression="gzip")
print("Titles shape:", titles.shape)
print("Titles columns:", titles.columns)
