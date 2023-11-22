import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
data = np.array([
    [80, 20, 66, 30, 87, 24],
    [63, 23, 28, 86, 31, 68],
    [35, 67, 66, 65, 83, 40],
    [77, 48, 56, 40, 51, 26],
    [60, 18, 39, 85, 100, 29],
    [35, 89, 66, 35, 51, 69],
    [40, 91, 62, 46, 26, 68],
    [68, 60, 64, 93, 38, 29],
    [66, 37, 45, 85, 46, 49],
    [70, 30, 45, 65, 16, 43],
    [57, 51, 42, 86, 48, 44],
    [38, 80, 14, 53, 67, 68],
    [68, 71, 43, 86, 63, 48],
    [54, 46, 95, 92, 88, 42],
    [58, 41, 43, 59, 14, 40],
    [93, 39, 36, 95, 81, 20],
])




language_codes = ['CN', 'CL', 'DE', 'IN', 'KR', 'UK', 'US', 'PL', 'TR','JO', 'ES', 'NL', 'FR', 'JP','IR', 'RU']
df = pd.DataFrame(data, language_codes)
df = pd.DataFrame({'data': data.tolist(), 'Language Code': language_codes})
# Calculate L1 distances
distances = cdist(data, data, metric='cosine')
# Create DataFrame with language pairs and distances
df_distances = pd.DataFrame(distances, columns=language_codes, index=language_codes)
df_distances.index.name = 'lang1'
df_distances.columns.name = 'lang2'
print(df_distances)
# Find neighbors
neigh = NearestNeighbors(n_neighbors=2, metric = 'precomputed')
top_nearest = neigh.fit(df_distances) 
# Make the graph
D = nx.from_scipy_sparse_array(neigh.kneighbors_graph())
for i, lang in enumerate(df_distances.index):
    D.nodes()[i]['lang'] = lang
    # print( G.nodes()[i]['lang'])
nx.draw(D, with_labels=True, labels = nx.get_node_attributes(D, 'lang'))
plt.savefig('graph_s.png')