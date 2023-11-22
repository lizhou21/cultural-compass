import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
plt.figure(figsize=(6, 4))
data = [[80, 20, 66, 30, 87, 24],
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
        [93, 39, 36, 95, 81, 20]]

row_names = ['CN',        'CL',         'DE',       'IN',       'KR',        'UK',          'US', 
             'PL',         'TR',        'JO',       'ES',       'NL',         'FR',         'JP',
             'IR',          'RU']
column_names = ['pdi', 'idv', 'mas', 'uai', 'lto', 'ivr']
df = pd.DataFrame(data, index=row_names, columns=column_names)
correlation_matrix = df.corr()
np.fill_diagonal(correlation_matrix.values, np.nan)
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(correlation_matrix, cmap=cmap, annot=True, mask=np.triu(np.ones_like(correlation_matrix, dtype=bool)))
plt.ylim(correlation_matrix.shape[0], 0)
plt.title('Correlation of Different Cultural Dimensions')
plt.show()
plt.tight_layout()
plt.savefig('Correlation.png', dpi=300,)