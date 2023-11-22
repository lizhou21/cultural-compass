import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
plt.figure(figsize=(6, 4))
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

tsne = TSNE(n_components=2, perplexity=4,random_state=42)

M_tsne = tsne.fit_transform(data)

# 提取降维后的坐标
x = M_tsne[:, 0]
y = M_tsne[:, 1]

# 绘制t-SNE图
plt.scatter(x, y)

countries = ['CN',        'CL',         'DE',       'IN',       'KR',        'UK',          'US', 
             'PL',         'TR',        'JO',       'ES',       'NL',         'FR',         'JP',
             'IR',          'RU']
categories =['East Asia', 'South America', 'Western Europe', 'South Asia', 'East Asia', 'Western Europe', 'North America',
              'East Europe', 'Middle East', 'Middle East', 'Western Europe', 'Western Europe', 'Western Europe', 'East Asia',
              'Middle East', 'East Europe']
'''
East Asia:
CN: China
JP: Japan
KR: South Korea

South Asia:
IN: India


Western Europe:
NL: Netherlands 
FR: France
UK: the United Kingdom
ES: Spain
DE: German 

Middle East:
TR: Turkey 
JO: Jordan 
IR: Iran

South America:
CL: Chile

North America
US: the United States

East Europe:
PL: Poland 
RU: Russia



China,Chile,German,India,South Korea,the United Kingdom,the United States,Poland,Turkey,Jordan,Spain,Netherlands,France,Japan,Iran,Russia
'''

# unique_categories = list(set(categories))
unique_categories = ['East Asia', 'South Asia', 'Western Europe', 'East Europe', 'Middle East', 'South America', 'North America']
colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan','gray']

for category in unique_categories:
    # 提取属于当前类别的国家的索引
    indices = [i for i, cat in enumerate(categories) if cat == category]
    # 根据索引绘制对应颜色的散点图
    plt.scatter(x[indices], y[indices], marker='s', s=120, color=colors[unique_categories.index(category)], label=category, )


for i, country in enumerate(countries):
    plt.text(x[i], y[i], country)

plt.legend()
plt.savefig('tsne.png', dpi=300)
