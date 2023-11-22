import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid", palette="dark")
# 读取CSV文件
df = pd.read_csv('/home/flt669/LiZhou/langrank-combine/features/fea_cor/feature_correlation.csv')

del df["Genetic"]

del df["Featural"]

del df["Phonological"]

# del df["Inventory"]

del df["Inventory"]

g=sns.pairplot(df,diag_kind="kde")
# g.map_lower(sns.kdeplot, levels=4, color=".2")
g.map_lower(sns.kdeplot, fill=True, cmap="Blues")

# g = sns.PairGrid(df)
# g.map_upper(sns.scatterplot)
# g.map_lower(sns.kdeplot, fill=True)
# g.map_diag(sns.kdeplot,fill=True)


correlation_matrix = df.corr()

# Visualize correlation coefficients on the pairplot
for i, (row_name, row) in enumerate(correlation_matrix.iterrows()):
    for j, (col_name, value) in enumerate(row.iteritems()):
        if i > j:
            g.axes[j, i].annotate(f"{value:.2f}", (0.5, 0.9), xycoords="axes fraction", ha="center", fontsize=8, color="r")


# 打印DataFrame
# for i, j in zip(*plt.np.triu_indices_from(plt.np.ones(df.shape, dtype=bool))):
#     if i != j:
#         sns.regplot(data=df, x=df.columns[j], y=df.columns[i], scatter=False, color='red')


plt.savefig('/home/flt669/LiZhou/langrank-combine/features/fea_cor/cor4.png', dpi=300)
print(df)

