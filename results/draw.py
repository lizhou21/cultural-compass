import pandas as pd 
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="darkgrid")

fig, ax = plt.subplots(2,1,constrained_layout=True, figsize=(4, 6))
data_file = '/home/flt669/LiZhou/langrank-combine/results/ablation.xlsx'
dataAll = pd.read_excel(data_file)

langrank_data = dataAll[dataAll['group']=='LangRank']
MTVEC_data = dataAll[dataAll['group']=='MTVEC']


axe_sub1 = sns.barplot(data=langrank_data, x="Features",y='Score', hue="metric", ax=ax[0])

axe_sub1.set_ylim(40, 75)



axe_sub1.legend_.remove()
handles, labels = axe_sub1.get_legend_handles_labels()
legend = axe_sub1.figure.legend(handles, labels, title="", loc="upper center", ncol=2, frameon=False)
legend.get_title().set_ha("center")

# for p in axe_sub1.patches:
#     ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center',
#                 xytext=(0, 5), textcoords='offset points')


axe_sub2 = sns.barplot(data=MTVEC_data, x="Features",y='Score', hue="metric", ax=ax[1])
axe_sub2.set_ylim(57, 82)

axe_sub2.legend_.remove()

handles, labels = axe_sub2.get_legend_handles_labels()

legend = axe_sub2.figure.legend(handles, labels, title="", loc="upper center", ncol=2, frameon=False)
legend.get_title().set_ha("center")



plt.tight_layout()
plt.savefig("/home/flt669/LiZhou/langrank-combine/results/langrank.png", bbox_inches="tight", dpi=300)
print('a')