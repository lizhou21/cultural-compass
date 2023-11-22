import json
import pickle
languages = ["zho", "spa", "deu", "hin", "kor", "eng", "pol", "tur"]

file_path = '/home/nlp/ZL/TEST/langrank-combine/features/Colex2Lang/wn_concept_glove_embeddings'  # 文件路径
with open(file_path, 'r') as file:
    lines = file.readlines()

lang_emb = {}

for line in lines:
    line = line.strip().split()  # 使用strip()方法去除行尾的换行符
    if line[0] in languages:
        la = line[0]
        vec = line[1:]
        vec = [float(v) for v in vec]
        lang_emb[la] = vec

file_path = '/home/nlp/ZL/TEST/langrank-combine/features/Colex2Lang/colex2lang.pkl'  # 文件路径

# 保存字典到文件
with open(file_path, 'wb') as file:
    pickle.dump(lang_emb, file)
print('a')
    # print('a')
    

