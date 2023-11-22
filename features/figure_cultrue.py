import os
import csv
import numpy as np

def read_cultures(f):
    feature_dict = {}
    with open(f, 'r') as f:
        for line in f.readlines()[1:]:
            line = line.strip().split(',')
            lang = line[0]
            feature_dict[lang] = [float(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5]), float(line[6])]
            # feature_dict[lang] = {}
            # feature_dict[lang]['pdi'] = line[1]
            # feature_dict[lang]['idv'] = line[2]
            # feature_dict[lang]['mas'] = line[3]
            # feature_dict[lang]['uai'] = line[4]
            # feature_dict[lang]['lto'] = line[5]
            # feature_dict[lang]['ivr'] = line[6]
    return feature_dict

def cul_features(lang, feature, feature_dir='./features', data_dir='./mono'):
    out_file = os.path.join(feature_dir, 'culture.csv')

    feature_dict = read_cultures(out_file)
    return feature_dict[lang][feature]

feature = read_cultures('/home/flt669/LiZhou/langrank-combine/features/culture.csv')




# feature_dict = {}
# for first_k, first_v in feature.items():
#     feature_dict[first_k] = {}
#     for sec_k, sec_v in feature.items():
#         feature_dict[first_k][sec_k] = np.sum(np.abs(np.array(first_v) - np.array(sec_v)))
        # feature_dict[first_k][sec_k] =  np.linalg.norm(np.array(first_v) - np.array(sec_v))
        
       

print('a')
