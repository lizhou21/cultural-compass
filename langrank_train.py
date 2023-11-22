import argparse
import pickle
import os, sys
root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)
import numpy as np
from langrank import prepare_train_file, train, rank_to_relevance
from scipy.stats import rankdata


code_convert = { 'ara': 'ar',
                 'ces': 'cs',
                 'deu': 'de',
                 'eng': 'en',
                 'spa': 'es',
                 'fas': 'fa',
                 'fra': 'fr',
                 'hin': 'hi',
                 'jpn': 'ja',
                 'kor': 'ko',
                 'nld': 'nl',
                 'pol': 'pl',
                 'rus': 'ru',
                 'tam': 'ta',
                 'tur': 'tr',
                 'zho': 'zh'}



def rerank(rank, without_idx=None):
    reranked = []
    for r in rank:
        r.pop(without_idx)
        rr = rankdata(r, method='min') - 1
        reranked.append(rr)
    return reranked

def train_langrank(task='sa', exclude_lang=None, feature='base',
                   num_leaves=16, max_depth=-1, learning_rate=0.1,
                   n_estimators=100, min_child_samples=5):
    # langs = ['ara', 'ces', 'deu', 'eng', 'fas',
    #          'fra', 'hin', 'jpn', 'kor', 'nld',
    #          'pol', 'rus', 'spa', 'tam', 'tur', 'zho'] # no tha
    langs = ['COLD', 'ChileOLD', 'DeTox', 'Hindi', 'KOLD', 'NJH_US', 'NJH_UK', 'PolEval', 'TurkishOLD']
    data_dir = f'datasets/{task}/'
    datasets = [os.path.join(data_dir, f'{l}.txt') for l in langs]

    ranking_f = open(f'rankings/{task}.pkl', 'rb')
    rank = pickle.load(ranking_f) # gold label

    # if exclude_lang is not None: # exclude for cross validation
    #     exclude_idx = langs.index(exclude_lang)
    #     langs.pop(exclude_idx)
    #     rank.pop(exclude_idx)
    #     rank = rerank(rank, exclude_idx)
    #     datasets.pop(exclude_idx)

    if exclude_lang is not None: # exclude for cross validation
        exclude_idx = langs.index(exclude_lang)
        langs.pop(exclude_idx)
        rank.pop(exclude_idx)#将当前test的数据集排名排除
        rank = rerank(rank, exclude_idx)
        datasets.pop(exclude_idx)
    else:
        exclude_lang = 'all' # for model file name

    model_save_dir = f'pretrained/{task.upper()}/{feature}/'
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    tmp_dir = 'tmp'
    preprocess = None
    prepare_train_file(datasets=datasets, langs=langs, rank=rank,
                       tmp_dir=tmp_dir, task=task.upper(), preprocess=preprocess,
                       feature=feature)

    output_model = f"{model_save_dir}/lgbm_model_{task}_{exclude_lang}.txt"

    feature_names = {
        "DataSpecific": ['transfer_dataset_size', 'task_data_size', 'ratio_dataset_size'],
        "TTR": ['transfer_ttr', 'task_ttr', 'distance_ttr'],
        "Typology5": ['genetic', 'syntactic', 'featural', 'phonological', 'inventory'],
        "Geography": ['geographical'],
        "Orthography": ['word_overlap'],
        "PRAG": ['distance_pron', 'distance_verb', 'emotion_dist', 'ltq_score'],
        "PRAGOFF": ['distance_pron', 'distance_verb', 'off_dist', 'ltq_score'],
        "Culture": ['pdi', 'idv', 'mas', 'uai', 'lto', 'ivr'],
        "Learned": ['learned'+str(i) for i in range(512)],
        "Colex": ['colex'+str(i) for i in range(200)],
        'OFF':['off_dist'],
        'EMO':['emotion_dist'],
        "noPDI": ['idv', 'mas', 'uai', 'lto', 'ivr'],
        "noIDV": ['pdi', 'mas', 'uai', 'lto', 'ivr'],
        "noMAS": ['pdi', 'idv', 'uai', 'lto', 'ivr'],
        "noUAI": ['pdi', 'idv', 'mas', 'lto', 'ivr'],
        "noLTO": ['pdi', 'idv', 'mas', 'uai', 'ivr'],
        "noIVR": ['pdi', 'idv', 'mas', 'uai', 'lto'],
    }



    feature_list = feature.split('_')
    feature_name = []
    for f in feature_list:
        feature_name += feature_names[f]




    print(f'Features used are {feature_name}')
    train(tmp_dir=tmp_dir, output_model=output_model, num_leaves=num_leaves,
          max_depth=max_depth, learning_rate=learning_rate,
          n_estimators=n_estimators, min_child_samples=min_child_samples,
          feature_name=feature_name, task=f"{task.upper()}")
    assert os.path.isfile(output_model)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='sa')
    parser.add_argument('--features', nargs='+')
    parser.add_argument('--num_leaves', type=int, default=16)
    parser.add_argument('--max_depth', type=int, default=-1)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--n_estimators', type=int, default=200)
    parser.add_argument('--min_child_samples', type=int, default=5)
    return parser.parse_args()


if __name__ == '__main__':
    # langs = ['ara', 'ces', 'deu', 'eng', 'fas',
    #          'fra', 'hin', 'jpn', 'kor', 'nld',
    #          'pol', 'rus', 'spa', 'tam', 'tur', 'zho']
    langs = ['COLD', 'ChileOLD', 'DeTox', 'Hindi', 'KOLD', 'NJH_US', 'NJH_UK', 'PolEval', 'TurkishOLD']
    args = parse_args()
    for f in args.features:
        for exclude in langs:
            print(f'\nStart training with {exclude} excluded for task {args.task}')
            print(f'Features: {f}')
            train_langrank(task=args.task, exclude_lang=exclude, feature=f,
                           num_leaves=args.num_leaves, max_depth=args.max_depth,
                           learning_rate=args.learning_rate,
                           n_estimators=args.n_estimators,
                           min_child_samples=args.min_child_samples)
            # train_langrank(task=args.task, feature=f,
            #                num_leaves=args.num_leaves, max_depth=args.max_depth,
            #                learning_rate=args.learning_rate,
            #                n_estimators=args.n_estimators,
            #                min_child_samples=args.min_child_samples)
