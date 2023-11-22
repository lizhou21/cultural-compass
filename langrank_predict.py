import langrank as lr
from langrank import rank_to_relevance
import os
import argparse
from utils import ndcg
import pickle
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score
import numpy as np
from collections import defaultdict


# def ap_score(pred_rank, gold_rank, k=3):
#     def precision(pred_rank, gold_rank, k):
#         relevant_idx = [idx for idx, r in enumerate(gold_rank) if r <= k]
#         tp = 0
#         for idx, rank in enumerate(pred_rank):
#             if rank <= k and idx in relevant_idx:
#                 tp += 1
#             else:
#                 pass
#         return tp / k
#     prec_scores = [precision(pred_rank, gold_rank, rank) for idx, rank in enumerate(pred_rank) if rank <= k]
#     if prec_scores == []:
#         return 0
#     return np.mean(prec_scores)

def ap_score(pred_rank, gold_rank, rel_cutoff=3):
    def precision(pred_rank, gold_rank, k):
        rel_idx = [idx for idx, r in enumerate(gold_rank) if r <= rel_cutoff]
        correct = 0
        for idx in np.argsort(pred_rank)[:k]:
            if idx in rel_idx:
                correct += 1
        return correct / k
    prec_scores = []
    for idx in np.argsort(pred_rank):
        if gold_rank[idx] <= rel_cutoff:
            p = precision(pred_rank, gold_rank, pred_rank[idx])
            prec_scores.append(p)
    return np.mean(prec_scores)

def ndcg_score(pred_rank, gold_rank, k=3):
    # NDCG@3 as default
    num_lang = len(pred_rank)
    pred_rel = rank_to_relevance(pred_rank, num_lang)
    gold_rel = rank_to_relevance(gold_rank, num_lang)
    pred_rel = np.expand_dims(pred_rel, axis=0)
    gold_rel = np.expand_dims(gold_rel, axis=0)
    return ndcg(y_score=pred_rel, y_true=gold_rel, k=k)

def evaluate(pred_rank, gold_rank):
    ndcg_3 = ndcg_score(pred_rank, gold_rank, 3)
    ap_3 = ap_score(pred_rank, gold_rank, 3)
    return ndcg_3, ap_3

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='sa')
    parser.add_argument('--features', nargs='+')
    return parser.parse_args()

def make_args(lang, feature, task='SA'):
    params = argparse.Namespace()
    params.orig = f'datasets/{task}/{lang}.txt'
    params.seg = None
    params.lang = lang
    params.num = 3
    params.candidates = f'*{lang};'
    params.task = task
    params.model = lang
    params.feature = feature
    return params

def read_file(fpath):
    if fpath is None:
        return None
    with open(fpath) as inp:
        lines = inp.readlines()
    return lines

def sort_prediction(cand_list, neg_scores):
    try:
        where = 3 if cand_list[0].startswith('.') else 2
        cand_list = [c.split('/')[where][:3] for c in cand_list]
        assert len(set(cand_list)) != 1, 'something is wrong'
    except:
        pass
    sorted_list = sorted(zip(cand_list, neg_scores), key=lambda x: x[0])
    pred_neg_scores = [z[1] for z in sorted_list]
    pred = rankdata(pred_neg_scores, method='max')
    return pred

def load_gold(task, target_lang):
    fpath = f'rankings/{task}.pkl'
    f = open(fpath, 'rb')
    gold_list = pickle.load(f)
    if task == 'OLD':
        self_rank = {'COLD': 1, 'ChileOLD': 1, 'DeTox': 3, 'Hindi': 1, 'KOLD': 1, 'NJH_US': 1, 'NJH_UK': 1, 'PolEval': 5, 'TurkishOLD': 1}
        self_line = {'COLD': 1, 'ChileOLD': 2, 'DeTox': 3, 'Hindi': 4, 'KOLD': 5, 'NJH_US': 6, 'NJH_UK': 7, 'PolEval': 8, 'TurkishOLD': 9}
    
    current_rank = gold_list[self_line[target_lang]-1]

    current_rank.pop(current_rank.index(self_rank[target_lang]))

    sorted_list = sorted(current_rank)
    index_list = []
    for num in current_rank:
        index = sorted_list.index(num) + 1
        index_list.append(index)

    # for l in gold_list:
    #     l.pop(l.index(self_rank[target_lang])) # drop self

    # langs = ['ara', 'ces', 'deu', 'eng', 'fas',
    #          'fra', 'hin', 'jpn', 'kor', 'nld',
    #          'pol', 'rus', 'spa', 'tam', 'tur', 'zho']
    langs = ['COLD', 'ChileOLD', 'DeTox', 'Hindi', 'KOLD', 'NJH_US', 'NJH_UK', 'PolEval', 'TurkishOLD']
    target_lang_idx = langs.index(target_lang)
    return gold_list[target_lang_idx]


def summarize_result(result, features, metric):
    res = defaultdict(lambda: 0)
    for feat in features:
        for l, res_by_feat in result.items():
            res[feat] += res_by_feat[feat]
    print(f'Averaged result({ metric })')
    num_lang= len(result)
    for feat in features:
        avg = res[feat] / num_lang
        print(f'{feat}: {avg:.4f}', end='\t')
    print('\n')

def format_print(result, features):
    result = sorted([(l, res_by_feat) for l, res_by_feat in result.items()], key=lambda x: x[0])
    print('\t' + '\t'.join(features))
    for lang, res_by_feat in result:
        print(f'{lang}', end='')
        for feat in features:
            score = res_by_feat[feat]
            print(f'\t{score:.4f}', end='')
        print()


if __name__ == '__main__':
    args = parse_args()
    # langs = ['ara', 'ces', 'deu', 'eng', 'fas',
    #          'fra', 'hin', 'jpn', 'kor', 'nld',
    #          'pol', 'rus', 'spa', 'tam', 'tur', 'zho']
    langs = ['COLD', 'ChileOLD', 'DeTox', 'Hindi', 'KOLD', 'NJH_US', 'NJH_UK', 'PolEval', 'TurkishOLD']
    result = defaultdict(dict)
    eval_metric = ['ndcg', 'ap']
    result_map = defaultdict(dict)
    result_ndcg = defaultdict(dict)
    for l in langs:
        for f in args.features:
            params = make_args(l, f, f'{args.task}')
            assert os.path.isfile(params.orig)
            assert (params.seg is None or os.path.isfile(params.seg))
            lines = read_file(params.orig)
            bpelines = read_file(params.seg)

            prepared = lr.prepare_new_dataset(params.lang, task=params.task,
                                              dataset_source=lines, dataset_subword_source=bpelines)
            candidates = "all" if params.candidates == "all" else params.candidates.split(";")
            cand_langs, neg_predicted_scores = lr.rank(prepared, task=params.task, candidates=candidates, print_topK=params.num,
                                                       model=params.model, feature=params.feature)
            pred = sort_prediction(cand_langs, neg_predicted_scores)
            gold = load_gold(params.task, params.lang)
            ndcg_3, ap_3 = evaluate(pred, gold)

            # NDCG@3 score
            result_ndcg[params.lang][params.feature] = ndcg_3
            # AP@3 score
            result_map[params.lang][params.feature] = ap_3

            pred_langs = [cand_langs[i] for i in np.argsort(pred)[:3]]
            gold_langs = [cand_langs[i] for i in np.argsort(gold)[:3]]
            print('*'*80)
            print(f'Prediction for lang {params.lang} with {params.feature} features, {args.task} task')
            print(f'Prediction is {pred}')
            print(f'Top 3 prediction langs: {pred_langs}')
            print(f'Gold is {gold}')
            print(f'Top 3 gold langs: {gold_langs}')
            print(f'ndcg is {ndcg_3}')
            print(f'ap is {ap_3}')
            print('*'*80, end='\n\n')

    summarize_result(result_map, args.features, 'MAP')
    summarize_result(result_ndcg, args.features, 'NDCG')
    format_print(result_map, args.features)
    format_print(result_ndcg, args.features)
