import lang2vec.lang2vec as l2v
import numpy as np
import pkg_resources
import os
import lightgbm as lgb
from sklearn.datasets import load_svmlight_file
from new_features import pos_features, emo_features, ltq_features, cul_features, off_features, colex_features
from scipy.spatial import distance
from scipy.spatial.distance import cdist
import pandas as pd

TASKS = ["DEP", "SA", 'OLD']

DEP_DATASETS = {
    "conll" : "conll.npy"
}

SA_DATASETS = {
    "sa": "sa.npy"
}

OLD_DATASETS = {
    "OLD": "OLD.npy"
}

DEP_MODELS = {
    "all":"lgbm_model_dep_all.txt",
    "ara":"lgbm_model_dep_ara.txt",
    "ces":"lgbm_model_dep_ces.txt",
    "deu":"lgbm_model_dep_deu.txt",
    "eng":"lgbm_model_dep_eng.txt",
    "fas":"lgbm_model_dep_fas.txt",
    "fra":"lgbm_model_dep_fra.txt",
    "hin":"lgbm_model_dep_hin.txt",
    "jpn":"lgbm_model_dep_jpn.txt",
    "kor":"lgbm_model_dep_kor.txt",
    "nld":"lgbm_model_dep_nld.txt",
    "pol":"lgbm_model_dep_pol.txt",
    "rus":"lgbm_model_dep_rus.txt",
    "spa":"lgbm_model_dep_spa.txt",
    "tam":"lgbm_model_dep_tam.txt",
    "tur":"lgbm_model_dep_tur.txt",
    "zho":"lgbm_model_dep_zho.txt"
}
SA_MODELS = {
    "all":"lgbm_model_sa_all.txt",
    "ara":"lgbm_model_sa_ara.txt",
    "ces":"lgbm_model_sa_ces.txt",
    "deu":"lgbm_model_sa_deu.txt",
    "eng":"lgbm_model_sa_eng.txt",
    "fas":"lgbm_model_sa_fas.txt",
    "fra":"lgbm_model_sa_fra.txt",
    "hin":"lgbm_model_sa_hin.txt",
    "jpn":"lgbm_model_sa_jpn.txt",
    "kor":"lgbm_model_sa_kor.txt",
    "nld":"lgbm_model_sa_nld.txt",
    "pol":"lgbm_model_sa_pol.txt",
    "rus":"lgbm_model_sa_rus.txt",
    "spa":"lgbm_model_sa_spa.txt",
    "tam":"lgbm_model_sa_tam.txt",
    "tur":"lgbm_model_sa_tur.txt",
    "zho":"lgbm_model_sa_zho.txt"
}

OLD_MODELS = {
    "ArabicOLD": "lgbm_model_OLD_ArabicOLD.txt",
    "DeTox": "lgbm_model_OLD_DeTox.txt",
    "KOLD": "lgbm_model_OLD_KOLD.txt",
    "NJH_UK": "lgbm_model_OLD_NJH_UK.txt",
    "NJH_US": "lgbm_model_OLD_NJH_US.txt",
    "OLID": "lgbm_model_OLD_OLID.txt",
    "TurkishOLD": "lgbm_model_OLD_TurkishOLD.txt",
    "ChileOLD": "lgbm_model_OLD_ChileOLD.txt",
    "COLD": "lgbm_model_OLD_COLD.txt",
    "Hindi": "lgbm_model_OLD_Hindi.txt",
    "PolEval": "lgbm_model_OLD_PolEval.txt"
}

# checks
def check_task(task):
    if task not in TASKS:
        raise Exception("Unknown task " + task + ". Only 'DEP', 'SA' are supported.")

def check_task_model(task, model):
    # langdict = {'KOLD': 'kor', 'COLD': 'zho', 'TurkishOLD': 'tur', 'ArabicOLD': 'ara', 'OLID': 'eng', 
    #          'DeTox': 'deu', 'NJH_US': 'eng', 'NJH_UK': 'eng', 'ChileOLD': 'spa'}
    # model = langdict[model]
    check_task(task)
    avail_models = map_task_to_models(task)
    if model not in avail_models:
        ll = ', '.join([key for key in avail_models])
        raise Exception("Unknown model " + model + ". Only "+ll+" are provided.")

def check_task_model_data(task, model, data):
    check_task_model(task, model)
    avail_data = map_task_to_data(task)
    if data not in avail_data:
        ll = ', '.join([key for key in avail_data])
        raise Exception("Unknown dataset " + data + ". Only "+ll+" are provided.")

# utils
def map_task_to_data(task):
    if task == "DEP":
        return DEP_DATASETS
    elif task == "SA":
        return SA_DATASETS
    elif task == "OLD":
        return OLD_DATASETS
    else:
        raise Exception("Unknown task")

def map_task_to_models(task):
    if task == "DEP":
        return DEP_MODELS
    elif task == "SA":
        return SA_MODELS
    elif task == "OLD":
        return OLD_MODELS
    else:
        raise Exception("Unknown task")

def read_vocab_file(fn):
    with open(fn) as inp:
        lines = inp.readlines()
    c = []
    v = []
    for l in lines:
        l = l.strip().split()
        if len(l) == 2:
            c.append(int(l[1]))
            v.append(l[0])
    return v,c

# used for ranking
def get_candidates(task, languages=None):
    if languages is not None and not isinstance(languages, list):
        raise Exception("languages should be a list of ISO-3 codes")

    datasets_dict = map_task_to_data(task)
    cands = []
    for dt in datasets_dict:
        fn = pkg_resources.resource_filename(__name__, os.path.join('indexed', task, datasets_dict[dt]))
        d = np.load(fn, encoding='latin1', allow_pickle=True).item()
        # languages with * means to exclude
        if task == 'SA':
            cands += [(key,d[key]) for key in d if '*' + key.split('/')[3][:3] not in languages]
        elif task == 'OLD':
            cands += [(key,d[key]) for key in d if '*' + key.split('/')[-1][:-4] not in languages]
        elif task == 'DEP':
            cands += [(key,d[key]) for key in d if '*' + key.split('_')[1] not in languages]
        cands = sorted(cands, key=lambda x: x[0])
    cand_langs = [i[0] for i in cands]
    if task == 'DEP':
        cand_langs = [i[0].split('_')[1] for i in cands]
    print(f"Candidate languages are: {cand_langs}")
    return cands

# extract dataset dependent feautures
def prepare_new_dataset(lang, task="SA", dataset_source=None,
                        dataset_target=None, dataset_subword_source=None,
                        dataset_subword_target=None):
    features = {}
    features["lang"] = lang

    # Get dataset features
    if dataset_source is None and dataset_target is None and dataset_subword_source is None and dataset_subword_target is None:
        # print("NOTE: no dataset provided. You can still use the ranker using language typological features.")
        return features
    elif dataset_source is None: # and dataset_target is None:
        # print("NOTE: no word-level dataset provided, will only extract subword-level features.")
        pass
    elif dataset_subword_source is None: # and dataset_subword_target is None:
        # print("NOTE: no subword-level dataset provided, will only extract word-level features.")
        pass


    source_lines = []
    if isinstance(dataset_source, str):
        with open(dataset_source) as inp:
            source_lines = inp.readlines()
    elif isinstance(dataset_source, list):
        source_lines = dataset_source
    else:
        raise Exception("dataset_source should either be a filnename (str) or a list of sentences.")
    if source_lines:
        features["dataset_size"] = len(source_lines)
        tokens = [w for s in source_lines for w in s.strip().split()]
        features["token_number"] = len(tokens)
        types = set(tokens)
        features["type_number"] = len(types)
        features["word_vocab"] = types
        features["type_token_ratio"] = features["type_number"]/float(features["token_number"])

    # read all pos features even if we might not use everything
    features["verb_ratio"] = pos_features(lang, 'verb')
    features["pron_ratio"] = pos_features(lang, 'pron')
    features["pdi"] = cul_features(lang, 'pdi')
    features["idv"] = cul_features(lang, 'idv')
    features["mas"] = cul_features(lang, 'mas')
    features["uai"] = cul_features(lang, 'uai')
    features["lto"] = cul_features(lang, 'lto')
    features["ivr"] = cul_features(lang, 'ivr')
    # features['colex'] = colex_features(lang)

    code = {'ara': 'arb', 'fas': 'pes'}
    langdict = {'KOLD': 'kor', 'COLD': 'zho', 'TurkishOLD': 'tur', 'ArabicOLD': 'ara', 'OLID': 'eng', 
             'DeTox': 'deu', 'NJH_US': 'eng', 'NJH_UK': 'eng', 'ChileOLD': 'spa', 'PolEval': 'pol', 'Hindi':'hin'}
    features['colex'] = np.array(colex_features(langdict[lang]))
    
    lang = code.get(langdict[lang], langdict[lang])# 在lang2vec中的
    if lang == 'eng':
        features["learned"] = np.zeros(512) # 512
    else:
        features["learned"] = np.array(l2v.get_features(lang, 'learned')[lang]) # 512
    return features

def uriel_distance_vec(languages):
    code = {'ara': 'arb', 'fas': 'pes', 'zho': 'cmn'}
    langdict = {'KOLD': 'kor', 'COLD': 'zho', 'TurkishOLD': 'tur', 'ArabicOLD': 'ara', 'OLID': 'eng', 
             'DeTox': 'deu', 'NJH_US': 'eng', 'NJH_UK': 'eng', 'ChileOLD': 'spa', 'PolEval': 'pol', 'Hindi':'hin'}

    languages = [langdict[a] for a in languages]

    new_languages = []
    for l in languages:
        new_languages.append(code.get(l, l))
    languages = new_languages

    geographic = l2v.geographic_distance(languages)
    genetic = l2v.genetic_distance(languages)
    inventory = l2v.inventory_distance(languages)
    syntactic = l2v.syntactic_distance(languages)
    phonological = l2v.phonological_distance(languages)
    featural = l2v.featural_distance(languages)
    uriel_features = [genetic, syntactic, featural, phonological, inventory, geographic]
    return uriel_features


def distance_vec(test, transfer, uriel_features, task, feature):
    langdict = {'KOLD': 'kor', 'COLD': 'zho', 'TurkishOLD': 'tur', 'ArabicOLD': 'ara', 'OLID': 'eng', 
             'DeTox': 'deu', 'NJH_US': 'eng', 'NJH_UK': 'eng', 'ChileOLD': 'spa', 'PolEval': 'pol', 'Hindi':'hin'}

    output = []
    # Dataset specific
    # Dataset Size
    transfer_dataset_size = transfer["dataset_size"]
    task_data_size = test["dataset_size"]
    ratio_dataset_size = float(transfer_dataset_size)/task_data_size
    # TTR
    transfer_ttr = transfer["type_token_ratio"]
    task_ttr = test["type_token_ratio"]
    distance_ttr = (1 - transfer_ttr/task_ttr) ** 2

    # Word overlap
    word_overlap = float(len(set(transfer["word_vocab"]).intersection(set(test["word_vocab"])))) / (transfer["type_number"] + test["type_number"])
    # Subword overlap

    # POS related features
    transfer_vr = transfer["verb_ratio"]
    transfer_pr = transfer["pron_ratio"]

    task_vr = test["verb_ratio"]
    task_pr = test["pron_ratio"]

    # Two choices available for distance computation
    # distance_verb = (1 - transfer_vr / task_vr) ** 2
    distance_verb = transfer_vr / task_vr
    # distance_pron = (1 - transfer_pr / task_pr) ** 2
    distance_pron = transfer_pr / task_pr

    emotion_dist = float(emo_features(langdict[test['lang']], langdict[transfer['lang']]))

    off_dist = off_features(langdict[test['lang']], langdict[transfer['lang']])

    ltq_score = ltq_features(langdict[test['lang']], langdict[transfer['lang']])

    # learned language vector
    rep_diff = test['learned'] - transfer['learned']
    colex_diff = test['colex'] - transfer['colex']

    # culture group # ratio
    pdi = float(transfer['pdi'])/float(test['pdi'])
    idv = float(transfer['idv'])/float(test['idv'])
    mas = float(transfer['mas'])/float(test['mas'])
    uai = float(transfer['uai'])/float(test['uai'])
    lto = float(transfer['lto'])/float(test['lto'])
    ivr = float(transfer['ivr'])/float(test['ivr'])

    transfer_cul = np.array([[float(transfer['pdi']), 
                             float(transfer['idv']), 
                             float(transfer['mas']), 
                             float(transfer['uai']), 
                             float(transfer['lto']), 
                             float(transfer['ivr'])]])
    
    test_cul = np.array([[float(test['pdi']), 
                         float(test['idv']), 
                         float(test['mas']), 
                         float(test['uai']), 
                         float(test['lto']), 
                         float(test['ivr'])]])

    cul_distances = float(cdist(transfer_cul, test_cul, metric='cosine')[0][0])

    # pdi = float(transfer['pdi'])-float(test['pdi'])
    # idv = float(transfer['idv'])-float(test['idv'])
    # mas = float(transfer['mas'])-float(test['mas'])
    # uai = float(transfer['uai'])-float(test['uai'])
    # lto = float(transfer['lto'])-float(test['lto'])
    # ivr = float(transfer['ivr'])-float(test['ivr'])

    # pdi = [float(transfer['pdi'])-float(test['pdi']), float(transfer['pdi'])/float(test['pdi'])]
    # idv = [float(transfer['idv'])-float(test['idv']), float(transfer['idv'])/float(test['idv'])]
    # mas = [float(transfer['mas'])-float(test['mas']), float(transfer['mas'])/float(test['mas'])]
    # uai = [float(transfer['uai'])-float(test['uai']), float(transfer['uai'])/float(test['uai'])]
    # lto = [float(transfer['lto'])-float(test['lto']), float(transfer['lto'])/float(test['lto'])]
    # ivr = [float(transfer['ivr'])-float(test['ivr']), float(transfer['ivr'])/float(test['ivr'])]


    feature_group = {
        "DataSpecific": [transfer_dataset_size, task_data_size, ratio_dataset_size],
        "TTR": [transfer_ttr, task_ttr, distance_ttr],
        "Typology5": uriel_features[:-1],
        "Geography": uriel_features[-1:],
        "Orthography": [word_overlap],
        "PRAG": [distance_pron, distance_verb, emotion_dist, ltq_score],
        "PRAGOFF": [distance_pron, distance_verb, off_dist, ltq_score],
        "Culture": [pdi, idv, mas, uai, lto, ivr],
        # "Culture": pdi + idv + mas + uai + lto + ivr,
        "Learned": rep_diff.tolist(),
        "Colex": colex_diff.tolist(),
        'OFF':[off_dist],
        'EMO':[emotion_dist],
        "noPDI": [idv, mas, uai, lto, ivr],
        "noIDV": [pdi, mas, uai, lto, ivr],
        "noMAS": [pdi, idv, uai, lto, ivr],
        "noUAI": [pdi, idv, mas, lto, ivr],
        "noLTO": [pdi, idv, mas, uai, ivr],
        "noIVR": [pdi, idv, mas, uai, lto],
        "cul": [cul_distances, off_dist, distance_ttr, distance_pron, distance_verb, emotion_dist, ltq_score]
    }
# DataSpecific_TTR_Orthography_Typology5_Geography

    feature_list = feature.split('_')
    feats = []
    for f in feature_list:
        feats += feature_group[f]
    
    # if feature == 'TTR_Typology5':
    #     feature = 'Pragmatic'
    # elif feature == 'DataSpecific_TTR_Orthography_Typology5_Geography':
    #     feature = 'LangRank'
    # elif feature == 'DataSpecific_TTR_Orthography_Typology5_Geography_PRAG':
    #     feature = 'LangRank+PRAG'
    # elif feature == 'DataSpecific_TTR_Orthography_Typology5_Geography_Culture':
    #     feature = 'LangRank+Culture'
    # elif feature == 'DataSpecific_TTR_Orthography_Typology5_Geography_PRAG_Culture':
    #     feature = 'LangRank+PRAG+Culture'
    # elif feature == 'Learned_PRAG':
    #     feature = 'MTVEC+PRAG'
    # elif feature == 'Learned_Culture'
    #     feature = 'MTVEC+Culture'
    # elif feature == 'Learned_PRAG_Culture':
    #     feature = 'MTVEC+PRAG+Culture'



    # if feature == 'base':
    #     feats = [word_overlap,
    #              transfer_dataset_size, task_data_size, ratio_dataset_size,
    #              transfer_ttr, task_ttr, distance_ttr]
    #     feats += uriel_features
    # elif feature == 'learned':
    #     feats = rep_diff.tolist()
    # elif feature == 'learned_ours':
    #     feats = rep_diff.tolist()
    #     feats += [distance_pron, distance_verb]# LCR
    #     feats += [emotion_dist] # ESD
    #     feats += [ltq_score] # LTQ
    # elif feature == 'learned_ours_nolcr':
    #     feats = rep_diff.tolist()
    #     feats += [emotion_dist]
    #     feats += [ltq_score]
    # elif feature == 'learned_ours_noltq':
    #     feats = rep_diff.tolist()
    #     feats += [distance_pron, distance_verb]
    #     feats += [emotion_dist]
    # elif feature == 'learned_ours_noesd':
    #     feats = rep_diff.tolist()
    #     feats += [distance_pron, distance_verb]
    #     feats += [ltq_score]

    # elif feature == 'all':
    #     feats = [word_overlap,
    #              transfer_dataset_size, task_data_size, ratio_dataset_size,
    #              transfer_ttr, task_ttr, distance_ttr]
    #     feats += [distance_pron, distance_verb]
    #     feats += [emotion_dist]
    #     feats += [ltq_score]
    #     feats += uriel_features
    # elif feature == 'all_no_lcr':
    #     feats = [word_overlap,
    #              transfer_dataset_size, task_data_size, ratio_dataset_size,
    #              transfer_ttr, task_ttr, distance_ttr]
    #     # feats += [distance_pron, distance_verb]
    #     feats += [emotion_dist]
    #     feats += [ltq_score]
    #     feats += uriel_features
    # elif feature == 'all_no_esd':
    #     feats = [word_overlap,
    #              transfer_dataset_size, task_data_size, ratio_dataset_size,
    #              transfer_ttr, task_ttr, distance_ttr]
    #     feats += [distance_pron, distance_verb]
    #     # feats += [emotion_dist]
    #     feats += [ltq_score]
    #     feats += uriel_features
    # elif feature == 'all_no_ltq':
    #     feats = [word_overlap,
    #              transfer_dataset_size, task_data_size, ratio_dataset_size,
    #              transfer_ttr, task_ttr, distance_ttr]
    #     feats += [distance_pron, distance_verb]
    #     feats += [emotion_dist]
    #     # feats += [ltq_score]
    #     feats += uriel_features

    # # below is for analyses
    # elif feature == 'typo_group':
    #     feats = uriel_features[:-1]
    # elif feature == 'geo_group':
    #     feats = uriel_features[-1:]
    # elif feature == 'cult_group':
    #     feats = [transfer_ttr, task_ttr, distance_ttr,
    #              distance_pron, distance_verb, ltq_score, emotion_dist]
    # elif feature == 'ortho_group':
    #     feats = [word_overlap]
    # elif feature == 'data_group':
    #     feats = [transfer_dataset_size, task_data_size, ratio_dataset_size]
    # elif feature == 'culture_new_group':
    #     feats = [pdi, idv, mas, uai, lto, ivr]
    # elif feature == 'culture_full_group':
    #     feats = [transfer_ttr, task_ttr, distance_ttr, pdi, idv, mas, uai, lto, ivr]
    # elif feature == 'base_culture':
    #     feats = [word_overlap,
    #              transfer_dataset_size, task_data_size, ratio_dataset_size,
    #              transfer_ttr, task_ttr, distance_ttr]
    #     feats += uriel_features
    #     feats += [pdi, idv, mas, uai, lto, ivr]

    # elif feature == 'base_p_c':
    #     feats = [word_overlap,
    #              transfer_dataset_size, task_data_size, ratio_dataset_size,
    #              transfer_ttr, task_ttr, distance_ttr]
    #     feats += [distance_pron, distance_verb]
    #     feats += [emotion_dist]
    #     feats += [ltq_score]
    #     feats += uriel_features
    #     feats += [pdi, idv, mas, uai, lto, ivr]
    return np.array(feats)

def rank_to_relevance(rank, num_lang): # so that lower ranks are given higher relevance
    if isinstance(rank, list):
        rank = np.array(rank)
    return np.where(rank != 0, -rank + num_lang, 0)
    # rel =  np.where(rank != 0, -rank + num_lang, 0)

    # cutoff = 10 # top 10 as learning signal
    # rel = np.where(rel >= num_lang - cutoff, rel, 0)
    # return rel

# preparing the file for training
def prepare_train_file(datasets, langs, rank, segmented_datasets=None,
                       task="SA", tmp_dir="tmp", preprocess=None, feature='all'):
    num_langs = len(langs)

    if not isinstance(rank, np.ndarray):
        rank = np.array(rank)
    relevance = rank_to_relevance(rank, len(langs))

    features = {}
    for i, (ds, lang) in enumerate(zip(datasets, langs)):
        with open(ds, "r") as ds_f:
            if preprocess is not None:
                lines = [preprocess(l.strip()) for l in ds_f.readlines()]
            else:
                lines = ds_f.readlines()#读取source文本
        seg_lines = None
        if segmented_datasets is not None:
            sds = segmented_datasets[i]
            with open(sds, "r") as sds_f:
                seg_lines = sds_f.readlines()
        features[lang] = prepare_new_dataset(lang=lang, task=task, dataset_source=lines,
                                             dataset_subword_source=seg_lines)#获取当前source的feature
    # save data numpy
    # features_save = {}
    # for i, (ds, lang) in enumerate(zip(datasets, langs)):
    #     features_save[ds]=features[lang]
    # np.save('OLD', features_save)

    uriel = uriel_distance_vec(langs)

    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    train_file = os.path.join(tmp_dir, f"train_{task}.csv")
    train_file_f = open(train_file, "w")
    train_size = os.path.join(tmp_dir, f"train_{task}_size.csv")
    train_size_f = open(train_size, "w")
        
    all_feature = []
    for i, lang1 in enumerate(langs):
        for j, lang2 in enumerate(langs):
            single_lang_pair = [lang2, lang1]
            if i != j:
                uriel_features = [u[i, j] for u in uriel] # 获取当前语言对的uriel特征 (genetic, syntactic, featural, phonological, inventory, geographic)： 6维度
                single_lang_pair.extend(uriel_features)
                distance_vector = distance_vec(features[lang1], features[lang2], uriel_features, task, feature) # 当前语言对的距离vector
                # distance_vector.tolist()
                single_lang_pair.extend([float(i) for i in distance_vector.tolist()])
                distance_vector = ["{}:{}".format(i, d) for i, d in enumerate(distance_vector)]
                line = " ".join([str(relevance[i, j])] + distance_vector) # svmlight format
                train_file_f.write(line + "\n")
                all_feature.append(single_lang_pair)
        train_size_f.write("{}\n".format(num_langs-1))
    # name_cul = ['transfer_lang', 'target_lang',	'Genetic', 'Syntactic',	'Featural',	
    #             'Phonological',	'Inventory', 'Geographic', 'CulDim', 'OffDist',	'Distance TTR',
    #             'LCR Noun',	'LCR Verb',	'ESD',	'LTQ']
    # all_feature=pd.DataFrame(all_feature, columns=name_cul)
    # all_feature.to_csv('/home/flt669/LiZhou/langrank-combine/features/fea_cor/feature_correlation.csv', index=False)
    train_file_f.close()
    train_size_f.close()

def train(tmp_dir, output_model, num_leaves=16, max_depth=-1, learning_rate=0.1,
          n_estimators=100, min_child_samples=5, feature_name='auto', task='SA'):
    train_file = os.path.join(tmp_dir, f"train_{task}.csv")
    train_size = os.path.join(tmp_dir, f"train_{task}_size.csv")
    X_train, y_train = load_svmlight_file(train_file)# X: (210,17), 210 language-pairs, 17 dimension feature; Y: (210, )rank
    print('Training in prog...')# LightGBM: Light Gradient Boosting Machine, 一种梯度提升框架，使用决策树为基学习器
    model = lgb.LGBMRanker(boosting_type='gbdt', num_leaves=num_leaves,
                           max_depth=max_depth, learning_rate=learning_rate,
                           n_estimators=n_estimators, min_child_samples=min_child_samples)
    model.fit(X_train, y_train, group=np.loadtxt(train_size),
              feature_name=feature_name)
    model.booster_.save_model(output_model)
    print(f'Model saved at {output_model}')

def rank(test_dataset_features, task="SA", candidates="all", model="best", feature='base', print_topK=3):
    '''
    est_dataset_features : the output of prepare_new_dataset(). Basically a dictionary with the necessary dataset features.
    '''
    # Checks
    check_task_model(task, model)

    # Get candidates to be compared against
    if candidates == 'all':
        candidate_list = get_candidates(task)
    else:
        # Restricts to a specific set of languages
        candidate_list = get_candidates(task, candidates)

    languages = [test_dataset_features["lang"]] + [c[1]["lang"] for c in candidate_list]
    uriel = uriel_distance_vec(languages)

    test_inputs = []
    for i,c in enumerate(candidate_list):
        key = c[0]
        cand_dict = c[1]
        candidate_language = key[-3:]
        uriel_j = [u[0,i+1] for u in uriel]
        distance_vector = distance_vec(test_dataset_features, cand_dict, uriel_j, task,  feature)
        test_inputs.append(distance_vector)

    # load model
    model_dict = map_task_to_models(task) # this loads the dict that will give us the name of the pretrained model
    # langdict = {'KOLD': 'kor', 'COLD': 'zho', 'TurkishOLD': 'tur', 'ArabicOLD': 'ara', 'OLID': 'eng', 
    #          'DeTox': 'deu', 'NJH_US': 'eng', 'NJH_UK': 'eng', 'ChileOLD': 'spa'}
    # model = langdict[model]
    model_fname = model_dict[model] # this gives us the filename (needs to be joined, see below)
    modelfilename = pkg_resources.resource_filename(__name__, os.path.join('pretrained', task, feature, model_fname))
    print(f"Loading model... {modelfilename}")

    # rank
    bst = lgb.Booster(model_file=modelfilename)

    # print("predicting...")
    predict_contribs = bst.predict(test_inputs, pred_contrib=True)
    predict_scores = predict_contribs.sum(-1)

    cand_langs = [c[0] for c in candidate_list]
    return cand_langs, -predict_scores # small is good

