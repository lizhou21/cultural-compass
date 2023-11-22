import os
import re
import sys
import ast
from copy import copy
from collections import Counter, defaultdict
from tqdm import tqdm
import os
import sys
from functools import partial
import pickle

lang2code = {
    'ara': 'ar', 'ces': 'cs',
    'deu': 'de', 'eng': 'en',
    'fas': 'fa', 'fra': 'fr',
    'hin': 'hi', 'jpn': 'ja',
    'kor': 'ko', 'nld': 'nl',
    'rus': 'ru', 'pol': 'pl',
    'spa': 'es', 'tam': 'ta',
    'tur': 'tr', 'zho': 'zh'
}

NOUN_TAGS = {
    'kor': ['NNG'],
    'jpn': ['名詞']
}
PRONOUN_TAGS = {
    'kor': ['NP'],
    'jpn': ['代名詞']
}
VERB_TAGS = {
    'kor': ['VV'],
    'jpn': ['動詞']
}

POS_FEATURES = ['pron', 'verb']


def listdir(dir_):
    return [os.path.join(dir_, f) for f in os.listdir(dir_)]

def find_rdr_dict(resource_path, kind='UPOS'):
    files = listdir(resource_path)
    rdr_path = [f for f in files if kind in f and f.endswith('RDR')][0]
    dict_path = [f for f in files if kind in f and f.endswith('DICT')][0]
    return rdr_path, dict_path

def get_resources_path(pos_tagger_dir, ud_resources, etc_resources):
    resources_by_lang = {}
    for lang, resource_id in ud_resources.items():
        resource_dir = f'Models/ud-treebanks-v2.4/UD_{resource_id}'
        resource_path = os.path.join(pos_tagger_dir, resource_dir)
        rdr_path, dict_path = find_rdr_dict(resource_path)
        resources_by_lang[lang] = [rdr_path, dict_path]
    for lang, resource_path in etc_resources.items():
        resource_path = [os.path.join(pos_tagger_dir, p) for p in resource_path]
        resources_by_lang[lang] = resource_path
    return resources_by_lang

def get_sample(fpath, num=5):
    samples = []
    with open(fpath, 'r') as f:
        f.readline() # remove header
        for _ in range(num):
            sample = f.readline().strip('\n').split('\t')[1]
            samples.append(sample)
        return samples

def load_sample_data(base_path, langs, num=5):
    samples_by_lang = {}
    for lang in langs:
        lang_dir = os.path.join(base_path, lang)
        print(lang_dir)
        fpath = [f for f in listdir(lang_dir) if f.endswith('.tsv')][0] # arbitrary data
        samples = get_sample(fpath, num)
        samples_by_lang[lang] = samples
    return samples_by_lang

def load_pos_taggers(pos_tagger_dir, resources_by_lang):
    py_tagger_path = os.path.join(pos_tagger_dir, 'pSCRDRtagger')
    os.chdir(py_tagger_path)
    import sys; sys.path.append('.')
    from RDRPOSTagger import RDRPOSTagger, readDictionary
    os.chdir('../../')
    taggers_by_lang = {}
    for lang, resources in resources_by_lang.items():
        tagger = RDRPOSTagger()
        rdr_path, dict_path = resources
        tagger.constructSCRDRtreeFromRDRfile(rdr_path)
        dict_ = readDictionary(dict_path)
        taggers_by_lang[lang] = partial(tagger.tagRawSentence, DICT=dict_)
    return taggers_by_lang

def fetch_files(cond, data_dir):
    return sorted([os.path.join(data_dir, f) for f
                   in os.listdir(data_dir) if cond in f])

def read_file(fname):
    with open(fname, 'r') as f:
        lines = [l.strip() for l in f.readlines()]
    return lines

def parse_pos(line, lang):
    lst = ast.literal_eval(line)
    if lang == 'kor':
        pos = [tag[1].split('+')[0] for tag in lst]
    elif lang == 'nld':
        pos = [tag[1].split('.')[0].upper() for tag in lst]
    else:
        pos = [tag[1] for tag in lst]
    return pos

def count_pos(lines, lang):
    counts = Counter()
    for l in tqdm(lines, desc=lang):
        counts.update(parse_pos(l, lang))
    return counts

def ratio_x2y(x, y):
    n2v = x / (x + y)
    return n2v

def build_counts(data_dir):
    pos_counts = {}
    for lang, code in lang2code.items():
        fname = f'{data_dir}/{code}_pos.txt'
        print(f'Reading {fname} ...')
        lines = read_file(fname)
        counts = count_pos(lines, lang)
        pos_counts[lang] = counts
    return pos_counts

def get_pos_ratio(lang, counter, pos):
    assert pos in ['verb', 'pron']
    num_tokens = sum(counter.values())
    if pos == 'noun':
        tag = NOUN_TAGS.get(lang, ['NOUN'])
    elif pos == 'verb':
        tag = VERB_TAGS.get(lang, ['VERB'])
    else:
        tag = PRONOUN_TAGS.get(lang, ['PRON'])
    cnt = sum([counter.get(t, 0) for t in tag]) / num_tokens
    return cnt

def get_feature(lang, counter, name):
    if name in ['pron', 'verb']:
        return get_pos_ratio(lang, counter, name)
    else:
        raise ValueError('Feature name should be pron, verb.')


def build_features(data_dir, feature_dir, feature_name):
    pos_fpath = os.path.join(feature_dir, 'pos-ratio.csv')
    feature_fpath = os.path.join(feature_dir, f'{feature_name}.csv')
    if os.path.isfile(pos_fpath):
        import pandas as pd
        df = pd.read_csv(pos_fpath, index_col=0)
        feature_df = df[feature_name]
        feature_df.to_csv(feature_fpath)
        feature_dict = read_features(feature_fpath)
    elif os.path.isfile(feature_fpath):
        feature_dict = read_features(feature_fpath)
    else:
        if isinstance(feature_name, str):
            feature_name = [feature_name]
        pos_count_dict = build_counts(data_dir)
        feature_dict = {}
        print(f"Building {feature_name} features ...")
        for lang, pos_counts in pos_count_dict.items():
            features = [get_feature(lang, pos_counts, n) for n in feature_name]
            feature_dict[lang] = tuple(features)
    return feature_dict


def read_features(f):
    feature_dict = {}
    with open(f, 'r') as f:
        for line in f.readlines()[1:]:
            lang = line.split(',')[0]
            features = map(float, line.split(',')[1:])
            feature_dict[lang] = tuple(features)[0]
    return feature_dict

def read_cultures(f):
    feature_dict = {}
    with open(f, 'r') as f:
        for line in f.readlines()[1:]:
            line = line.strip().split(',')
            lang = line[0]
            feature_dict[lang] = {}
            feature_dict[lang]['pdi'] = line[1]
            feature_dict[lang]['idv'] = line[2]
            feature_dict[lang]['mas'] = line[3]
            feature_dict[lang]['uai'] = line[4]
            feature_dict[lang]['lto'] = line[5]
            feature_dict[lang]['ivr'] = line[6]
    return feature_dict

def write_output(feature_dict, col_name, out_file):
    if isinstance(col_name, str):
        col_name = [col_name]
    col_name = ['lang'] + col_name
    header = ','.join(col_name) + '\n'
    with open(out_file, 'w') as f:
        f.write(header)
        for lang, features in feature_dict.items():
            if isinstance(features, list):
                row = [lang] + list(map(str, features))
            else:
                row = [lang, str(features)]
            row = ','.join(row)
            print(row, file=f)
    print(f'Results saved as {out_file}')


def pos_features(lang, feature, feature_dir='./features', data_dir='./mono'):
    assert feature in POS_FEATURES
    out_file = os.path.join(feature_dir, f'{feature}.csv')

    if not os.path.isfile(out_file):
        if 'news' in feature_dir:
            data_dir = './mono-news-processed'
        feature_dict = build_features(data_dir, feature_dir, feature)
        write_output(feature_dict, feature, out_file)
    else:
        feature_dict = read_features(out_file)
    langdict = {'KOLD': 'kor', 'COLD': 'zho', 'TurkishOLD': 'tur', 'ArabicOLD': 'ara', 'OLID': 'eng', 
             'DeTox': 'deu', 'NJH_US': 'eng', 'NJH_UK': 'eng', 'ChileOLD': 'spa', 'PolEval': 'pol', 'Hindi':'hin'}
    return feature_dict[langdict[lang]]

def cul_features(lang, feature, feature_dir='./features', data_dir='./mono'):
    out_file = os.path.join(feature_dir, 'culture.csv')

    feature_dict = read_cultures(out_file)
    return feature_dict[lang][feature]

def colex_features(lang, feature_dir='./features/Colex2Lang', data_dir='./mono'):
    out_file = os.path.join(feature_dir, 'colex2lang.pkl')
    with open(out_file, 'rb') as file:
        feature_dict = pickle.load(file)
    return feature_dict[lang]

def emo_features(lang1, lang2, fpath='./features/', pairwise=True):
    if pairwise:
        fpath = os.path.join(fpath, 'emo-diffs-cc-cos-5iter-zero-one-norm.txt') # en
    else:
        pass

    feature_dict = defaultdict(dict)
    with open(fpath) as f:
        for line in f:
            lang1_code, lang2_code, emo_score = line.split('\t')
            feature_dict[lang1_code][lang2_code] = emo_score
    if lang1 == lang2:
        return 0.0
    return feature_dict[lang2code[lang1]][lang2code[lang2]].strip()

def ltq_features(lang1, lang2, fpath='./features/', norm=True):
    if norm:
        fpath = os.path.join(fpath, 'ltq_500_norm_download.txt')
    else:
        fpath = os.path.join(fpath, 'ltq_either_norm.txt')

    feature_dict = defaultdict(dict)
    with open(fpath) as f:
        for line in f:
            lang1_code, lang2_code, ltq_score = line.split('\t')
            feature_dict[lang1_code][lang2_code] = ltq_score
    if lang1 == lang2:
        return 0.0
    return feature_dict[lang2code[lang1]][lang2code[lang2]].strip()


def off_features(lang1, lang2, fpath='./features/', pairwise=True):
    if pairwise:
        fpath = os.path.join(fpath, 'avg_cos_dist_16.txt') # en
    else:
        pass

    feature_dict = defaultdict(dict)
    with open(fpath) as f:
        for line in f:
            lang1_code, lang2_code, emo_score = line.split()
            feature_dict[lang1_code][lang2_code] = emo_score
    if lang1 == lang2:
        return 0.0
    return feature_dict[lang2code[lang1]][lang2code[lang2]].strip()