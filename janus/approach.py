import copy
import cv2
import multiprocessing
import pickle
import utils
import multiprocessing

import numpy as np
import vision_transformer as vits

from collections import defaultdict, OrderedDict
from itertools import combinations, combinations_with_replacement, permutations
from joblib import Parallel, delayed
from pathlib import Path
from eval import *
from features import *
from prep import *
from tqdm.auto import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


def flatten_dict(d_in, d_out, parent_key):
    for k, v in d_in.items():
        if isinstance(v, dict):
            flatten_dict(v, d_out, parent_key + (k,))
        else:
            d_out[parent_key + (k,)] = v


def gen_extracted_features(vid_ds, mdl, fps, ftk):
    vid_ds_features = {}
    for app in tqdm(vid_ds.labels):
        vid_ds_features[app] = {}
        for bug in vid_ds[app]:
            vid_ds_features[app][bug] = {}
            for report in vid_ds[app][bug]:
                vid_ds_features[app][bug][report] = {
                    'features': extract_features(vid_ds[app][bug][report], mdl, fps, frames_to_keep = ftk)
                }

    return vid_ds_features


def get_text_features(txt_path, strategy):
    text = json.load(open(txt_path / strategy / 'preprocessed_docs.json', 'r'))
    all_text = defaultdict(lambda: defaultdict(defaultdict))
    all_text_vec = defaultdict(lambda: defaultdict(defaultdict))

    for br in text:
        app, bug, report = br.split('-')
        all_text[app][bug][report] = text[br]['docText']
    
    for app in all_text:
        cnt = 0
        corpus = []
        vectorizer = TfidfVectorizer()
        for bug in all_text[app]:
            for report in all_text[app][bug]:              
                corpus.append(all_text[app][bug][report])
        text_vec = vectorizer.fit_transform(corpus).toarray()
        for bug in all_text[app]:
            for report in all_text[app][bug]: 
                all_text_vec[app][bug][report] = text_vec[cnt]
                cnt += 1
            
    return all_text_vec


def get_text_frame_features(txt_path):
    all_text_files = list(txt_path.glob('*.json'))
    all_text = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    all_text_vec = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for text_file in all_text_files:
        texts = utils.read_json_line_by_line(text_file)
        br = str(text_file).split('/')[-1][:-5]
        app, bug, report = br.split('-')
        for text in texts:
            all_text[app][bug][report].append(utils.format_str(text['txt']))
    
    for app in all_text:
        cnt = 0
        corpus = []
        vectorizer = TfidfVectorizer()
        for bug in all_text[app]:
            for report in all_text[app][bug]:
                for frame in all_text[app][bug][report]:
                    corpus.append(frame)
        text_vec = vectorizer.fit_transform(corpus).toarray()
        for bug in all_text[app]:
            for report in all_text[app][bug]: 
                for frame in all_text[app][bug][report]:
                    tmp = [text_vec[cnt].tolist()]
                    all_text_vec[app][bug][report].append(tmp)
                cnt += 1
            
    return all_text_vec


def gen_tfidfs(vid_ds_features, vw, codebook, df):
    vid_tfids = defaultdict(
        lambda: defaultdict(dict)
    )

    for app in vid_ds_features:
        for bug in vid_ds_features[app]:
            for report in vid_ds_features[app][bug]:
                bovw = new_get_bovw(
                    vid_ds_features[app][bug][report]['features'],
                    codebook, vw
                )
                vid_tfids[app][bug][report] = calc_tf_idf(bovw, df)

    return vid_tfids


def gen_bovw_similarity(vid_ds, vid_ds_features, codebook, vw):
    results = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: defaultdict(float)
                    )
                )
            )
        )
    )

    vid_ds_features = copy.deepcopy(vid_ds_features)
    df = np.histogram(codebook.labels_, bins = range(vw + 1))[0]
    vid_tfids = gen_tfidfs(vid_ds_features, vw, codebook, df)
    for app, bugs in vid_ds.labels.items():
        l = [(bug, report) for bug in bugs for report in bugs[bug]]
        pairs = list(x for x in combinations_with_replacement(l, 2) if x[0] != x[1])
        for (bug_i, report_i), (bug_j, report_j) in pairs:
            results[app][bug_i][report_i][bug_j][report_j]['bovw'] = np.dot(vid_tfids[app][bug_i][report_i], vid_tfids[app][bug_j][report_j]) / (np.linalg.norm(vid_tfids[app][bug_i][report_i]) * np.linalg.norm(vid_tfids[app][bug_j][report_j]))

    return results


def fuzzy_LCS(X, Y, m, n, sim_func, mdl_frame_threshold = 0.0):
    LCSuff = [[0 for k in range(n + 1)] for l in range(m + 1)]
    LCSuff_weighted = [[0 for k in range(n + 1)] for l in range(m + 1)]

    # To store the length of
    # longest common substring
    result = result_weighted = 0

    # Following steps to build
    # LCSuff[m+1][n+1] in bottom up fashion
    for i in range(0, m + 1):
        for j in range(0, n + 1):
            if (i == 0 or j == 0):
                LCSuff[i][j] = 0
                LCSuff_weighted[i][j] = 0
                continue

            sim = sim_func(X[i - 1], Y[j - 1])
            if sim > mdl_frame_threshold:
                LCSuff[i][j] = LCSuff[i - 1][j - 1] + sim
                LCSuff_weighted[i][j] = LCSuff_weighted[i - 1][j - 1] + sim * (i / m) * (j / n)
                if LCSuff[i][j] > result:
                    result = LCSuff[i][j]
                    result_weighted = LCSuff_weighted[i][j]
            else:
                LCSuff[i][j] = 0
                LCSuff_weighted[i][j] = 0

    mini, maxi = min(m, n), max(m, n)
    sum_w = 0
    max_v = maxi + 1
    for i in reversed(range(1, mini + 1)):
        sum_w += (i / mini) * (max_v / maxi)
        max_v -= 1
    return result / min(m, n), result_weighted / sum_w


def fuzzy_LCS_multi(item):
    X, Y = item
    m, n = len(X), len(Y)
    mdl_frame_threshold = 0.0
    sim_func = vits.frame_sim
    LCSuff = [[0 for k in range(n + 1)] for l in range(m + 1)]
    LCSuff_weighted = [[0 for k in range(n + 1)] for l in range(m + 1)]

    # To store the length of longest common substring
    result = result_weighted = 0

    # Following steps to build LCSuff[m+1][n+1] in bottom up fashion
    for i in range(0, m + 1):
        for j in range(0, n + 1):
            if (i == 0 or j == 0):
                LCSuff[i][j] = 0
                LCSuff_weighted[i][j] = 0
                continue

            sim = sim_func(X[i - 1], Y[j - 1])
            if sim > mdl_frame_threshold:
                LCSuff[i][j] = LCSuff[i - 1][j - 1] + sim
                LCSuff_weighted[i][j] = LCSuff_weighted[i - 1][j - 1] + sim * (i / m) * (j / n)
                if LCSuff[i][j] > result:
                    result = LCSuff[i][j]
                    result_weighted = LCSuff_weighted[i][j]
            else:
                LCSuff[i][j] = 0
                LCSuff_weighted[i][j] = 0

    mini, maxi = min(m, n), max(m, n)
    sum_w = 0
    max_v = maxi + 1
    for i in reversed(range(1, mini + 1)):
        sum_w += (i / mini) * (max_v / maxi)
        max_v -= 1
    return result / min(m, n), result_weighted / sum_w


def gen_lcs_similarity(vid_ds, vid_ds_features, sim_func):
    results = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: defaultdict(float)
                    )
                )
            )
        )
    )

    vid_ds_features = copy.deepcopy(vid_ds_features)
    for app, bugs in vid_ds.labels.items():
        l = [(bug, report) for bug in bugs for report in bugs[bug]]
        pairs = list(x for x in combinations_with_replacement(l, 2) if x[0] != x[1])
        for (bug_i, report_i), (bug_j, report_j) in tqdm(pairs):
            lcs_sim, weighted_lcs_sim = fuzzy_LCS(
                vid_ds_features[app][bug_i][report_i]['features'],
                vid_ds_features[app][bug_j][report_j]['features'],
                len(vid_ds_features[app][bug_i][report_i]['features']),
                len(vid_ds_features[app][bug_j][report_j]['features']),
                sim_func
            )
            results[app][bug_i][report_i][bug_j][report_j]['lcs'] = lcs_sim
            results[app][bug_i][report_i][bug_j][report_j]['weighted_lcs'] = weighted_lcs_sim

    return results


def gen_text_similarity(vid_ds, all_text_vec):
    results = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: defaultdict(float)
                    )
                )
            )
        )
    )
    for app, bugs in vid_ds.labels.items():
        l = [(bug, report) for bug in bugs for report in bugs[bug]]
        pairs = list(x for x in combinations_with_replacement(l, 2) if x[0] != x[1])
        for (bug_i, report_i), (bug_j, report_j) in pairs:
            results[app][bug_i][report_i][bug_j][report_j]['bovw'] = np.dot(all_text_vec[app][bug_i][report_i], all_text_vec[app][bug_j][report_j]) / (np.linalg.norm(all_text_vec[app][bug_i][report_i]) * np.linalg.norm(all_text_vec[app][bug_j][report_j]))
    return results


def gen_text_lcs_similarity(vid_ds, vid_ds_features, sim_func):
    results = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: defaultdict(float)
                    )
                )
            )
        )
    )

    vid_ds_features = copy.deepcopy(vid_ds_features)
    for app, bugs in tqdm(vid_ds.labels.items()):
        l = [(bug, report) for bug in bugs for report in bugs[bug]]
        pairs = list(x for x in combinations_with_replacement(l, 2) if x[0] != x[1])
        for (bug_i, report_i), (bug_j, report_j) in tqdm(pairs):
            lcs_sim, weighted_lcs_sim = fuzzy_LCS(
                vid_ds_features[app][bug_i][report_i],
                vid_ds_features[app][bug_j][report_j],
                len(vid_ds_features[app][bug_i][report_i]),
                len(vid_ds_features[app][bug_j][report_j]),
                sim_func
            )
            results[app][bug_i][report_i][bug_j][report_j]['lcs'] = lcs_sim
            results[app][bug_i][report_i][bug_j][report_j]['weighted_lcs'] = weighted_lcs_sim

    return results


def gen_text_lcs_similarity_multi(vid_ds, vid_ds_features):
    results = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: defaultdict(float)
                    )
                )
            )
        )
    )

    vid_ds_features = copy.deepcopy(vid_ds_features)
    for app, bugs in tqdm(vid_ds.labels.items()):
        l = [(bug, report) for bug in bugs for report in bugs[bug]]
        pairs = list(x for x in combinations_with_replacement(l, 2) if x[0] != x[1])
        agg_features = []
        for (bug_i, report_i), (bug_j, report_j) in tqdm(pairs):
            agg_features.append((vid_ds_features[app][bug_i][report_i], vid_ds_features[app][bug_j][report_j]))

        cpu_cont = 8
        pool = multiprocessing.Pool(cpu_cont)
        agg_results = pool.map(fuzzy_LCS_multi, tqdm(agg_features,total=len(agg_features)))
        pool.close()
        pool.join()

        for index, ((bug_i, report_i), (bug_j, report_j)) in enumerate(pairs):
            results[app][bug_i][report_i][bug_j][report_j]['lcs'] = agg_results[index][0]
            results[app][bug_i][report_i][bug_j][report_j]['weighted_lcs'] = agg_results[index][1]

    return results


def fix_sims(vid_sims, vid_ds):
    for sim_type in vid_sims:
        for app in vid_sims[sim_type]:
            l = [(bug, report) for bug in vid_ds[app] for report in vid_ds[app][bug]]
            pairs = reversed(list(x for x in permutations(l, 2) if x[0] != x[1]))
            for (bug_i, report_i), (bug_j, report_j) in pairs:
                if (bug_i, report_i) == (bug_j, report_j): continue
                vid_sims[sim_type][app][bug_i][report_i][bug_j][report_j] = vid_sims[sim_type][app][bug_j][report_j][bug_i][report_i]

    return vid_sims


def sort_rankings(vid_sims):
    sorted_rankings = {}
    for sim_type in vid_sims:
        sorted_rankings[sim_type] = {}
        for app in vid_sims[sim_type]:
            sorted_rankings[sim_type][app] = {'elapsed_time': 1}
            for bug in vid_sims[sim_type][app]:
                if bug == 'elapsed_time': continue
                sorted_rankings[sim_type][app][bug] = {}
                for report in vid_sims[sim_type][app][bug]:
                    sorted_rankings[sim_type][app][bug][report] = []
                    d_out = {}
                    flatten_dict(vid_sims[sim_type][app][bug][report], d_out, tuple())
                    sorted_rankings[sim_type][app][bug][report] = OrderedDict(
                        sorted(d_out.items(), key = lambda x: str(x[1]), reverse = True)
                    )

    return sorted_rankings


def approach(
    vid_ds, bovw_vid_ds_sims, lcs_vid_ds_sims
):
    vid_ds_sims = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: defaultdict(
                            lambda: defaultdict(float)
                        )
                    )
                )
            )
        )
    )

    bovw_vid_ds_sims = copy.deepcopy(bovw_vid_ds_sims)
    lcs_vid_ds_sims = copy.deepcopy(lcs_vid_ds_sims)
    for app, bugs in vid_ds.labels.items():
        l = [(bug, report) for bug in bugs for report in bugs[bug]]
        pairs = list(x for x in combinations_with_replacement(l, 2) if x[0] != x[1])
        for (bug_i, report_i), (bug_j, report_j) in pairs:
            lcs = lcs_vid_ds_sims[app][bug_i][report_i][bug_j][report_j]['lcs']
            weighted_lcs = lcs_vid_ds_sims[app][bug_i][report_i][bug_j][report_j]['weighted_lcs']
            vid_ds_sims['lcs'][app][bug_i][report_i][bug_j][report_j] = lcs
            vid_ds_sims['weighted_lcs'][app][bug_i][report_i][bug_j][report_j] = weighted_lcs

            bovw = bovw_vid_ds_sims[app][bug_i][report_i][bug_j][report_j]['bovw']
            vid_ds_sims['bovw'][app][bug_i][report_i][bug_j][report_j] = bovw
            vid_ds_sims['bovw_lcs'][app][bug_i][report_i][bug_j][report_j] = (bovw + lcs) / 2
            vid_ds_sims['bovw_weighted_lcs'][app][bug_i][report_i][bug_j][report_j] = (bovw + weighted_lcs) / 2

    fixed_vid_ds_sims = fix_sims(vid_ds_sims, vid_ds)
    rankings = sort_rankings(fixed_vid_ds_sims)
    return rankings


def compute_sims(q_vid, vid_ds, model, codebook, vw, fps, ftk):
    df = np.histogram(codebook.labels_, bins = range(vw + 1))[0]

    q_features = extract_features(q_vid, model, fps, frames_to_keep = ftk)
    bovw = new_get_bovw(
        q_features,
        codebook, vw
    )
    q_tfids = calc_tf_idf(bovw, df)

    vid_ds_features = gen_extracted_features(vid_ds, model, fps, ftk)
    vid_ds_tfids = gen_tfidfs(vid_ds_features, vw, codebook, df, ftk)
    results = {}
    for app in tqdm(vid_ds.labels):
        results[app] = {}
        for bug in vid_ds[app]:
            results[app][bug] = {}
            for report in vid_ds[app][bug]:
                results[app][bug][report] = np.dot(q_tfids, vid_ds_tfids[app][bug][report]) / (np.linalg.norm(q_tfids) * np.linalg.norm(vid_ds_tfids[app][bug][report]))

    d_out = {}
    flatten_dict(results, d_out, tuple())
    sorted_rankings = OrderedDict(
        sorted(d_out.items(), key = lambda x: str(x[1]), reverse = True)
    )

    return sorted_rankings