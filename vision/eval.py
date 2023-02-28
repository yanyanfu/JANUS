import cv2
import ffmpeg
import pickle

import numpy as np

from fastprogress.fastprogress import progress_bar
from matplotlib import pyplot as plt
from pathlib import Path
from prep import *
from sklearn.cluster import KMeans


def calc_tf_idf(tfs, dfs):
    tf_idf = np.array([])
    for tf, df in zip(tfs, dfs):
        tf = tf / np.sum(tfs)
        idf = np.log(len(tfs) / (df + 1))
        tf_idf = np.append(tf_idf, tf * idf)

    return tf_idf


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def hit_rate_at_k(rs, k):
    hits = 0
    for r in rs:
        if np.sum(r[:k]) > 0: hits += 1

    return hits / len(rs)


def mean_reciprocal_rank(rs):
    """Score is reciprocal of the rank of the first relevant item

    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).

    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.61111111111111105
    >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> mean_reciprocal_rank(rs)
    0.5
    >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.75

    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)

    Returns:
        Mean reciprocal rank
    """
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])

def r_precision(r):
    """Score is precision after all relevant documents have been retrieved

    Relevance is binary (nonzero is relevant).

    >>> r = [0, 0, 1]
    >>> r_precision(r)
    0.33333333333333331
    >>> r = [0, 1, 0]
    >>> r_precision(r)
    0.5
    >>> r = [1, 0, 0]
    >>> r_precision(r)
    1.0

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)

    Returns:
        R Precision
    """
    r = np.asarray(r) != 0
    z = r.nonzero()[0]
    if not z.size:
        return 0.
    return np.mean(r[:z[-1] + 1])


def precision_at_k(r, k):
    """Score is precision @ k

    Relevance is binary (nonzero is relevant).

    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k


    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)

    Returns:
        Precision @ k

    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def average_precision(r):
    """Score is average precision (area under PR curve)

    Relevance is binary (nonzero is relevant).

    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> delta_r = 1. / sum(r)
    >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
    0.7833333333333333
    >>> average_precision(r)
    0.78333333333333333

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)

    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def mean_average_precision(rs):
    """Score is mean average precision

    Relevance is binary (nonzero is relevant).

    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
    >>> mean_average_precision(rs)
    0.78333333333333333
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
    >>> mean_average_precision(rs)
    0.39166666666666666

    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)

    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])

def recall_at_k(r, k, l):
    """Score is recall @ k

    Relevance is binary (nonzero is relevant).

    >>> r = [0, 0, 1]
    >>> recall_at_k(r, 1, 2)
    0.0
    >>> recall_at_k(r, 2, 2)
    0.0
    >>> recall_at_k(r, 3, 2)
    0.5
    >>> recall_at_k(r, 4, 2)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k


    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: the length or size of the relevant items

    Returns:
        Recall @ k

    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    assert l >= 1
    r2 = np.asarray(r)[:k] != 0
    if r2.size != k:
        raise ValueError('Relevance score length < k')
    return np.sum(r2)/l


def rank_stats(rs):
    ranks = []
    for r in rs:
        ranks.append(r.nonzero()[0][0] + 1)
    ranks = np.asarray(ranks)
    recipical_ranks = 1 / ranks
    return np.std(ranks), np.mean(ranks), np.median(ranks), np.mean(recipical_ranks)


def evaluate(rankings, top_k = [1, 5, 10]):
    output = {}
    for app in rankings:
        output[app] = {}
        app_rs = []
        for bug in rankings[app]:
            if bug == 'elapsed_time': continue
            output[app][bug] = {}
            bug_rs = []
            for report in rankings[app][bug]:
                output[app][bug][report] = {'ranks': []}
                r = []
                for labels, score in rankings[app][bug][report].items():
                    output[app][bug][report]['ranks'].append((labels, score))
                    if labels[0] == bug: r.append(1)
                    else: r.append(0)
                r = np.asarray(r)
                output[app][bug][report]['rank'] = r.nonzero()[0][0] + 1
                output[app][bug][report]['average_precision'] = average_precision(r)
                bug_rs.append(r)

            bug_rs_std, bug_rs_mean, bug_rs_med, bug_mRR = rank_stats(bug_rs)
            bug_mAP = mean_average_precision(bug_rs)

            output[app][bug]['Bug std rank'] = bug_rs_std
            output[app][bug]['Bug mean rank'] = bug_rs_mean
            output[app][bug]['Bug median rank'] = bug_rs_med
            output[app][bug]['Bug mRR'] = bug_mRR
            output[app][bug]['Bug mAP'] = bug_mAP
            for k in top_k:
                bug_hit_rate = hit_rate_at_k(bug_rs, k)
                output[app][f'Bug Hit@{k}'] = bug_hit_rate
            app_rs.extend(bug_rs)

        app_rs_std, app_rs_mean, app_rs_med, app_mRR = rank_stats(app_rs)
        app_mAP = mean_average_precision(app_rs)

        output[app]['App std rank'] = app_rs_std
        output[app]['App mean rank'] = app_rs_mean
        output[app]['App median rank'] = app_rs_med
        output[app]['App mRR'] = app_mRR
        output[app]['App mAP'] = app_mAP
        print(f'{app} Elapsed Time in Seconds', rankings[app]['elapsed_time'])
        print(f'{app} σ Rank', app_rs_std)
        print(f'{app} μ Rank', app_rs_mean)
        print(f'{app} Median Rank', app_rs_med)
        print(f'{app} mRR:', app_mRR)
        print(f'{app} mAP:', app_mAP)
        for k in top_k:
            app_hit_rate = hit_rate_at_k(app_rs, k)
            output[app][f'App Hit@{k}'] = app_hit_rate
            print(f'{app} Hit@{k}:', app_hit_rate)
    
    return output


def get_eval_results(evals, app, item):
    for bug in evals[app]:
        if bug == 'elapsed_time': continue
        for vid in evals[app][bug]:
            try:
                print(evals[app][bug][vid][item])
            except: continue


def evaluate_ranking(ranking, ground_truth):
    relevance = []
    for doc in ranking:
        if doc in ground_truth:
            relevance.append(1)
        else:
            relevance.append(0)

    r = np.asarray(relevance)
    first_rank = int(r.nonzero()[0][0] + 1)
    avg_precision = average_precision(r)
    recip_rank = 1 / first_rank

    ranks = []
    precisions = []
    recalls = []

    limit = 10

    for k in range(1, limit + 1):
        ranks.append(1 if first_rank <= k else 0)
        precisions.append(precision_at_k(r, k))
        recalls.append(recall_at_k(r, k, len(ground_truth)))

    results = {
        'first_rank': first_rank,
        'recip_rank': recip_rank,
        'avg_precision': avg_precision
    }

    for i in range(limit):
        k = i + 1
        results["rr@" + str(k)] = ranks[i]
        results["p@" + str(k)] = precisions[i]
        results["r@" + str(k)] = recalls[i]

    return results