import json
import ntpath
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

from collections import OrderedDict
from pathlib import Path
from eval import *
from utils import *


def execute_retrieval_run(run, similarities):
    ranking = {}
    query = run["query"]
    corpus = run["dup_corpus"] + run["non_dup_corpus"]

    query_tokens = query.split("-")
    query_sims = similarities[query_tokens[0]][query_tokens[1]][query_tokens[2]]

    for doc in corpus:
        doc_tokens = doc.split("-")
        ranking[doc] = query_sims[(doc_tokens[1], doc_tokens[2])]

    ranking = OrderedDict(sorted(ranking.items(), key=lambda t: t[1], reverse=True))
    return ranking


def run_settings(settings, similarities, config, systems_allowed=[]):

    all_results, all_rankings = [], []

    for run in settings:
        query = run["query"]
        query_tokens = query.split("-")

        if len(systems_allowed) != 0 and query_tokens[0] not in systems_allowed:
            continue

        ranking = execute_retrieval_run(run, similarities)
        ranking_results = evaluate_ranking(ranking, run["gnd_trh"])
        ranking_results["app"] = query_tokens[0]
        ranking_results["run_id"] = run["run_id"]
        ranking_results.update(config)
        ranking_info = {"run_id": run["run_id"], "query": query, "ranking": ranking}
        ranking_info.update(config)

        all_results.append(ranking_results)
        all_rankings.append(ranking_info)

    return all_results, all_rankings


def write_results(output_path, results):
    Path(output_path).mkdir(parents=True, exist_ok=True)
    pd.read_json(json.dumps(results)).to_csv(os.path.join(output_path, 'all_results.csv'),
                                                 index=False, sep=";")


def write_rankings(output_path, rankings):
    Path(output_path).mkdir(parents=True, exist_ok=True)
    write_json_line_by_line(rankings, os.path.join(output_path, 'all_rankings.csv'))


def convert_results_format(out_path, settings_path, model):
    techniques = ["weighted_lcs", "bovw", "bovw_weighted_lcs"]
    systems_allowed = []
    settings = load_settings(settings_path)
    all_rankings, all_results = [], []

    sim_files = find_file("rankings*.pkl", out_path)
    for sim_file in sim_files:
        model_similarities = pickle.load(open(sim_file, 'rb'))
        for technique in techniques:
            print("Running technique: ", technique)
            similarities = model_similarities[technique]
            config = {"technique": technique, 'model': model}
            results, rankings = run_settings(settings, similarities, config, systems_allowed)
            all_results.extend(results)
            all_rankings.extend(rankings)

    print("Writing results and rankings")
    write_results(out_path, all_results)
    write_rankings(out_path, all_rankings)
    print("done")


def get_info_to_ranking_results(ranking, ranking_results, run, dl_model, ir_model, new_technique, weight_str):
    new_model = dl_model + "-" + ir_model
    new_config = "({},{})".format(dl_model, ir_model)
    new_config_weight = "({},{},{})".format(weight_str, dl_model, ir_model)
    config = {
        "model": new_model,
        "technique": new_technique,
        "weight": weight_str,
        "model_config": new_config,
        "model_config_weight": new_config_weight
    }

    query = run["query"]
    query_tokens = query.split("-")

    ranking_results["app"] = query_tokens[0]
    ranking_results["run_id"] = run["run_id"]
    ranking_results.update(config)

    ranking_info = {"run_id": run["run_id"], "query": query, "ranking": ranking}
    ranking_info.update(config)

    return ranking_info, ranking_results


def combined(
    out_path, dl_rankings_path, ir_rankings_path, ir_bow_rankings_path,
    settings_path, dl_model, ir_model
):

    # read data
    settings = load_settings(settings_path)
    all_new_rankings, all_new_results = [], []
    dl_rankings = read_json_line_by_line(dl_rankings_path)
    dl_rankings_by_config = group_dict(dl_rankings, lambda rec: (rec['technique']))

    if ir_rankings_path:
        ir_rankings = read_json_line_by_line(ir_rankings_path)
        ir_rankings_by_config = group_dict(ir_rankings, lambda rec: (rec['technique'])) 
        dl_runs = group_dict(dl_rankings_by_config['bovw_weighted_lcs'], lambda rec: rec["run_id"])
        ir_runs = group_dict(ir_rankings_by_config['bovw_weighted_lcs'], lambda rec: rec["run_id"])

        for run in settings:
            run_id = run["run_id"]
            ir_run_ranking = ir_runs[run_id][0]["ranking"]
            dl_run_ranking = dl_runs[run_id][0]["ranking"]

            # rankings based on all weights
            for weight in np.arange(0, 1.1, 0.1):
                new_ranking = {}
                for doc in dl_run_ranking:
                    ir_score = 0 if doc not in ir_run_ranking else ir_run_ranking[doc]
                    dl_score = dl_run_ranking[doc]
                    new_score = weight * ir_score + (1 - weight) * dl_score
                    new_ranking[doc] = new_score

                ranking = OrderedDict(sorted(new_ranking.items(), key=lambda t: t[1], reverse=True))
                ranking_results = evaluate_ranking(ranking, run["gnd_trh"])

                ranking_info, ranking_results = get_info_to_ranking_results(ranking, ranking_results,
                                                                            run, dl_model, ir_model, 
                                                                            'bovw_weighted_lcs-bow_weighted_lcs', str(weight))

                all_new_results.append(ranking_results)
                all_new_rankings.append(ranking_info)

    
    if ir_bow_rankings_path:
        ir_rankings = read_json(ir_bow_rankings_path)
        ir_rankings_by_config = group_dict(ir_rankings, lambda rec: (rec['technique']))
        dl_runs = group_dict(dl_rankings_by_config['bovw'], lambda rec: rec["run_id"])
        ir_runs = group_dict(ir_rankings_by_config['bovw'], lambda rec: rec["runId"])

        for run in settings:
            run_id = run["run_id"]
            ir_run_ranking = ir_runs[str(run_id)][0]["ranking"]
            ir_run_ranking = dict(
                zip((rec["docName"] for rec in ir_run_ranking), (rec for rec in ir_run_ranking)))
            dl_run_ranking = dl_runs[run_id][0]["ranking"]

            # rankings based on all weights
            for weight in np.arange(0, 1.1, 0.1):
                new_ranking = {}
                for doc in dl_run_ranking:
                    ir_score = 0 if doc not in ir_run_ranking else ir_run_ranking[doc]["score"]
                    dl_score = dl_run_ranking[doc]
                    new_score = weight * ir_score + (1 - weight) * dl_score
                    new_ranking[doc] = new_score

                ranking = OrderedDict(sorted(new_ranking.items(), key=lambda t: t[1], reverse=True))
                ranking_results = evaluate_ranking(ranking, run["gnd_trh"])

                ranking_info, ranking_results = get_info_to_ranking_results(ranking, ranking_results,
                                                                            run, dl_model, ir_model,
                                                                            'bovw-bow', str(weight))

                all_new_results.append(ranking_results)
                all_new_rankings.append(ranking_info)
    
    print("Writing data")

    Path(out_path).mkdir(parents=True, exist_ok=True) 
    pd.read_json(json.dumps(all_new_results)).to_csv(os.path.join(out_path, 'all_results.csv'),
                                                     index=False, sep=";")
    write_json_line_by_line(all_new_rankings, os.path.join(out_path, 'all_rankings.json'))