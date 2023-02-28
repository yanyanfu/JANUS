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
    all_results = {}
    all_rankings = {}
    for setting in settings:
        all_results[setting] = []
        all_rankings[setting] = []

        runs = settings[setting]

        print("Running setting", setting)
        for run in runs:
            query = run["query"]
            query_tokens = query.split("-")

            if len(systems_allowed) != 0 and query_tokens[0] not in systems_allowed:
                continue

            ranking = execute_retrieval_run(run, similarities)
            ranking_results = evaluate_ranking(ranking, run["gnd_trh"])

            ranking_results["setting"] = setting
            ranking_results["app"] = query_tokens[0]
            ranking_results["run_id"] = run["run_id"]
            ranking_results.update(config)

            ranking_info = {"run_id": run["run_id"], "query": query, "ranking": ranking}
            ranking_info["setting"] = setting
            ranking_info.update(config)

            all_results[setting].append(ranking_results)
            all_rankings[setting].append(ranking_info)

    return all_results, all_rankings


def write_results(output_path, results):
    Path(output_path).mkdir(parents=True, exist_ok=True)
    all_results = []
    for setting in results:
        pd.read_json(json.dumps(results[setting])).to_csv(os.path.join(output_path, setting + '.csv'),
                                                          index=False, sep=";")
        all_results.extend(results[setting])

    pd.read_json(json.dumps(all_results)).to_csv(os.path.join(output_path, 'all_results.csv'),
                                                 index=False, sep=";")

def write_rankings(output_path, rankings):
    Path(output_path).mkdir(parents=True, exist_ok=True)
    all_rankings = []
    for setting in rankings:
        write_json_line_by_line(rankings[setting], os.path.join(output_path, setting + '.csv'))
        all_rankings.extend(rankings[setting])
    write_json_line_by_line(all_rankings, os.path.join(output_path, 'all_rankings.csv'))

# Cell
def convert_results_format(sim_path, settings_path, out_path, models):
    similarities_path = sim_path
    # output_results = out_path/"user_results_weighted_all"
    # output_rankings = out_path/"user_rankings_weighted_all"
    techniques = ["weighted_lcs", "bovw", "lcs", "bovw_lcs", "bovw_weighted_lcs"]
    systems_allowed = []

    settings_path = settings_path
    settings = load_settings(settings_path)

    all_results = {}
    all_rankings = {}

    for setting in settings:
        all_results[setting] = []
        all_rankings[setting] = []

    for model in models:
        print (os.path.join(similarities_path, model))
        sim_files = find_file("rankings_*.pkl", os.path.join(similarities_path, model))
        print (sim_files)
        for sim_file in sim_files:
            file_name = ntpath.basename(sim_file).split(".")[0]
            file_tokens = file_name.split("_")

            vwords = file_tokens[3]
            frames_per_sec = file_tokens[4]
            ftk = file_tokens[5]

            model_similarities = pickle.load(open(sim_file, 'rb'))

            for technique in techniques:
                print(model_similarities.keys())
                similarities = model_similarities[technique]
                configuration = {
                    "model": model,
                    "vwords": vwords,
                    "fps": frames_per_sec,
                    "ftk": ftk,
                    "technique": technique
                }

                print("Running config: ", configuration)

                results, rankings = run_settings(settings, similarities, configuration, systems_allowed)

                for setting in settings:
                    all_results[setting].extend(results[setting])
                    all_rankings[setting].extend(rankings[setting])

    print("Writing results and rankings")

    write_results(out_path, all_results)
    write_rankings(out_path, all_rankings)

    print("done")


# Cell
def get_info_to_ranking_results(ranking, ranking_results, run, dl_model, ir_model, weight_str, setting):
    new_model = dl_model[0] + "-" + ir_model[0]
    new_vwords = dl_model[1]
    new_fps = dl_model[2] + "-" + ir_model[1] + "ftk"
    new_technique = dl_model[3] + "-" + ir_model[2]
    new_config = "({},{})".format("-".join(dl_model), "-".join(ir_model))
    new_config_weight = "({},{},{})".format(weight_str, "-".join(dl_model), "-".join(ir_model))
    config = {
        "model": new_model,
        "vwords": new_vwords,
        "fps": new_fps,
        "technique": new_technique,
        "weight": weight_str,
        "model_config": new_config,
        "model_config_weight": new_config_weight
    }

    query = run["query"]
    query_tokens = query.split("-")

    ranking_results["setting"] = setting
    ranking_results["app"] = query_tokens[0]
    ranking_results["run_id"] = run["run_id"]
    ranking_results.update(config)

    ranking_info = {"run_id": run["run_id"], "query": query, "ranking": ranking}
    ranking_info.update(config)

    return ranking_info, ranking_results

# Cell
def tango_combined(
    out_path, dl_rankings_path, ir_rankings_path,
    settings_path, dl_models, ir_models
):
    # all_data
    results_out_path = out_path/"tango_comb_results"
    rankings_out_path = out_path/"tango_comb_rankings"

    Path(results_out_path).mkdir(parents=True, exist_ok=True)
    Path(rankings_out_path).mkdir(parents=True, exist_ok=True)

    # read data
    settings = load_settings(settings_path)

    dl_rankings = read_json_line_by_line(dl_rankings_path)
    dl_rankings_by_config = group_dict(dl_rankings, lambda rec: (rec['model'], rec['vwords'], rec['ftk'],
                                                                    rec['technique'],))

    ir_rankings = read_json(ir_rankings_path)
    ir_rankings_by_config = group_dict(ir_rankings, lambda rec: (rec['model'], rec['fps'] + "ftk",
                                                                    rec['technique'],))


    ir_model_apps_for_comb = {
        "1ftk-all_text": ['APOD', 'DROID', 'GNU', 'GROW'],
        "5ftk-all_text": ['APOD', 'DROID', 'GNU', 'GROW'],
        "5ftk-unique_frames": ['APOD', 'DROID', 'GROW'],
        "5ftk-unique_words": ['APOD', 'GROW'],
    }

    settings_to_run = ["setting2"]

    dl_models = list(filter(lambda rec: "-".join([rec[0], rec[1], rec[2], rec[3]]) in dl_models,
                            dl_rankings_by_config.keys()))
    ir_models = list(filter(lambda rec: "-".join([rec[0], rec[1], rec[2]]) in ir_models,
                            ir_rankings_by_config.keys()))
    # run combinations

    start_time = time.time()

    all_new_rankings = []
    all_new_results = []
    for dl_model in dl_models:
        dl_mod_rankings = group_dict(dl_rankings_by_config[dl_model], lambda rec: rec["setting"])
        for ir_model in ir_models:
            ir_mod_rankings = group_dict(ir_rankings_by_config[ir_model], lambda rec: rec["setting"])

            print(dl_model, ir_model)

            app_for_comb = ir_model_apps_for_comb["-".join([ir_model[1], ir_model[2]])]

            for setting in settings_to_run:
                dl_runs = group_dict(dl_mod_rankings[setting], lambda rec: rec["run_id"])
                ir_runs = group_dict(ir_mod_rankings[setting], lambda rec: rec["runId"])

                setting_runs = settings[setting]

                for run in setting_runs:

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
                                                                                    str(weight), setting)

                        all_new_results.append(ranking_results)
                        all_new_rankings.append(ranking_info)

                # -------------------------------------------------------------

                # rankings of approach based vocabulary agreement (e.g., 0.2-0: 0.2 weight for all apps except TIME,
                # TOK, and 0 weight for TIME and TOK)
                for run in setting_runs:

                    run_id = run["run_id"]

                    ir_run_ranking = ir_runs[str(run_id)][0]["ranking"]
                    ir_run_ranking = dict(
                        zip((rec["docName"] for rec in ir_run_ranking), (rec for rec in ir_run_ranking)))
                    dl_run_ranking = dl_runs[run_id][0]["ranking"]

                    query = run["query"]
                    query_tokens = query.split("-")

                    app = query_tokens[0]

                    for base_weight in np.arange(0.1, 1.1, 0.1):

                        best_weights_name = f'{base_weight:0.1f}' + "-0"

                        weight = 0
                        if app in app_for_comb:
                            weight = base_weight

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
                                                                                    best_weights_name, setting)

                        all_new_results.append(ranking_results)
                        all_new_rankings.append(ranking_info)

    print("--- %s seconds ---" % (time.time() - start_time))

    print("Writing data")

    pd.read_json(json.dumps(all_new_results)).to_csv(os.path.join(results_out_path, 'all_results.csv'),
                                                     index=False, sep=";")
    write_json_line_by_line(all_new_rankings, os.path.join(rankings_out_path, 'all_rankings.json'))