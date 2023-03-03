import prep
import utils
import eval
import combo
import approach
import features
import json
import subprocess

import argparse
import time
import ntpath 
import os
import pickle
import logging
import trocr
import vision_transformer as vits
import pandas as pd
import tensorflow as tf
import tensorflow.compat.v1 as tf

from pathlib import Path
from collections import OrderedDict


def get_avg_vision_results(out_path):
    new_ranking = {}
    new_results, new_rankings, dl_runs = [],[], []
    setting_path = out_path / 'evaluation_settings'
    sample_path = out_path / 'vision' / 'sample'
    for i in range (args.n_cbs):
        sample = str(sample_path) + str(i+1) + '/all_rankings.csv'
        dl_rankings = utils.read_json_line_by_line(sample)
        dl_rankings_by_config = utils.group_dict(dl_rankings, lambda rec: rec['technique'])[('bovw')]	
        dl_runs.append(utils.group_dict(dl_rankings_by_config, lambda rec: rec["run_id"]))

    settings = utils.read_json_line_by_line(str(setting_path) + '/setting2.json')
    config = {
        "model": 'Dino',
        "vwords": '1000vw',
        "ftk": '5ftk',
        "technique": 'bovw',
        "setting": 'setting2'
    }

    for run in settings:
        new_ranking = {}
        id = run['run_id']
        rank_1, rank_2 = dl_runs[0][id][0]['ranking'], dl_runs[1][id][0]['ranking']
        rank_3, rank_4 = dl_runs[2][id][0]['ranking'], dl_runs[3][id][0]['ranking']
        for doc in rank_1:
            new_ranking[doc] = (rank_1[doc] + rank_2[doc] + rank_3[doc] + rank_4[doc]) / 4
        ranking = OrderedDict(sorted(new_ranking.items(), key=lambda t: t[1], reverse=True))
        ranking_results = eval.evaluate_ranking(ranking, run["gnd_trh"])

        ranking_results["run_id"] = run["run_id"]
        ranking_results["app"] = run['query'].split('-')[0]
        new_results.append(ranking_results)
        ranking_info = {"run_id": run["run_id"], "query": run['query'], "ranking": ranking}
        ranking_info.update(config)
        new_rankings.append(ranking_info)

    avg_path = out_path / 'vision' / 'avg'
    Path(avg_path).mkdir(parents=True, exist_ok=True)
    pd.read_json(json.dumps(new_results)).to_csv(os.path.join(avg_path, 'average_results.csv'),
                                                        index=False, sep=";")
    utils.write_json_line_by_line(new_rankings, os.path.join(avg_path, 'average_rankings.json'))


def get_vis_results (vid_ds, model_path, out_path):
    ckpt_path = model_path / 'dino_rico_ckpt' / 'checkpoint.pth'
    settings_path = out_path / 'evaluation_settings'

    dino = vits.__dict__["vit_base"](args.patch_size, num_classes=0)
    dino.eval()
    utils.load_pretrained_weights(dino, ckpt_path, "teacher", "vit_base", args.patch_size)
    model = features.DinoExtractor (dino)

    sim_func = vits.frame_sim
    FPS, ftk, vw = args.FPS, args.ftk, args.vw
    for i in range (args.n_cbs):
        sample = 'sample'+str(i+1)
        results_path = out_path / 'vision' / sample
        cb_path = model_path / 'codebooks' / sample / 'codebook_dino_1000vw.model'
        codebook = pickle.load(open(cb_path, "rb"))
        vid_ds_features = approach.gen_extracted_features(vid_ds, model, FPS, ftk)

        df, bovw_vid_ds_sims = approach.gen_bovw_similarity(
            vid_ds, vid_ds_features, codebook, vw, ftk
        )
        lcs_vid_ds_sims = approach.gen_lcs_similarity(
            vid_ds, vid_ds_features, sim_func, codebook, df, vw, ftk
        )
        rankings = approach.approach(
            vid_ds, vid_ds_features, bovw_vid_ds_sims, lcs_vid_ds_sims, sim_func,
            codebook, df, vw, fps = FPS, ftk = ftk,
        )       

        id_name = f"user_{args.n_imgs}n_{vw}vw_{FPS}fps_{ftk}ftk"
        results_path.mkdir(parents=True, exist_ok=True)
        with open(results_path / f"rankings_{id_name}.pkl", "wb") as f:
            pickle.dump(rankings, f, protocol=pickle.HIGHEST_PROTOCOL)
        combo.convert_results_format(results_path, settings_path, "Dino")
        get_avg_vision_results (out_path)


def get_txt_results (txt_path, setting_path):
    engine_path = txt_path / 'Lucene'
    if not (engine_path / 'extracted_txt' / f"text_{args.ftk}").exists():
        trocr.get_text(vid_ds, txt_path, args.ftk)
    subprocess.check_output(
        ["sh", "build_run.sh", str('./extracted_txt'), str(setting_path)],
        cwd=str(engine_path),
    )



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--repro_path", default='/scratch/projects/yyan/DINO/JANUS_reproduction_package', type=str, required=False,
                        help="The artifact path.")
    parser.add_argument("--ftk", default=5, type=int, required=False,
                        help="Frame rate")
    parser.add_argument("--FPS", default=30, type=int, required=False,
                        help="Frame rate")
    parser.add_argument("--n_cbs", default=4, type=int, required=False,
                        help="# of codebooks")
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--vw', default=1_000, type=int, help='Patch resolution of the model.')
    parser.add_argument('--n_imgs', default=15_000, type=int, help='# images for generating codebooks')
    
    #print arguments
    args = parser.parse_args()

    art_path = Path(args.repro_path) / 'artifacts'
    out_path = Path(args.repro_path) / 'outputs'

    vid_ds = prep.VideoDataset.from_path(
        art_path/"videos", fr = None
    ).label_from_paths()

    # get_vis_results (vid_ds, art_path / 'models' / 'vision', out_path)
    get_txt_results (art_path / 'models' / 'text', out_path / 'evaluation_settings')
    
    ir_ranking_path = out_path / 'text' / 'all_rankings.json'
    dl_ranking_path = out_path / 'vision' / 'avg' / 'average_rankings.json'
    settings_path = out_path / 'evaluation_settings'
    combo.combined(out_path / 'combined', dl_ranking_path, ir_ranking_path, settings_path, "Dino-1000vw-5ftk-bovw", "east_trocr-5ftk-all_text")
