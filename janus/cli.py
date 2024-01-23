import logging
import prep
import utils
import eval
import combo
import approach
import features

import json
import subprocess
import argparse
import os
import pickle
import vision_transformer as vits
import pandas as pd
import numpy as np
from pathlib import Path
from collections import OrderedDict, defaultdict

import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_avg_vision_results(setting_path, out_path):
    new_results, new_rankings = [],[]
    sample_path = out_path / 'sample'
    techniques = ["weighted_lcs", "bovw", "bovw_weighted_lcs"]
    settings = utils.read_json_line_by_line(str(setting_path / 'setting.json'))
    model_name = str(out_path).split('/')[-1]

    for technique in techniques:
        dl_runs = []
        for i in range (args.n_cbs):
            sample = str(sample_path) + str(i+1) + '/all_rankings.csv'
            dl_rankings = utils.read_json_line_by_line(sample)
            dl_rankings_by_config = utils.group_dict(dl_rankings, lambda rec: rec['technique'])[(technique)]	
            dl_runs.append(utils.group_dict(dl_rankings_by_config, lambda rec: rec["run_id"]))

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
            ranking_results["technique"] = technique
            ranking_results["model"] = model_name
            new_results.append(ranking_results)
            ranking_info = {"run_id": run["run_id"], "query": run['query'], "ranking": ranking}
            ranking_info["technique"] = technique
            ranking_info["model"] = model_name

            new_rankings.append(ranking_info)

    avg_path = out_path / 'avg'
    Path(avg_path).mkdir(parents=True, exist_ok=True)
    pd.read_json(json.dumps(new_results)).to_csv(os.path.join(avg_path, 'all_results.csv'),
                                                        index=False, sep=";")
    utils.write_json_line_by_line(new_rankings, os.path.join(avg_path, 'all_rankings.json'))


def get_vis_results (vid_ds, model_arch, patch_size, art_path, out_path):
    logging.info("-" * 20)
    logging.info("Computing Visual JANUS results")

    model_name = model_arch + str(patch_size)   
    settings_path = art_path / 'evaluation_settings'
    art_path = art_path / 'models' / 'vision'
    ckpt_path = art_path / 'dino_rico_ckpt' /  (model_name + '.pth')
    cb_path = art_path / 'codebooks' / model_name
    out_path = out_path / 'vision' / model_name
    
    sim_func = vits.frame_sim
    FPS, ftk, vw = args.FPS, args.ftk, args.vw

    dino = vits.__dict__[model_arch](args.patch_size, num_classes=0)
    dino.eval()
    utils.load_pretrained_weights(dino, ckpt_path, "teacher", model_arch, args.patch_size)
    model = features.DinoExtractor (dino, model_arch)

    logging.info(f"Extracting visual features based on Dino ({str(model_name)}).")
    vid_ds_features = approach.gen_extracted_features(vid_ds, model, FPS, ftk)
 
    for i in range (args.n_cbs):
        sample = 'sample'+str(i+1)
        results_path = out_path / sample
        Path(results_path).mkdir(parents=True, exist_ok=True)       
        if not (results_path / "rankings.pkl").exists():
            logging.info(f"Extracting video geatures and computing similarity scores based on {str(i)} codebook.")
            cb_path_s = cb_path / sample / 'codebook_dino_1000vw.model'
            codebook = pickle.load(open(cb_path_s, "rb"))
            bovw_vid_ds_sims = approach.gen_bovw_similarity(
                vid_ds, vid_ds_features, codebook, vw
            )
            lcs_vid_ds_sims = approach.gen_lcs_similarity(
                vid_ds, vid_ds_features, sim_func
            )
            rankings = approach.approach(
                vid_ds, bovw_vid_ds_sims, lcs_vid_ds_sims
            )       

            results_path.mkdir(parents=True, exist_ok=True)
            with open(results_path / "rankings.pkl", "wb") as f:
                pickle.dump(rankings, f, protocol=pickle.HIGHEST_PROTOCOL)
        combo.convert_results_format(results_path, settings_path, model_name)

    logging.info(f"Averaging the results across four codebooks.")
    get_avg_vision_results (settings_path, out_path)

    logging.info("-" * 20)


def get_txt_results (lucene, art_path, out_path):
    logging.info("Computing Textual JANUS results")

    txt_path = art_path / 'models' / 'text'
    engine_path = txt_path / 'Lucene'
    output_txt_path = engine_path / 'extracted_txt' / f"text_{args.ftk}"
    if not output_txt_path.exists():
        logging.info("Extracting Video Text")
        # trocr.get_text(vid_ds, txt_path, args.ftk, args.cpu_mode)
    if lucene:
        logging.info("Generating textual JANUS results based on Lucene with the bag-of-words strategy")
        subprocess.check_output(
            ["sh", "build_run.sh", 'extracted_txt', str(art_path / 'evaluation_settings')],
            cwd=str(engine_path),
        )
    else:
        logging.info("Generating textual JANUS results based on Sklearn with the bag-of-words + weighted_lcs strategy")
        out_seq_path = out_path / 'text'/ 'sklearn_seq'
        Path(out_seq_path).mkdir(parents=True, exist_ok=True)
        if not (out_seq_path / "rankings.pkl").exists(): 
            all_text_vec = approach.get_text_features (output_txt_path, 'all_text')
            all_text_vec_frame = approach.get_text_frame_features (output_txt_path)
            text_sim = approach.gen_text_similarity(vid_ds, all_text_vec)
            lcs_vid_ds_sims = approach.gen_text_lcs_similarity_multi(vid_ds, all_text_vec_frame)         
            rankings = approach.approach(vid_ds, text_sim, lcs_vid_ds_sims)     
            with open(out_seq_path / f"rankings.pkl", "wb") as f:
                pickle.dump(rankings, f, protocol=pickle.HIGHEST_PROTOCOL)

        combo.convert_results_format(out_seq_path, art_path / 'evaluation_settings', "east_trocr")

    logging.info("-" * 20)
    

def _print_performance(model, mRR, mAP, mean_R):
    print(
        f"""\
        Model: {model}
        Overall mRR: {mRR}
        Overall mAP: {mAP}
        Overall Mean Rank: {mean_R}
        """
    )


def _output_performance(out_path, results_type):

    print("Reproduction Results")
    all_results = pd.read_csv(out_path / 'all_results.csv', sep=";")

    if results_type == 'vision':
        print("Visual JANUS")
        techniques = ["bovw", "bovw_weighted_lcs"]       
        for technique in techniques:
            vision_results = all_results[all_results["technique"] == technique]
            info = 'Visual Information only'
            if technique == 'bovw_weighted_lcs':
                info = 'Visual + Seqential Information'
            _print_performance(
            f"{info}, dino_{all_results.iloc[0]['model']},{technique}", np.mean(vision_results['recip_rank'].values),
            np.mean(vision_results['avg_precision'].values), np.mean(vision_results['first_rank'].values)
            )
    elif results_type == 'text':
        print("Textual JANUS")
        if str(out_path).split('/')[-1] == 'Lucene_bow':
            text_results = all_results[all_results["technique"] == 'bovw']
            _print_performance(
                f"Textual Information Only, {all_results.iloc[0]['model']},bovw", np.mean(text_results['recip_rank'].values),
                np.mean(text_results['avg_precision'].values), np.mean(text_results['first_rank'].values)
                )
        else:    
            text_results = all_results[all_results["technique"] == "bovw_weighted_lcs"]
            _print_performance(
            f"Textual + Sequential Information, {all_results.iloc[0]['model']}, bovw-weighted_lcs", np.mean(text_results['recip_rank'].values),
            np.mean(text_results['avg_precision'].values), np.mean(text_results['first_rank'].values)
            )

    else:
        print("Combined JANUS")
        if results_type == 'vision-text':
            technique = 'bovw-bow'
            weight = 0.1
        else:
            technique = 'bovw_weighted_lcs-bow_weighted_lcs'
            weight = 0.4
        comb_results = all_results[
            all_results["technique"] == technique
        ][all_results["weight"] == weight]
        _print_performance(
            f"dino_{all_results.iloc[0]['model']}, {technique},weight={weight}", np.mean(comb_results['recip_rank'].values),
            np.mean(comb_results['avg_precision'].values), np.mean(comb_results['first_rank'].values)
        )
    print("-" * 20)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--repro_path", default='/projects/DINO/JANUS_reproduction_package', type=str, required=False,
                        help="The replication package path")
    parser.add_argument('--arch', default='vit_base', type=str, choices=['vit_small', 'vit_base'],
                        help='Name of visual architecture to fine-tune')
    parser.add_argument('--patch_size', default=16, type=int, choices=[8, 16],
                        help='Patch resolution of the model.')
    parser.add_argument('--results_type', default='vision-text-seq', type=str, choices=['vision', 'text', 'vision-text', 'vision-text-seq'],
                        help='The results_type to generate')
    parser.add_argument('--cpu_mode', action='store_true', 
                        help='whether to use GPU for textual JANUS')
    parser.add_argument('--lucene', default=False, type=bool, 
                        help='whether to use lucene to compute bow similaity for textual JANUS')

    parser.add_argument("--ftk", default=5, type=int, help="Frame rate")
    parser.add_argument("--FPS", default=30, type=int, help="Frame per second")
    parser.add_argument("--n_cbs", default=4, type=int, help="# of codebooks")
    parser.add_argument('--vw', default=1_000, type=int, help='# of visual words in a generated codebook.')
    parser.add_argument('--n_imgs', default=15_000, type=int, 
                        help='# of images for a generating codebook')

    args = parser.parse_args()
    art_path = Path(args.repro_path) / 'artifacts'
    out_path = Path(args.repro_path) / 'outputs'

    if not args.cpu_mode:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    logging.info("Loading Videos")
    vid_ds = prep.VideoDataset.from_path(
        art_path/'videos', fr = None
    ).label_from_paths()

    if args.results_type == 'vision':
        vision_model_name = args.arch + str(args.patch_size)
        get_vis_results (vid_ds, args.arch, args.patch_size, art_path, out_path)
        _output_performance(out_path / 'vision' / vision_model_name / 'avg', args.results_type)

    elif args.results_type == 'text':
        get_txt_results (args.lucene, art_path, out_path)
        if args.lucene:
            _output_performance(out_path / 'text' / 'Lucene_bow', args.results_type)
        else:
            _output_performance(out_path / 'text' / 'sklearn_seq', args.results_type)

    else:
        vision_model_name = args.arch + str(args.patch_size)
        ir_ranking_path, ir_bow_rankings_path = '', ''

        dl_ranking_path = out_path / 'vision' / vision_model_name / 'avg' / 'all_rankings.json'
        if not Path(dl_ranking_path).exists():
            get_vis_results (vid_ds, args.arch, args.patch_size, art_path, out_path)
        
        args.lucene = True if args.results_type == 'vision-text' else False      
        if not args.lucene:
            ir_ranking_path = (out_path / 'text' / 'sklearn_seq' / 'all_rankings.csv')  
            if not Path(ir_ranking_path).exists():
                get_txt_results (args.lucene, art_path, out_path)
        else:
            ir_bow_rankings_path = out_path / 'text' / 'Lucene_bow' / 'all_rankings.json'
            if not Path(ir_bow_rankings_path).exists():
                get_txt_results (args.lucene, art_path, out_path)       

        settings_path = art_path / 'evaluation_settings'
        combo.combined(out_path / 'combined', dl_ranking_path, ir_ranking_path, ir_bow_rankings_path, settings_path, vision_model_name, "east_trocr")
        
        _output_performance(out_path / 'vision' / vision_model_name / 'avg', 'vision')
        if args.lucene:
            _output_performance(out_path / 'text' / 'Lucene_bow', 'text')
        else:
            _output_performance(out_path / 'text' / 'sklearn_seq', 'text')
        _output_performance(out_path / 'combined', args.results_type)

