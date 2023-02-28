import prep
import utils
import model
import eval
import combo
import approach
import features

import argparse
import time
import ntpath 
import os
import pickle
import logging
import vision_transformer as vits
from pathlib import Path



def get_vis_results (vid_ds, model_path, out_path):
    ckpt_path = model_path / 'dino_rico_ckpt' / 'checkpoint.pth'
    settings_path = out_path / 'evaluation_settings'
    results_path = out_path / 'vision'

    dino = vits.__dict__["vit_base"](args.patch_size, num_classes=0)
    dino.eval()
    utils.load_pretrained_weights(dino, ckpt_path, "teacher", "vit_base", args.patch_size)

    sim_func = vits.frame_sim
    FPS, ftk, vw = args.FPS, args.ftk, args.vw
    for i in range (len(args.n_cbs)):
        cb_path = model_path / 'codebooks' / 'sample'+str(i) / 'codebook_dino_1000vw.model'
        codebook = pickle.load(open(cb_path, "rb"))
        vid_ds_features = approach.gen_extracted_features(vid_ds, dino, FPS, ftk)

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
        evaluation_metrics = {}
        for k, v in rankings.items():
            evaluation_metrics[k] = eval.evaluate(rankings[k])        

        id_name = f"user_{args.n_imgs}n_{vw}vw_{FPS}fps_{ftk}ftk"
        results_path.mkdir(parents=True, exist_ok=True)
        with open(results_path / 'sample'+str(i) / f"rankings_{id_name}.pkl", "wb") as f:
            pickle.dump(rankings, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(results_path / 'sample'+str(i)/ f"evaluation_metrics_{id_name}.pkl", 'wb') as f:
            pickle.dump(evaluation_metrics, f, protocol=pickle.HIGHEST_PROTOCOL)
        combo.convert_results_format(results_path / 'sample'+str(i), settings_path, out_path, ["Dino"])




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--repro_path", default='../../JANUS_reproduction_package/', type=str, required=False,
                        help="The artifact path.")
    parser.add_argument("--ftk", default=5, type=int, required=False,
                        help="Frame rate")
    parser.add_argument("--FPS", default=30, type=int, required=False,
                        help="Frame rate")
    parser.add_argument("--n_cbs", default=4, type=int, required=False,
                        help="# of codebooks")
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--vw', default=1_000, type=int, help='Patch resolution of the model.')
    parser.add_argument('--n_imgs', default=1000, type=int, help='Patch resolution of the model.')
    
    #print arguments
    args = parser.parse_args()

    art_path = Path(args.repro_path) / 'artifacts'
    out_path = Path(args.repro_path) / 'outputs'

    vid_ds = prep.VideoDataset.from_path(
        art_path/"videos", fr = args.FPS
    ).label_from_paths()

    get_vis_results (vid_ds, art_path / 'models' / 'vision', out_path / 'vision')
