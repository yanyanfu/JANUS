import os
import re
import cv2
import time
import math
import ntpath
import torch
import requests 
import pytesseract
import argparse

import lanms
import model
import prep
import utils

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf
import locality_aware_nms as nms_locality

from pathlib import Path
from PIL import Image 
from tqdm import tqdm
from icdar import restore_rectangle
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset


tf.app.flags.DEFINE_string('repro_path', '', '')
tf.app.flags.DEFINE_string('arch', '', '')
tf.app.flags.DEFINE_string('patch_size', '', '')
tf.app.flags.DEFINE_string('results_type', '', '')
tf.app.flags.DEFINE_bool('lucene', False, '')


def resize_image(im, max_side_len=2400):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _= im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    resize_h = max(32, resize_h)
    resize_w = max(32, resize_w)
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def detect(score_map, geo_map, timer, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
    # print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start
    # nms part
    start = time.time()
    # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
    timer['nms'] = time.time() - start

    if boxes.shape[0] == 0:
        return None, timer

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes, timer


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]


def get_text(vid_ds, text_path, ftk):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = text_path / 'east_icdar2015_resnet_v1_50_rbox'
    output_dir = text_path / 'Lucene'/ 'extracted_txt'

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed") 
    trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed")
    trocr_model.to(device)

    trocr_model.config.vocab_size = trocr_model.config.decoder.vocab_size
    trocr_model.config.eos_token_id = processor.tokenizer.sep_token_id
    trocr_model.config.max_length = 64
    trocr_model.config.early_stopping = True
    trocr_model.config.no_repeat_ngram_size = 3
    trocr_model.config.length_penalty = 2.0
    trocr_model.config.num_beams = 4

    trocr_model.eval()

    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        f_score, f_geometry = model.model(input_images, is_training=False)
        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(checkpoint_path)
            model_path = os.path.join(checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            saver.restore(sess, model_path)

            video_output_path = os.path.join(output_dir, "text_" + str(ftk))
            Path(video_output_path).mkdir(parents=True, exist_ok=True)

            videos = [vid.vid_path for vid in vid_ds.videos]

            for video_path in videos:
                video_path_obj = Path(video_path)
                file_name = ntpath.basename(video_path).split(".")[0]
                video_name = file_name + "-" + str(video_path_obj.parent.parent.parent.stem)

                out_file = os.path.join(video_output_path, video_name + '.json')
                frame_path = os.path.join(output_dir, "frames_" + str(ftk), video_name)
                Path(frame_path).mkdir(parents=True, exist_ok=True)

                frames = utils.find_file("*.jpeg", frame_path)
                if not frames:
                    utils.extract_frames(video_path_obj, frame_path, ftk)
                frames = utils.find_file("*.jpeg", frame_path)
                
                frames = utils.find_file("*.jpeg", frame_path)

                frames_text = []
                for frame in tqdm(frames):
                    image_frame = cv2.imread(frame)[:, :, ::-1]               
                    im_resized, (ratio_h, ratio_w) = resize_image(image_frame)
                    h, w, _= image_frame.shape

                    timer = {'net': 0, 'restore': 0, 'nms': 0}
                    score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized]})
                    boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)

                    frame_text=[]
                    pixel_values = []
                    if boxes is not None:
                        boxes = boxes[:, :8].reshape((-1, 4, 2))
                        boxes[:, :, 0] /= ratio_w
                        boxes[:, :, 1] /= ratio_h

                        for box in boxes:
                            if np.linalg.norm(box[0] - box[1]) < 80 or np.linalg.norm(box[3]-box[0]) < 40:
                                continue                  
                            #apply padding to each side of the bounding box, respectively
                            dX = (box[2,0] - box[0,0]) * 0.03
                            dY = (box[2,1] - box[0,1]) * 0.04                      
                            startX = max(0, int(box[0,0] - dX))
                            startY = max(0, int (box[0,1] - dY))
                            endX = min(w, int (box[2,0] + (dX * 2)))
                            endY = min(h, int (box[2,1] + (dY * 2)))
                            if startY < endY and startX < endX:
                                try:
                                    roi = image_frame[startY:endY, startX:endX]
                                    roi_pixel_values = processor(roi, return_tensors="pt").pixel_values.squeeze()
                                    pixel_values.append(roi_pixel_values.cpu().numpy())
                                except Exception as e:
                                    pass
                    
                        roi_dataset = TensorDataset(torch.tensor(pixel_values))
                        roi_sampler = SequentialSampler(roi_dataset)
                        roi_dataloader = DataLoader(roi_dataset, sampler = roi_sampler, batch_size = 8, num_workers = 2)
                        with torch.no_grad():
                            for batch in roi_dataloader:
                                ids = trocr_model.generate(batch[0].to(device))
                                text = processor.batch_decode(ids, skip_special_tokens=True)
                                frame_text += text
                        
                        frame_name = ntpath.basename(frame).split(".")[0]
                        record = {"f": frame_name, "txt": '\n\f'.join(frame_text)}
                        frames_text.append(record)

                frames_text = sorted(frames_text, key=lambda t: t["f"])
                utils.write_json_line_by_line(frames_text, out_file)
                print("done: " + video_name)

