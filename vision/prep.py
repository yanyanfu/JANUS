# Cell
import concurrent.futures
import csv
import cv2
import ffmpeg
import json
import ntpath
import numpy
import os
import pprint
import pytesseract
import random

import matplotlib.pyplot as plt
import more_itertools as mit
import pandas as pd

from collections import defaultdict, OrderedDict
from pathlib import Path
from PIL import Image
from DINO.JANUS.vision.utils2 import *
from shutil import copyfile
from tqdm.auto import tqdm

def get_rand_imgs(vid_path, max_msecs, n = 10):
    vid = cv2.VideoCapture(str(vid_path))

    imgs = []
    while len(imgs) < n:
        msec = random.randrange(1_000, max_msecs, 1_000)
        vid.set(cv2.CAP_PROP_POS_MSEC, msec)

        success, img = vid.read()
        if success:
            imgs.append(img)

    return imgs

def vid_from_frames(frames, output = None, fr = 30):
    """Generate video from list of frame paths."""
    if not output: output = frames.parent

    try:
        stream = ffmpeg.input(frames/'%04d.jpg')
        stream = ffmpeg.output(stream, str(output/"gen_vid.mp4"), r = fr)
        out, err = ffmpeg.run(stream)
    except Exception as e:
        print("Error occured:", e)


class Video:
    def __init__(self, vid_path, fr = None, overwrite = False):
        self.vid_path = vid_path
        self.fr = eval(ffmpeg.probe(vid_path)["streams"][0]["avg_frame_rate"])
        if fr is not None:
            self.fr = fr
            self.vid_path = self._fix_framerate(vid_path, fr, overwrite)

        self.video = cv2.VideoCapture(str(self.vid_path))

    def show_frame(self, i):
        plt.imshow(self[i])
        plt.show()

    def _fix_framerate(self, vid_path, fr, overwrite):
        """
            Fixes each video in the list of video paths to a certain frame rate.
        """
        output_path = str(vid_path) if overwrite else str(vid_path.parent/f'{vid_path.stem}_fixed_{fr}.mp4')
        stream = ffmpeg.input(vid_path)
        stream = ffmpeg.output(stream, output_path, r = fr)
        stream = ffmpeg.overwrite_output(stream)
        out, err = ffmpeg.run(stream)

        return Path(output_path)

    def __len__(self):
        return int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

    def __getitem__(self, i):
        if i >= len(self) or i < 0:
            raise Exception(f'Frame index is not in the proper range (0, {len(self) - 1}).')
        self.video.set(cv2.CAP_PROP_POS_FRAMES, i)
        suc, frame = self.video.read()
        if not suc: return None
        return Image.fromarray(frame)


class VideoDataset:
    def __init__(self, videos):
        self.videos = videos
        self.labels = None
        self.data = None

    def label_from_paths(self):
        self.labels = defaultdict(
            lambda: defaultdict(dict)
        )
        for vid in self.videos:
            self.labels[vid.vid_path.parent.parent.name][vid.vid_path.parent.name][vid.vid_path.parent.parent.parent.name] = vid

        return self

    def get_labels(self):
        return list(self.labels.keys())

    @staticmethod
    def from_path(path, extract_frames = False, fr = None, overwrite = False):
        videos = []
        fixed_vid_paths = sorted(path.rglob(f"*fixed_{fr}.mp4"))
        if len(fixed_vid_paths) > 0 and fr is not None:
            for vid_path in fixed_vid_paths:
                videos.append(Video(vid_path, overwrite = overwrite))
        else:
            vid_paths = list(filter(lambda x: "fixed" not in str(x), sorted(path.rglob('*.mp4'))))
            for vid_path in vid_paths:
                videos.append(Video(vid_path, fr = fr, overwrite = overwrite))

        return VideoDataset(videos)

    def __getitem__(self, label):
        return self.labels[label]


def get_rico_imgs(path, n = None):
    rico_path = path/'rico-images/data'
    img_paths = sorted(rico_path.glob('*.jpg'))
    if n == None: n = len(img_paths)

    return [Image.open(img) for img in random.sample(img_paths, n)]


def get_all_texts(vid_ds, out_path, fps):
    Path(out_path).mkdir(parents=True, exist_ok=True)

    video_output_path = os.path.join(out_path, "text_" + str(fps))
    Path(video_output_path).mkdir(parents=True, exist_ok=True)

    videos = [vid.vid_path for vid in vid_ds.videos]
    for video_path in videos:
        video_path_obj = Path(video_path)

        file_name = ntpath.basename(video_path).split(".")[0]
        video_name = file_name + "-" + str(video_path_obj.parent.parent.parent.stem)

        frame_path = os.path.join(out_path, "frames_" + str(fps), video_name)
        Path(frame_path).mkdir(parents=True, exist_ok=True)

        frames = find_file("*.jpeg", frame_path)
        if not frames:
            extract_frames(video_path_obj, frame_path, fps)
        frames = find_file("*.jpeg", frame_path)

        frames_text = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            futures = []
            for frame in frames:
                futures.append(executor.submit(process_frame, frame))
            for future in concurrent.futures.as_completed(futures):
                frames_text.append(future.result())

        frames_text = sorted(frames_text, key=lambda t: t["f"])

        video_name = video_name.replace("_fixed_30", "")
        out_file = os.path.join(video_output_path, video_name + '.json')
        write_json_line_by_line(frames_text, out_file)

        print("done: " + video_name)