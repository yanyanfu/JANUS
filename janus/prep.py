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
import time
import threading
import multiprocessing

import matplotlib.pyplot as plt
import more_itertools as mit
import pandas as pd
import numpy as np

from collections import defaultdict, OrderedDict
from pathlib import Path
from PIL import Image
from utils import *
from shutil import copyfile

try:
    import queue
except ImportError:
    import Queue as queue


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
                # if 'FCS' in str(vid_path):
                print(vid_path)
                videos.append(Video(vid_path, fr = fr, overwrite = overwrite))

        return VideoDataset(videos)

    def __getitem__(self, label):
        return self.labels[label]


def get_rico_imgs(path, n = None):
    rico_path = path/'rico-images/data'
    img_paths = sorted(rico_path.glob('*.jpg'))
    if n == None: n = len(img_paths)

    return [Image.open(img) for img in random.sample(img_paths, n)]



'''
this class is modified from keras implemention of data process multi-threading,
see https://github.com/fchollet/keras/blob/master/keras/utils/data_utils.py
'''
class GeneratorEnqueuer():
    """Builds a queue out of a data generator.

    Used in `fit_generator`, `evaluate_generator`, `predict_generator`.

    # Arguments
        generator: a generator function which endlessly yields data
        use_multiprocessing: use multiprocessing if True, otherwise threading
        wait_time: time to sleep in-between calls to `put()`
        random_seed: Initial seed for workers,
            will be incremented by one for each workers.
    """

    def __init__(self, generator,
                 use_multiprocessing=False,
                 wait_time=0.05,
                 random_seed=None):
        self.wait_time = wait_time
        self._generator = generator
        self._use_multiprocessing = use_multiprocessing
        self._threads = []
        self._stop_event = None
        self.queue = None
        self.random_seed = random_seed

    def start(self, workers=1, max_queue_size=10):
        """Kicks off threads which add data from the generator into the queue.

        # Arguments
            workers: number of worker threads
            max_queue_size: queue size
                (when full, threads could block on `put()`)
        """

        def data_generator_task():
            while not self._stop_event.is_set():
                try:
                    if self._use_multiprocessing or self.queue.qsize() < max_queue_size:
                        generator_output = next(self._generator)
                        self.queue.put(generator_output)
                    else:
                        time.sleep(self.wait_time)
                except Exception:
                    self._stop_event.set()
                    raise

        try:
            if self._use_multiprocessing:
                self.queue = multiprocessing.Queue(maxsize=max_queue_size)
                self._stop_event = multiprocessing.Event()
            else:
                self.queue = queue.Queue()
                self._stop_event = threading.Event()

            for _ in range(workers):
                if self._use_multiprocessing:
                    # Reset random seed else all children processes
                    # share the same seed
                    np.random.seed(self.random_seed)
                    thread = multiprocessing.Process(target=data_generator_task)
                    thread.daemon = True
                    if self.random_seed is not None:
                        self.random_seed += 1
                else:
                    thread = threading.Thread(target=data_generator_task)
                self._threads.append(thread)
                thread.start()
        except:
            self.stop()
            raise

    def is_running(self):
        return self._stop_event is not None and not self._stop_event.is_set()

    def stop(self, timeout=None):
        """Stops running threads and wait for them to exit, if necessary.

        Should be called by the same thread which called `start()`.

        # Arguments
            timeout: maximum time to wait on `thread.join()`.
        """
        if self.is_running():
            self._stop_event.set()

        for thread in self._threads:
            if thread.is_alive():
                if self._use_multiprocessing:
                    thread.terminate()
                else:
                    thread.join(timeout)

        if self._use_multiprocessing:
            if self.queue is not None:
                self.queue.close()

        self._threads = []
        self._stop_event = None
        self.queue = None

    def get(self):
        """Creates a generator to extract data from the queue.

        Skip the data if it is `None`.

        # Returns
            A generator
        """
        while self.is_running():
            if not self.queue.empty():
                inputs = self.queue.get()
                if inputs is not None:
                    yield inputs
            else:
                time.sleep(self.wait_time)