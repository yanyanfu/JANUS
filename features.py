# Cell
import cv2
import pickle
# import ffmpeg
import random
import time
import torch
import os

import numpy as np

from abc import ABC, abstractmethod
from pathlib import Path
from PIL import Image
from prep import *
from tqdm.auto import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
import torch

# Cell
class Extractor(ABC):

    def __init__(self, extractor):
        self.extractor = extractor
        super().__init__()

    @abstractmethod
    def extract(self, img):
        pass

# Cell
class SIFTExtractor(Extractor):

    '''Exposed SIFTExtractor class used for retrieving features.'''

    def extract(self, img):
        '''Given an image, extract features using SIFT. Returns the feature vector.'''
        img = np.array(img)
        _, features = self.extractor.detectAndCompute(img, None)
        return features


class CNNExtractor(Extractor):

    '''Exposed CNNExtractor class used for retrieving features.'''

    def extract(self, img):
        '''Given an image, extract features from the layers of a CNN. Returns the feature vector.'''

        return self.extractor.getFeatures(img)

# Cell
from torchvision import transforms

def imagenet_normalize_transform():
    return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def get_transforms(size=224):
    return transforms.Compose([
        transforms.Resize(size=(size, size)),
        transforms.ToTensor(),
        imagenet_normalize_transform()
    ])


def get_transforms_dino(size=224):
    return transforms.Compose([
        transforms.Resize(size=(size, size), interpolation=3), 
        transforms.ToTensor(),
        imagenet_normalize_transform()
    ])


class SimCLRExtractor(Extractor):
    
    '''Exposed CNNExtractor class used for retrieving features.'''
    def __init__(self, extractor):
        super().__init__(extractor)
        self.transforms = get_transforms()

    def extract(self, img):
        '''Given an image, extract features from the layers of a CNN. Returns the feature vector.'''
        img = self.transforms(img).float()
        img = img.unsqueeze(0)
        return self.extractor(img).detach().numpy()


class DinoExtractor(Extractor):
    
    '''Exposed CNNExtractor class used for retrieving features.'''
    def __init__(self, extractor):
        super().__init__(extractor)
        self.transforms = get_transforms_dino()

    def extract_CLS(self, img):
        '''Given an image, extract features from the layers of a CNN. Returns the feature vector.'''
        img = self.transforms(img).float()
        intermediate_output = self.extractor.get_intermediate_layers(img.unsqueeze(0), n=1)
        features = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
        features = torch.cat((features.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
        features = features.reshape(features.shape[0], -1).detach().numpy()
        return features

    def extract_like_copy_detection(self, img):
        '''Given an image, extract features from the layers of a CNN. Returns the feature vector.'''
        img = self.transforms(img).float()
        samples = img.unsqueeze(0)
        feats = self.extractor.get_intermediate_layers(samples.unsqueeze(0), n=1)[0].clone()
        cls_output_token = feats[:, 0, :]  #  [CLS] token
        b, h, w, d = len(samples), int(samples.shape[-2] / model.patch_embed.patch_size), int(samples.shape[-1] / model.patch_embed.patch_size), feats.shape[-1]
        feats = feats[:, 1:, :].reshape(b, h, w, d)
        feats = feats.clamp(min=1e-6).permute(0, 3, 1, 2)
        feats = nn.functional.avg_pool2d(feats.pow(4), (h, w)).pow(1. / 4).reshape(b, -1)
        # concatenate [CLS] token and GeM pooled patch tokens
        feats = torch.cat((cls_output_token, feats), dim=1)
        
        return feats


def gen_vcodebook(path, img_paths, model_name, extractor, vwords):
    """
        Constructs a visual codebook based on the given images.
        You can change vwords to increase the vocabulary of the codebook.
    """
    fname = path/f'models/features/{model_name}/{len(img_paths)}n_features.pkl'
    features_list = []
    feature_time = 0
    if not fname.is_file():
        feature_start = time.time()
        for img in tqdm(img_paths):
            features = extractor.extract(Image.open(img))
            if features is None: continue
            features_list.extend(features)

        features_list = np.asarray(features_list)
        pickle.dump(features_list, open(fname, 'wb'), protocol=4)
        feature_end = time.time()
        feature_time = feature_end - feature_start
        fname = path/f'models/features/{model_name}/{len(img_paths)}n_features_elapsed_time.txt'
        with open(fname, 'w') as f:
            f.write(f'{feature_time}')
    else:
        features_list = pickle.load(open(fname, 'rb'))
        with open(fname, 'r') as f:
            feature_time = [float(x) for x in f][0]
    codebooks = []
    for vw in vwords:
        cb_start = time.time()
        codebook = KMeans(n_clusters = vw)
        codebook.fit(features_list)
        cb_end = time.time()
        codebooks.append((cb_end - cb_start + feature_time, codebook))

    return codebooks


def gen_codebooks(path, models, vwords, samples = 15_000):

    rico_path = Path("/scratch/projects/yyan/DINO/data/version_sort_img/sample_3")
    # img_paths = random.sample(list(rico_path.glob('*.jpg')), samples)
    img_paths = list(rico_path.glob('*.jpg'))
    codebooks = gen_vcodebook(path, img_paths, "dino", models, vwords)
    for (cb_time, codebook), vw in zip(codebooks, vwords):
        fname = Path('./quality_on_imagenet_16/dino_vit_backbone/sample3')/f'codebook_dino_{vw}vw.model'
        pickle.dump(codebook, open(fname, 'wb'))


def get_df(imgs, extractor, codebook, vwords):
    """Generates the document frequency for the visual words"""
    arr = []
    for img in imgs:
        features = extractor.extract(img)
        vw = codebook.predict(features)
        arr.extend(vw)
    arr = np.asarray(arr)

    return np.histogram(arr, bins = range(vwords + 1))


def get_bovw(vid_path, extractor, codebook, vwords, n = None):
    """Generates the bag of visual words (bovw) for an entire video."""
    vid = cv2.VideoCapture(str(vid_path))
    if n is None: n = vid.get(cv2.CAP_PROP_FRAME_COUNT)

    # checks whether frames were extracted
    success = 1
    bovw = np.array([])
    for i in progress_bar(range(n)):
        # vid object calls read
        # function extract frames
        success, img = vid.read()
        if success:
            features = extractor.extract(img)
            vw = codebook.predict(features)
            bovw = np.concatenate((bovw, vw))

    hist = np.histogram(bovw, bins = range(vwords + 1))[0]
    return hist, bovw

@torch.no_grad()
def extract_features(vid, extractor, fps = 30, frames_to_keep = 5):
    extracted_features = []
    for i in range(0, len(vid), int(fps / frames_to_keep)):
        img = vid[i]
        if not img: continue
        extracted_features.append(extractor.extract(img))

    return extracted_features


def new_get_bovw(features, codebook, vwords):
    bovw = []
    for f in features:
        vw = codebook.predict(f)
        bovw.extend(vw)

    bovw = np.array(bovw)
    bovw = np.histogram(bovw, bins = range(vwords + 1))[0]
    return bovw


def calc_tf_idf(tfs, dfs):
    tf_idf = np.array([])
    for tf, df in zip(tfs, dfs):
        tf = tf / np.sum(tfs)
        idf = np.log(len(tfs) / (df + 1))
        tf_idf = np.append(tf_idf, tf * idf)

    return tf_idf