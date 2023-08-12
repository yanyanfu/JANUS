# JANUS: Enhanced Duplicate Detection for Video-based Bug reports from Two Modalities

## Introduction
This is the official codebase for the approach "A Tale of Learning from Two Modalities: Enhanced Duplicate Detection for Video-based Bug Reports"

## Results

### Visual

![image_1](tabels/visual.png)
![image_2](tabels/visual_config.png)

### textual
![image_3](tabels/text.png)
![image_4](tabels/text_comparison.png)

## Data

All the data related to the benchmark and the approach itself is provided in [Zendo](https://sandbox.zenodo.org/record/1166765#.Y_Y4CexBx8Y).

* The video-based bug reports are located in artifacts/videos folder
* The generated codebooks and the DINO checkpoint fine-tuned on Rico dataset for visual JANUS are in the artifacts/models/vision folder
* The needed east checkpoint and the extracted text based on EAST & TrOCR can be found in the artifacts/models/text folder 
* The evaluation setting corresponding to 7,290 tasks is provided in the outputs/evaluation_settings folder
* All the results for the visual, textual and combined JANUS are located in the outputs folder

## Training Prerequisites
- CUDA 11.0
- python 3.6
- pytorch 1.7.1
- torchvision 0.8.2

## Installation

```bash
git clone https://anonymous.4open.science/r/JANUS-A0FB/
cd JANUS
pip install -r requirements.txt
```

## Reproduce the results

```bash
python cli.py
```

