# Semantic GUI Scene Learning and Video Alignment for DetectingDuplicate Video-based Bug Reports

## Introduction
This is the official codebase for the approach "Semantic GUI Scene Learning and Video Alignment for DetectingDuplicate Video-based Bug Reports"

## Results

### Visual

We compared JANUS(visual) with baseline TANGO(visual) by experimenting with two ViT models: ViT-Small (ViT-S) and ViT-Base (ViT-B). * ViT-S has a similar size to RestNet-50's size (used by TANGO's SimCLR): ~23M parameters *. We also experimented with the following patch sizes for ViT: $16 x 16$ (/16) and $8 x 8$ (/8) pixels, as the patch size can affect JANUS performance. It is worth noting that both * ViT-S/16 and ViT-S/8 used by JANUS's DINO outperform ResNet-50 used by TANGO's SimCLR with statistical significance * even though the model size of ViT-S is comparable to ResNet-50. 

<img src="tabels/visual.png" alt="visual results" width="640">

The following table shows the network configurations and primary hyperparameters used by TANGO(visual) and JANUS(visual).

<img src="tabels/visual_config.png" alt="visual results" width="320">


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

