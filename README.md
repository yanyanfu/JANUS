# Semantic GUI Scene Learning and Video Alignment for Detecting Duplicate Video-based Bug Reports
### Introduction
This is the official codebase for the approach "Semantic GUI Scene Learning and Video Alignment for Detecting Duplicate Video-based Bug Reports"

### Results
**Visual:** We compared JANUS (visual) with the baseline TANGO (visual) by experimenting with two ViT models: ViT-Small (ViT-S) and ViT-Base (ViT-B). Although ViT-S has a **similar model size** to RestNet-50: ~23M parameters, the **ViT-S (used by JANUS's DINO) outperform ResNet-50 (used by TANGO's SimCLR) with statistical significance** on duplicate video-based bug report detection. 

<img src="tabels/visual.png" alt="visual results" width="720">

The following table shows the network configurations and primary hyperparameters used by TANGO(visual) and JANUS(visual).

<img src="tabels/visual_config.png" alt="visual results" width="480">

**Textual:** We compared JANUS (textual) against TANGO's textual component by experimenting with different configurations for the EAST and TrOCR models. For EAST, we used three different resolution thresholds to filter out small text regions: 5 x 5 (EAST-5), 40 x 20 (EAST-40), and 80 x 40 (EAST-80). For TrOCR, two fine-tuned TrOCR-Large models are used, namely TrOCR-p (fine-tuned on the printed text dataset SROIE) and  TrOCR-s (finetuned on the synthetic scene text datasets such as ICDAR15 and SVT)

<img src="tabels/text.png" alt="textual results" width="560">

<img src="tabels/text_comparison.png" alt="trocr results" width="420">

### Data

All the data related to the benchmark and the approach itself is provided in [Zendo](https://sandbox.zenodo.org/record/1166765#.Y_Y4CexBx8Y).

* The video-based bug reports are located in artifacts/videos folder
* The generated codebooks and the DINO checkpoint fine-tuned on Rico dataset for visual JANUS are in the artifacts/models/vision folder
* The needed east checkpoint and the extracted text based on EAST & TrOCR can be found in the artifacts/models/text folder 
* The evaluation setting corresponding to 7,290 tasks is provided in the outputs/evaluation_settings folder
* All the results for the visual, textual and combined JANUS are located in the outputs folder

### Installation

#### Training Prerequisites
- CUDA 11.0
- python 3.6
- pytorch 1.7.1
- torchvision 0.8.2

#### Reproduce the results
```bash
git clone https://anonymous.4open.science/r/JANUS-B65E/
cd JANUS
pip install -r requirements.txt
python cli.py
```

