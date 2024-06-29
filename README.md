# BiReNet
This is the official code repository for "BiReNet: Bilateral Network with Feature Aggregation and Edge Detection for Remote Sensing Images Road Extraction"(PRCV 2024).

## Abstract
Extracting roads from remote sensing images is a significant and challenging topic. Most existing advanced methods perform well in general scenarios but cannot cope well with complex scenarios, such as roads being obscured and covered. This paper proposes a Bilateral Road Extraction Network (BiReNet) consisting of an edge detection branch and a road extraction branch. Firstly, we adopt an effective and efficient LinkNet architecture in the road extraction branch and add an extra Feature Fusion Module (FFM) behind the skip connection part to efficiently aggregate the features at the same level from the decoder and the corresponding encoder to preserve abundant spatial information and amplify comprehension of road structures. Secondly, we design a novel Edge Detection Module (EDM) in the edge detection branch to enhance the road edge features by capturing the gradient information using Pixel Difference Convolution (PDC), which enables fine-grained constraints on the road extraction and improves the accuracy and connectivity of the road. Extensive experiments on two publicly available road datasets show that BiReNet performs favourably against a few state-of-the-art remote sensing road extraction methods and demonstrates stronger robustness in complex scenarios. Specifically, for a 1024 x 1024 input, BiReNet achieved 0.6769 IoU on the DeepGlobe road dataset and 0.6072 IoU on the Massachusetts road dataset with a speed of 24 FPS on one GeForce GTX3090. Source code is available at https://github.com/LPeng625/BiReNet.


## 1.Training
DeepGlobe
`python main_BiReNet_Road.py`

Massachusetts
`python main_BiReNet_Mas.py`

## 2.Evaluation
DeepGlobe
`python test_BiReNet_Road.py`

Massachusetts
`python test_BiReNet_Mas.py`

## 3.Acknowledgments
We thank the author of [RCFSNet](https://github.com/CVer-Yang/RCFSNet) for open-source code.

