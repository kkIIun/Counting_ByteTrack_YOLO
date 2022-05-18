# ByteTrack

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bytetrack-multi-object-tracking-by-1/multi-object-tracking-on-mot17)](https://paperswithcode.com/sota/multi-object-tracking-on-mot17?p=bytetrack-multi-object-tracking-by-1)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bytetrack-multi-object-tracking-by-1/multi-object-tracking-on-mot20-1)](https://paperswithcode.com/sota/multi-object-tracking-on-mot20-1?p=bytetrack-multi-object-tracking-by-1)

#### ByteTrack is a simple, fast and strong multi-object tracker.

<p align="center"><img src="assets/sota.png" width="500"/></p>

> [**ByteTrack: Multi-Object Tracking by Associating Every Detection Box**](https://arxiv.org/abs/2110.06864)
> 
> Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, Zehuan Yuan, Ping Luo, Wenyu Liu, Xinggang Wang
> 
> *[arXiv 2110.06864](https://arxiv.org/abs/2110.06864)*

## Demo Links
| Google Colab demo | Huggingface Demo | Original Paper: ByteTrack |
|:-:|:-:|:-:|
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bDilg4cmXFa8HCKHbsZ_p16p0vrhLyu0?usp=sharing)|[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/bytetrack)|[arXiv 2110.06864](https://arxiv.org/abs/2110.06864)|
* Integrated to [Huggingface Spaces](https://huggingface.co/spaces) with [Gradio](https://github.com/gradio-app/gradio).


## Abstract
Multi-object tracking (MOT) aims at estimating bounding boxes and identities of objects in videos. Most methods obtain identities by associating detection boxes whose scores are higher than a threshold. The objects with low detection scores, e.g. occluded objects, are simply thrown away, which brings non-negligible true object missing and fragmented trajectories. To solve this problem, we present a simple, effective and generic association method, tracking by associating every detection box instead of only the high score ones. For the low score detection boxes, we utilize their similarities with tracklets to recover true objects and filter out the background detections. When applied to 9 different state-of-the-art trackers, our method achieves consistent improvement on IDF1 scores ranging from 1 to 10 points. To put forwards the state-of-the-art performance of MOT, we design a simple and strong tracker, named ByteTrack. For the first time, we achieve 80.3 MOTA, 77.3 IDF1 and 63.1 HOTA on the test set of MOT17 with 30 FPS running speed on a single V100 GPU.
<p align="center"><img src="assets/teasing.png" width="400"/></p>

## Tracking performance
### Results on MOT challenge test set
| Dataset    |  MOTA | IDF1 | HOTA | MT | ML | FP | FN | IDs | FPS |
|------------|-------|------|------|-------|-------|------|------|------|------|
|MOT17       | 80.3 | 77.3 | 63.1 | 53.2% | 14.5% | 25491 | 83721 | 2196 | 29.6 |
|MOT20       | 77.8 | 75.2 | 61.3 | 69.2% | 9.5%  | 26249 | 87594 | 1223 | 13.7 |

### Visualization results on MOT challenge test set
<img src="assets/MOT17-01-SDP.gif" width="400"/>   <img src="assets/MOT17-07-SDP.gif" width="400"/>
<img src="assets/MOT20-07.gif" width="400"/>   <img src="assets/MOT20-08.gif" width="400"/>

## Installation
### 1. Installing on the host machine
Step1. Install ByteTrack.
```shell
git clone https://github.com/ByungOhKo/Counting_ByteTrack_YOLO.git
cd ByteTrack
pip3 install -r requirements.txt
```

Step2. Install [pycocotools](https://github.com/cocodataset/cocoapi).

```shell
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

Step3. Others
```shell
pip3 install cython_bbox
```

## Acknowledgement

A large part of the code is borrowed from [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), [FairMOT](https://github.com/ifzhang/FairMOT), [TransTrack](https://github.com/PeizeSun/TransTrack) and [JDE-Cpp](https://github.com/samylee/Towards-Realtime-MOT-Cpp). Many thanks for their wonderful works.
