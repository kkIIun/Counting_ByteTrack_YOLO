# Introduction

다중 객체 추적(Multi Object Tracking, MOT)은 연속된 프레임 속에서 다중 객체에 대한 bounding box와 ID를 지속적으로 추적하는 것을 목표로 합니다. 대부분의 방법은 신뢰도(confidence score)가 임계값(threshold)보다 높게 검출된 객체의 bounding box를 연결하여 ID를 부여합니다.

해당 시스템은 Raspberry Pi 4 Model B (RPi4B) 와 Intel Neural Compute Stick 2 (NCS2) 상에서 구동을 확인했습니다.
추가적으로 NCS2를 활용하기 위한 OpenVINO toolkit 그리고 ByteTrack 알고리즘을 이용합니다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0b7d7985-4a55-43ae-8299-8082918e205b/Untitled.png)
![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/906ddb12-2362-4bde-a8ea-a72b66960cbb/Untitled.png)

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

## Inference
```shell
cd ByteTrack
python3 openvino_inference.py -m model/yolox_tiny_openvino/yolox_tiny -i "416,416" -s 0.5 --track_thresh 0.5
```
