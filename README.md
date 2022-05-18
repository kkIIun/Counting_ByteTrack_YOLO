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
