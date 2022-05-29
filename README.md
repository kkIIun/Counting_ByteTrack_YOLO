# 소개

다중 객체 추적(Multi Object Tracking, MOT)은 연속된 프레임 속에서 다중 객체에 대한 bounding box와 ID를 지속적으로 추적하는 것을 목표로 합니다. 대부분의 방법은 신뢰도(confidence score)가 임계값(threshold)보다 높게 검출된 객체의 bounding box를 연결하여 ID를 부여합니다.

해당 시스템은 Raspberry Pi 4 Model B (RPi4B) 와 Intel Neural Compute Stick 2 (NCS2) 상에서 구동을 확인했습니다.
추가적으로 NCS2를 활용하기 위한 OpenVINO toolkit 그리고 ByteTrack 알고리즘을 이용합니다.

ByteTrack에 대해선 다음을 참고하였습니다.

[ByteTrack Paper](https://arxiv.org/abs/2110.06864)

[ByteTrack Official GitHub Repo](https://github.com/ifzhang/ByteTrack)

OpenVINO는 AI 추론을 엣지 디바이스 상에서 최적화하기 위한 오픈 소스 toolkit 입니다. OpenVINO를 참조하기 위해서는 다음과 같은 페이지를 참조하시기 바랍니다.

[OpenVINO_GitHub](https://github.com/openvinotoolkit/openvino)

[OpenVINO_Overview](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)

[OpenVINO_Releases_Notes](https://www.intel.com/content/www/us/en/developer/articles/release-notes/openvino-relnotes.html)

지원 OS
- [Linux](https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_linux.html)
- [Windows](https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_windows.html)
- [macOS](https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_macos.html)
- [Raspbian](https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_raspbian.html)

# 개발환경

# 사용법 혹은 설치방법

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
