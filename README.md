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

OpenVINO의 지원 OS
- [Linux](https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_linux.html)
- [Windows](https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_windows.html)
- [macOS](https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_macos.html)
- [Raspbian](https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_raspbian.html)

# 개발환경
# 개발환경(Environment)

- Device :
Raspberry Pi 4 model B+ (4GB,8GB)
Intel Neural Compute Stick 2
usb2.0 camera
- OS : 
Raspbian Buster, 32-bit
- Util:
openvino == 2021.4.2
cmake == 3.16.3
- Python :
python == 3.7.3
- pip3:
numpy == 1.21.6
scipy == 1.7.3
loguru == 0.6.0
lap == 0.4.0
cython_bbox == 0.1.3
Pillow == 5.4.1

라즈베리파이 설정(필요 하신분 가져가세요)

1. Raspbian Buster, 32-bit OS를 설치. [https://www.raspberrypi.com/software/](https://www.raspberrypi.com/software/)
2. 라즈베리파이에 OpenVINO toolkit을 설치한다.
모든 버전 : [https://storage.openvinotoolkit.org/repositories/openvino/packages/](https://storage.openvinotoolkit.org/repositories/openvino/packages/)
    1. tem 경로로 이동
    `cd /tmp`
    2. 빌드된 openvino 파일 다운로드
    `wget [https://storage.openvinotoolkit.org/repositories/openvino/packages/2021.4.2/l_openvino_toolkit_runtime_raspbian_p_2021.4.752.tgz](https://storage.openvinotoolkit.org/repositories/openvino/packages/2021.4.2/l_openvino_toolkit_runtime_raspbian_p_2021.4.752.tgz)`
    3. openvino 설치 경로 생성 및 받은 파일 압축 해제
    `sudo mkdir -p /opt/intel
     cd /opt/intel
     sudo tar -xf /tmp/l_openvino_toolkit_runtime_raspbian_p_2021.4.752.tgz -C /opt/intel`
    `sudo mv l_openvino_toolkit_runtime_raspbian_p_2021.4.752 openvino`
    4. 다운로드 파일 삭제
    `rm -f /tmp/l_openvino_toolkit_runtime_raspbian_p_2021.4.752.tgz`
    5. cmake 설치
    `sudo apt install cmake`
    6. openvino 환경 실행
    `source /opt/intel/openvino/bin/setupvars.sh`
    7. (optional) 터미널을 킬 때마다 위의 명령어를 실행
    `echo "source /opt/intel/openvino/bin/setupvars.sh" >> ~/.bashrc`
    8. NCS2 규칙 추가 `$(whoami)`에 계정이름(defult = pi)
    `sudo usermod -a -G users "$(whoami)”
     sh /opt/intel/openvino/install_dependencies/install_NCS_udev_rules.sh`
    9. 기타 파일 설치
    `sudo apt install libgfortran5 libatlas3-base
    sudo apt-get install libatlas-base-dev`

메인 기기 설정(window, ubuntu)

`pip install openvino==2021.4.1`

`pip install openvino-dev==2021.4.1`

`pip install openvino-dev[onnx]`

# 사용법 혹은 설치방법(How to install?)

## Installation

Step1. clone this repo:

```
git clone https://github.com/ByungOhKo/Counting_ByteTrack_YOLO.git
cd ByteTrack
pip3 install -r requirements.txt
```

Step2. Install other dependencies

```jsx
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip3 install cython_bbox
sudo apt install libgfortran5 libatlas3-base
sudo apt-get install libatlas-base-dev
```

## Inference

Use either -i or -s to specify your config.

```jsx
cd ByteTrack
python3 openvino_inference.py -m model/yolox_tiny_openvino/yolox_tiny -i "416,416" -s 0.5 --track_thresh 0.5
```

## **How to track with your custom model:**

Step1. Train your own model.

Step2. Run Model Optimizer to convert the model to IR:

```jsx
mo --input_model INPUT_MODEL_DIR
```

Step3. Create the folder `your_model_name` in the directory `model\`

Step4. Put IR files (.xml .bin) in the directory `model\your_model_name\`

# about runtime parameters

 model: yolo 모델
