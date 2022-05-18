import argparse
import os
import time

import cv2
import numpy as np
from loguru import logger

from openvino.inference_engine import IECore

from yolox.data.data_augment import preproc as preprocess
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import mkdir, multiclass_nms, demo_postprocess, vis
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer


def make_parser():
    parser = argparse.ArgumentParser("openvino inference sample")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="model/yolox_nano",
        help="Input your onnx model.",
    )
    parser.add_argument(
        "-i",
        "--video_path",
        type=str,
        default='videos/palace.mp4',
        help="Path to your input image.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default='demo_output',
        help="Path to your output directory.",
    )
    parser.add_argument(
        "-s",
        "--score_thr",
        type=float,
        default=0.7,
        help="Score threshould to filter the result.",
    )
    parser.add_argument(
        "-n",
        "--nms_thr",
        type=float,
        default=0.7,
        help="NMS threshould.",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="416,416",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        "--with_p6",
        action="store_true",
        help="Whether your model uses p6 in FPN/PAN.",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.7, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument('--min-box-area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser


class Predictor(object):
    def __init__(self, args):
#         self.rgb_means = (0.485, 0.456, 0.406)
#         self.std = (0.229, 0.224, 0.225)
        self.args = args
        self.ie = IECore()
        self.session = self.ie.read_network(model=args.model+'.xml', weights=args.model+'.bin')
        self.exec_net = self.ie.load_network(network=self.session, device_name='MYRIAD')
        self.input_shape = tuple(map(int, args.input_shape.split(',')))
        self.input_key = list(self.exec_net.input_info)[0]
        self.output_key = list(self.exec_net.outputs.keys())[0]

    def inference(self, ori_img, timer):
#         t0 = time.time()
        img_info = {"id": 0}
        height, width = ori_img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = ori_img

        img, ratio = preprocess(ori_img, self.input_shape, self.rgb_means, self.std)
        img = img[np.newaxis, ...]
        
#         t1 = time.time()
#         preprocess_t = t1 - t0

        img_info["ratio"] = ratio
        
        timer.tic()
        output = self.exec_net.infer(inputs={self.input_key: img})
        output = output[self.output_key]
#         t2 = time.time()
#         forward_t = t2 - t1
        
        predictions = demo_postprocess(output, self.input_shape, p6=self.args.with_p6)[0]
#         t3 = time.time()
#         demo_postprocess_t = t3 - t2
        
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:8] # predictions[:, 5:8] = probs per person, bicycle, car

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=self.args.nms_thr, score_thr=self.args.score_thr)
        
#         t4 = time.time()
#         nms_t = t4 - t3
#         print(f'preproc_t : {preprocess_t:.2f}, forward_t = {forward_t:.2f}, demo_postprocess_t : {demo_postprocess_t:.2f}, nms_t = {nms_t:.2f}')

        if dets is not None:
            return dets, img_info
        else:
            return np.zeros((1, 6)), img_info
        

def imageflow_demo(predictor, args):
    cap = cv2.VideoCapture(0)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
#     save_folder = args.output_dir
#     os.makedirs(save_folder, exist_ok=True)
#     save_path = os.path.join(save_folder, args.video_path.split("/")[-1])
#     logger.info(f"video save_path is {save_path}")
#     vid_writer = cv2.VideoWriter(
#         save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
#     )
    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    results = []
    while True:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame, timer)
            online_targets = tracker.update(outputs[:, :-1], [img_info['height'], img_info['width']], [img_info['height'], img_info['width']])
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
            timer.toc()
            results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
            det_im = vis(img_info['raw_img'], outputs[:, :4], outputs[:, 4], outputs[:, 5],
                         conf=args.score_thr, class_names=COCO_CLASSES)
            online_im = plot_tracking(img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1,
                                      fps=1. / timer.average_time)
            cv2.imshow('det', det_im)
            cv2.imshow('out', online_im)
            # vid_writer.write(online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1
    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    args = make_parser().parse_args()

    predictor = Predictor(args)
    imageflow_demo(predictor, args)
