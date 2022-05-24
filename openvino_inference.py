import argparse
import os
import time

import cv2
import numpy as np
from loguru import logger

from openvino.inference_engine import IECore

from yolox.utils import preprocess, postprocess, multiclass_nms, is_in_line, vis, mkdir
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer


def make_parser():
    parser = argparse.ArgumentParser("openvino inference sample")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="model/yolox_tiny_openvino/yolox_tiny",
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
        ie = IECore()
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.args = args
        self.net = ie.read_network(model=args.model+'.xml', weights=args.model+'.bin')
        self.exec_net = ie.load_network(network=self.net, device_name='MYRIAD')
        self.input_shape = tuple(map(int, args.input_shape.split(',')))
        self.input_key = list(self.exec_net.input_info)[0]
        self.output_key = list(self.exec_net.outputs.keys())[0]
        print('Predictor initialization Complete')

    def inference(self, ori_img, timer):
        # Load a image info
        img_info = {"id": 0}
        height, width = ori_img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = ori_img
        
        # Preprocess the input image
        img, ratio = preprocess(ori_img, self.input_shape, self.rgb_means, self.std)
        img = img[np.newaxis, ...]
        img_info["ratio"] = ratio
        
        # Inference on NCS2
        timer.tic()
        output = self.exec_net.infer(inputs={self.input_key: img})
        output = output[self.output_key]
        
        # Postprocess the output
        predictions = postprocess(output, self.input_shape, p6=self.args.with_p6)[0]
                
        boxes = predictions[:, :4]
        # 5 = person, 6 = bicycle, 7 = car
        # if only want to detect bicycle,
        # scores = predictions[:, 4:5] * predictions[:, 6]
        scores = predictions[:, 4:5] * predictions[:, 5:8] # predictions[:, 5:8] = person, bicycle, car

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=self.args.nms_thr, score_thr=self.args.score_thr)
        
        if dets is not None:
            return dets, img_info
        else:
            return np.zeros((1, 6)), img_info
        

def imageflow_demo(predictor, args):
    # Define a cv2.VideoCapture
    cap = cv2.VideoCapture(0)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    ''' For saving the output video
    save_folder = args.output_dir
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, args.video_path.split("/")[-1])
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    '''
    
    # Define a tracker (ByteTrack)
    tracker = BYTETracker(args, frame_rate=10)
    timer = Timer()
    frame_id = 0
    
    appear_flags = {}
    cnt = 0
    
    while True:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        # Get the current frame
        ret_val, frame = cap.read()
        if ret_val:
            # Get the detections and image_info
            outputs, img_info = predictor.inference(frame, timer)
            
            # Define two lines for counting objects
            line = [((0, int(0.2 * img_info['height'])), (int(img_info['width']), int(0.2 * img_info['height']))),
                    ((0, int(0.8 * img_info['height'])), (int(img_info['width']), int(0.8 * img_info['height'])))]
            
            # Update the tracklets
            online_targets = tracker.update(outputs[:, :-1], [img_info['height'], img_info['width']], [img_info['height'], img_info['width']])
            
            online_tlwhs = []
            online_centroids = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                centroid = t.tlwh_to_xyah(tlwh)[:2]
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    online_centroids.append(centroid)
                
                # Count the objects
                if is_in_line(centroid, line[0], margin=10):
                    if tid in appear_flags:
                        if appear_flags[tid] == 2:
                            cnt += 1
                            del appear_flags[tid]
                    appear_flags[tid] = 1
                    
                if is_in_line(centroid, line[1], margin=10):
                    if tid in appear_flags:
                        if appear_flags[tid] == 1:
                            cnt += 1
                            del appear_flags[tid]
                    appear_flags[tid] = 2
                
            timer.toc()
            
            # Draw the tracking results
            online_im = plot_tracking(img_info['raw_img'], online_tlwhs, online_centroids, online_ids, online_scores, cnt,
                                      frame_id=frame_id + 1, fps=1. / timer.average_time)
            
            # Draw two lines for counting objects
            cv2.line(online_im, line[0][0], line[0][1], (0, 255, 0), 2)
            cv2.line(online_im, line[1][0], line[1][1], (0, 255, 0), 2)
            
            # Plot the current frame
            cv2.imshow('out', online_im)
            # vid_writer.write(online_im)
            
            # Press 'q' to stop this system
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
