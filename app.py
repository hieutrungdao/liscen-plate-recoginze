import argparse
import os
import re 
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import (
    check_img_size, non_max_suppression, scale_coords, plot_one_box, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.ocr import crop_plate, crop_char, sort_blocks, transform_char, get_key, img_for_train
from models.conv import ConvNet



def detect(save_img=False):
    out, source, imgsz = \
        opt.output, opt.source, opt.img_size

    weights_plate='weights/best_model_detect_plate.pt'
    weights_char='weights/best_model_detect_char.pt'
    weights_class = 'weights/best_char.ckpt'

    # Initialize
    set_logging()
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model_plate = attempt_load(weights_plate, map_location=device)  # load FP32 model
    imgsz_plate = check_img_size(imgsz, s=model_plate.stride.max())  # check img_size

    model_char = attempt_load(weights_char, map_location=device)  # load FP32 model
    imgsz_char = check_img_size(imgsz, s=model_char.stride.max())  # check img_size

    classify_model = ConvNet().to(device)
    classify_model.load_state_dict(torch.load(weights_class))
    classify_model.eval()

    if half:
        model_plate.half()  # to FP16
        model_char.half()  # to FP16

    save_img = True
    dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names1 = model_plate.module.names if hasattr(model_plate, 'module') else model_plate.names
    colors1 = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names1))]

    names2 = model_char.module.names if hasattr(model_char, 'module') else model_char.names
    colors2 = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names2))]


    # Run detect
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model_plate(img.half() if half else img) if device.type != 'cpu' else None  # run once

    for path, img, im0s, vid_cap in dataset:
        print('\n')

        t1 = time_synchronized()

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        pred = model_plate(img, augment=opt.augment)[0]
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, agnostic=opt.agnostic_nms)

        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names1[int(c)])  # add to string
                print('%sDone. (%.3fs)' % (s, t2 - t1))

                for *xyxy, conf, cls in reversed(det):
                    crop_img = crop_plate(xyxy, im0)
                    plate = {'img':im0, 'crop': crop_img, 'bbox': xyxy}

                    img = torch.zeros((1, 3, imgsz, imgsz), device=device)
                    _ = model_char(img.half() if half else img) if device.type != 'cpu' else None  # run once
                    img = img_for_train(plate['crop'])

                    t3 = time_synchronized()

                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)
                    # Inference
                    pred = model_char(img, augment=opt.augment)[0]
                    # Apply NMS
                    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, agnostic=opt.agnostic_nms)
                    
                    t4 = time_synchronized()
                    
                    # Process detections
                    for i, det in enumerate(pred):
                        if det is not None and len(det):
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], plate['crop'].shape).round()
                            s0 = ''
                            for c in det[:, -1].unique():
                                n = (det[:, -1] == c).sum()  # detections per class
                                s0 += '%g %ss, ' % (n, names1[int(c)])  # add to string
                            print('%sDone. (%.3fs)' % (s0, t4 - t3))

                            # Write results
                            char_list = []
                            for *xyxy, conf, cls in reversed(det):
                                char_dict = crop_char(xyxy, plate['crop'])
                                char_list.append(char_dict)
                            
                            # Label 
                            char_list = sort_blocks(char_list)
                            string_output = ''
                            for i, v in enumerate(char_list):
                                input = transform_char(v['img'])
                                classify_output = classify_model(input)
                                _, predicted = torch.max(classify_output.data, 1)
                                predicted = get_key(int(predicted.numpy()))
                                string_output += str(predicted)
                    
                    # Draw retangle and label plate
                    plot_one_box(plate['bbox'], plate['img'], label=string_output, color=0, line_thickness=2)
                    print('\nResult: ',string_output)
                    print('Time. (%.3fs)' % (time.time() - t1))
                    print('\n')

            # Save results (image with detections)
            if save_img:
                cv2.imwrite(save_path, plate['img'])
                pass

    if save_img:
        print('Results saved to %s' % Path(out))
        if platform.system() == 'Darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='inference/images/', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output/', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect()
