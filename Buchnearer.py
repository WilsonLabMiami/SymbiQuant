import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
#assert torch.__version__.startswith("1.7")

import matplotlib.pyplot as plt
import numpy as np
import cv2

import os
from os.path import isfile, join
import matplotlib.pyplot as plt
import math
import random
import re
import json
import pickle

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("Buchnearer",)
cfg.DATASETS.TEST = ()   
cfg.DATALOADER.NUM_WORKERS = 8 
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.BASE_LR = 0.01 
cfg.SOLVER.MAX_ITER = 40000 
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1024 #1024 in 20201103
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 
cfg.TEST.DETECTIONS_PER_IMAGE = 3000
cfg.SOLVER.CHECKPOINT_PERIOD = 1000
cfg.MODEL.WEIGHTS = "/content/model_final_randomscale_0.5_1_40000.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2   # set the testing threshold for this model, higher = more certainty but worse inclusion, lower = less certainty, but better inclusion
cfg.TEST.DETECTIONS_PER_IMAGE = 5000 #max number of objects per image. 5000 should be overkill, I think there are rarely more than 2000
cfg.DATASETS.TEST = ("Buchnearer", )

predictor = DefaultPredictor(cfg)

_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
    ]
).astype(np.float32).reshape(-1, 3)

def random_color(rgb=False, maximum=255):
    """
    Args:
        rgb (bool): whether to return RGB colors or BGR colors.
        maximum (int): either 255 or 1

    Returns:
        ndarray: a vector of 3 numbers
    """
    idx = np.random.randint(0, len(_COLORS))
    ret = _COLORS[idx] * maximum
    if not rgb:
        ret = ret[::-1]
    return ret

def iou(boxA,boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def pad_mask(h, w, mask, y_start, x_start):
    new_mask = np.zeros((mask.shape[0],h, w), dtype=bool)
    for i in range(mask.shape[0]):
        new_mask[i,x_start:x_start+mask.shape[1],y_start:y_start+mask.shape[2]] = mask[i,:,:]
    return torch.from_numpy(new_mask)

def area(box):
    return((box[2]-box[0])*(box[3]-box[1]))

def delete_overlap(boxes,masks,scores):
    newboxes = []
    newmasks = []
    newscores = []
    for box_i in range(len(boxes)):
        overlap = False
        for box_j in range(box_i+1,len(boxes)):
            if iou(boxes[box_i],boxes[box_j])>0.5:
                overlap = True
                break
        if not overlap:
            newboxes.append(boxes[box_i])
            newmasks.append(masks[box_i])
            newscores.append(scores[box_i])
    return newboxes, newmasks, newscores

def Buchnearer_detector(input_file, output_file, predictor, visualize=False, save_result=False):
    im = cv2.imread(input_file)
    x_end = [min(im.shape[0],512+427*n) for n in range(math.ceil((im.shape[0]+85)/427))]
    x_start = [x-512 for x in x_end]
    y_end = [min(im.shape[1],512+427*n) for n in range(math.ceil((im.shape[1]+85)/427))]
    y_start = [y-512 for y in y_end]
    len_x = len(x_end)
    len_y = len(y_end)
    x_end = x_end * len_y
    x_start =x_start * len_y
    y_end = [item for item in y_end for i in range(len_x)]
    y_start = [item for item in y_start for i in range(len_x)]
    outputs = []
    for i in range(len(x_end)):
        im_crop = im[x_start[i]:x_end[i], y_start[i]:y_end[i],:]
        output = predictor(im_crop)
        outputs.append(output["instances"].to("cpu"))
    del output
    for i in range(0,len(x_end)):
        outputs[i].pred_boxes.tensor = outputs[i].pred_boxes.tensor+np.array([y_start[i],x_start[i],y_start[i],x_start[i]])
        outputs[i].pred_masks = pad_mask(im.shape[0],im.shape[1],outputs[i].pred_masks,y_start[i],x_start[i])

    full_outputs = detectron2.structures.Instances.cat(outputs)
    del outputs

    full_outputs._image_size = (im.shape[0],im.shape[1])
    full_outputs = full_outputs.to("cpu")

    boxes = tuple(full_outputs.pred_boxes.tensor.numpy())
    masks = full_outputs.pred_masks.numpy()
    scores = full_outputs.scores.numpy()

    
    box_area = [area(x) for x in boxes]
    median_size = np.median(box_area)
    lower_bound = 0.1*median_size
    upper_bound = 5*median_size

    filtered_boxes = []
    filtered_masks = []
    filtered_scores = []
    for i in range(len(boxes)):
        if area(boxes[i])>lower_bound and area(boxes[i])<upper_bound:
            filtered_boxes.append(boxes[i])
            filtered_masks.append(masks[i])
            filtered_scores.append(scores[i])

    filtered_boxes,filtered_masks,filtered_scores = delete_overlap(filtered_boxes,filtered_masks,filtered_scores)
    
    f = open('/content/SymbiQuant_count_noQC.txt', "a")
    f.write('Buchenera count: {} '.format(len(filtered_masks)) + input_file + '\n')
    f.close()
    
    if visualize:
        v = Visualizer(im[:, :, ::-1],
                          scale=1.0, 
                          instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
            )
        assigned_colors = [random_color(rgb=True, maximum=1) for _ in range(len(filtered_masks))]
        v = v.overlay_instances(masks=filtered_masks, assigned_colors=assigned_colors)
        image_with_instances = v.get_image()[:, :, ::-1]
        cv2.imwrite(output_file, image_with_instances)    
    
    if save_result:
        result = {}
        result['boxes'] = filtered_boxes
        # mask are (x,y) coordinates of where is inside mask. (not binary masks)
        result['masks'] = [np.array(filtered_masks[i].nonzero()).T for i in range(len(filtered_masks))]
        result['scores'] = filtered_scores
        with open(output_file[0:-3]+'result', 'wb') as outfile:
            pickle.dump(result, outfile)
            
    return filtered_boxes, filtered_masks, filtered_scores

def get_gtboxex(file_path):
    with open(file_path) as data_file:
        data = json.load(data_file)
    gt_boxes = []
    for box_i in range(len(data['shapes'])):
        box = np.concatenate((np.min(data['shapes'][box_i]['points'],0),np.max(data['shapes'][box_i]['points'],0)))
        gt_boxes.append(box)
    return gt_boxes

def validateBoxes(gtboxes, predboxes):
    validation_result = np.zeros(predboxes)
    for box_i in range(len(predboxes)):
        overlap = False
        for box_j in range(len(gtboxes)):
            if iou(predboxes[box_i],gtboxes[box_j])>0.5:
                overlap = True
                break
        if overlap == True:
            validation_result[box_i] = 1
    return validation_result

# Precision recall curve.
def PRC(scores,validation,n):
    order = np.argsort(-scores)
    ordered_socres = scores[order]
    ordered_validation = validation[order]
    step_size = np.floor(len(scores)/100)
    threshold = [int(x*step_size) for x in range(1,100)] + [len(scores)-1]
    TP = np.array([np.sum(ordered_validation[0:t]) for t in threshold])
    # precision = TP/(np.array(threshold)-1)
    precision = TP/(np.array(threshold))
    recall = TP/n
    threshold_score = ordered_socres[threshold]
    return precision, recall, threshold_score

def average_precision(recall, precision):
    dx = recall[1:-1]-recall[0:-2]
    return np.inner(precision[0:-2], dx)

def f1_score(recall, precision):
    return(2/(1/recall+1/precision))

# metric util
def get_gtboxes(file_path):
    with open(file_path) as data_file:
        data = json.load(data_file)
    gt_boxes = []
    for box_i in range(len(data['shapes'])):
        box = np.concatenate((np.min(data['shapes'][box_i]['points'],0),np.max(data['shapes'][box_i]['points'],0)))
        gt_boxes.append(box)
    return gt_boxes

# metric util
def validateBoxes(gtboxes, predboxes):
    validation_result = np.zeros(len(predboxes))
    for box_i in range(len(predboxes)):
        overlap = False
        for box_j in range(len(gtboxes)):
            if iou(predboxes[box_i],gtboxes[box_j])>0.5:
                overlap = True
                break
        if overlap == True:
            validation_result[box_i] = 1
    return validation_result