# !pip install pyyaml==5.1 

import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
# !gcc --version
assert torch.__version__.startswith("1.7")
# !pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.7/index.html
# 
# '''
# !git clone https://github.com/facebookresearch/detectron2 detectron2_repo
# !pip install -e detectron2_repo
# '''

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

from Buchnearer import PRC, average_precision, Buchnearer_detector, get_gtboxes, validateBoxes, f1_score

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("Buchnearer",)
cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 8 #number of cores loading an image at once - higher = faster, but more memory
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.01 #If learn rate goes too low I get memory problems. I should look into optimization
cfg.SOLVER.MAX_ITER = 40000 #80000 is the most I've run, it doesn't cause much difference, I could even step this back to ~10-20. I should run a loss curve though.
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1024   #reset from 100 in tutorial, #1024 in 20201103
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 
cfg.TEST.DETECTIONS_PER_IMAGE = 3000
cfg.SOLVER.CHECKPOINT_PERIOD = 1000

cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32], [64], [128], [128]]
cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[1.0]]
cfg.MODEL.RPN.IN_FEATURES = ["p2", "p3", "p4", "p5"]

cfg.MODEL.WEIGHTS = "/home/xupan/Projects/Buchnearer/20210410_30000.pth"

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05   # set the testing threshold for this model, higher = more certainty but worse inclusion, lower = less certainty, but better inclusion
cfg.TEST.DETECTIONS_PER_IMAGE = 5000 #max number of objects per image. 5000 should be overkill, I think there are rarely more than 2000
cfg.DATASETS.TEST = ("Buchnearer", )

predictor = DefaultPredictor(cfg)

# Get all scores vs prediction label.
input_path = '/home/xupan/Projects/Buchnearer/test_image_by_age'
total_validation = np.array([])
total_scores = np.array([])
total_positive = 0
for f in os.listdir(input_path):
    age_folder = join(input_path, f)
    for im in os.listdir(age_folder):
        if im[-1] == 'g':
            print(im)
            output_path = join(age_folder,'Prediction')
            pred_boxes, pred_masks, pred_scores = Buchnearer_detector(join(age_folder, im),None,predictor)
            gt_boxes = get_gtboxes(join(age_folder, im)[0:-3]+'json')
            validation_result = validateBoxes(gt_boxes, pred_boxes)
            total_validation = np.concatenate((total_validation,validation_result))
            total_scores = np.concatenate((total_scores,pred_scores))
            total_positive += len(gt_boxes)
np.save('total_validation.npy', total_validation)
np.save('total_scores.npy', total_scores)
np.save('total_positive.npy', total_positive)

precision, recall, threshold_score = PRC(total_scores,total_validation,total_positive)

plt.figure()
plt.step(recall, precision, where='post')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.savefig("PRC.png")
plt.show()

f1 = f1_score(recall,precision)
plt.figure()
plt.plot(threshold_score, f1, label='F1')
plt.plot(threshold_score, precision, label='precision')
plt.plot(threshold_score, recall, label='recall')

plt.xlabel('threshold')
plt.legend()
plt.savefig("f1.png")
plt.show()

print("Average precision: {}".format(average_precision(recall, precision)))