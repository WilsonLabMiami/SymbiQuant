#Import torch and detectron, set up detectron
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
print('')

# import some common libraries
import numpy as np
import cv2
import os
from os.path import isfile, join
import matplotlib.pyplot as plt
import math
import random
import re

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

#Build config for predictor
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("Buchnearer",)
cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 8
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.01
cfg.SOLVER.MAX_ITER = 40000 
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1024   #1024 in 20201103
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 
cfg.TEST.DETECTIONS_PER_IMAGE = 3000
cfg.SOLVER.CHECKPOINT_PERIOD = 1000
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32], [64], [128], [128]]
cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[1.0]]
cfg.MODEL.RPN.IN_FEATURES = ["p2", "p3", "p4", "p5"]
#Bring in weights from Buchnearer - can be swapped for custom training sets!
cfg.MODEL.WEIGHTS = "/content/model_final_randomscale_0.5_1_40000.pth"

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model, higher = more certainty but worse inclusion, lower = less certainty, but better inclusion
cfg.TEST.DETECTIONS_PER_IMAGE = 5000 #max number of objects per image. 
cfg.DATASETS.TEST = ("Buchnearer", )

predictor = DefaultPredictor(cfg)

from Buchnearer import Buchnearer_detector
#input_path = '/home/xupan/Projects/Buchnearer/test_image_by_age'
for f in os.listdir(input_path):
    age_folder = join(input_path, f)
    output_path = join(age_folder,'Prediction')
    os.mkdir(output_path)
    for im in os.listdir(age_folder):
        if im[-1] == 'g':
            print(im)
            Buchnearer_detector(join(age_folder, im),join(output_path, im),predictor, visualize=True, save_result=True)