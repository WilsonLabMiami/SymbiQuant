from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os

import sys
sys.path.insert(1, '/content/Buchnearer') # replace xxxxxx with the folder name where trainer_augmentation.py is in.
from trainer_augmentation import BuchneraTrainer

cfg = get_cfg()
cfg.merge_from_file("./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
#Xu's idea to save and load the config from this place here


cfg.DATASETS.TRAIN = ("Buchnearer",)
cfg.DATASETS.TEST = ("Buchnearer_test")   #add test
cfg.DATALOADER.NUM_WORKERS = 2 #number of cores loading an image at once - higher = faster, but more memory
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.MAX_ITER = 40000 
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1024   
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32], [64], [128], [128]]
cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[1.0]]
cfg.MODEL.RPN.IN_FEATURES = ["p2", "p3", "p4", "p5"]
cfg.TEST.DETECTIONS_PER_IMAGE = 3000

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = BuchneraTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()