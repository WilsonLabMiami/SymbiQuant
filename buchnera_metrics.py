#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 16:50:43 2021

@author: ed
"""
import cv2
import pickle
import numpy as np
import copy
import hashlib
import os
from os.path import join
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

def __add_all_masks(image, masks, color):
        mask_image = copy.deepcopy(image)
        for i in range(len(masks)):
            for (a,b) in masks[i]:
                mask_image[a,b,:] = mask_image[a,b,:]*0.5 + color[i]*255*0.5
        return mask_image

def get_segimg(im_file, result_file, output_file):
	image = cv2.imread(im_file)
	with open(result_file, 'rb') as dbfile:      
	            result = pickle.load(dbfile)
	masks = result['masks']
	assigned_colors = [_COLORS[int(hashlib.sha256(str(masks[i]).encode('utf-8')).hexdigest(), 16) % 55] for i in range(len(masks))]
	segimage = __add_all_masks(image, masks, assigned_colors)
	cv2.imwrite(output_file, segimage)

def get_cellcount(result_file):
	with open(result_file, 'rb') as dbfile:      
	            result = pickle.load(dbfile)
	return len(result['masks'])


def get_average_area(result_file):
    with open(result_file, 'rb') as dbfile:      
            result = pickle.load(dbfile)
    return np.mean([len(x) for x in result['masks']])

def get_std_area(result_file):
    with open(result_file, 'rb') as dbfile:      
            result = pickle.load(dbfile)
    return np.std([len(x) for x in result['masks']])

def get_total_area(result_file):
    with open(result_file, 'rb') as dbfile:      
            result = pickle.load(dbfile)
    return np.sum([len(x) for x in result['masks']])

def get_total_area_without_buc_overlaps(result_file):
    with open(result_file, 'rb') as dbfile:      
            result = pickle.load(dbfile)
    total = []
    for cell in result['masks']:
        total = total+list(map(str,cell.tolist()))
    total = list(dict.fromkeys(total))
    return len(total)

def individual_buchnera_areas(result_file):
    areas =[]
    with open(result_file, 'rb') as dbfile:      
            result = pickle.load(dbfile)
            for buchnera in range(len(result['masks'])):
                area = (len(result['masks'][buchnera]))
                areas.append(area)
    return areas
                

            
'''
# get new segmentation images:
input_path = '/Users/ed/Documents/UM/NN_outs/qc_check'
for f in os.listdir(input_path):
    age_folder = os.path.join(input_path, f)
    for im in os.listdir(age_folder):
        if im[-1] == 'g':
            print(im)
            qcname = 'qc' + str(im[0:(len(im)-4)]) + '.result'
            result_file = os.path.join(age_folder, qcname) # find result here (maybe from "age_foler" & "im").
            output_file = os.path.join(im, '_qc.png') # change this
            get_segimg(im, result_file, output_file)



#get cell count:
count_file = '/Users/ed/Documents/UM/NN_outs/qc_check/qc_check.txt'
input_path = '/Users/ed/Documents/UM/NN_outs/qc_check'
for f in os.listdir(input_path):
    age_folder = os.path.join(input_path, f)
    for result in os.listdir(age_folder):
        if im[-1] == 'g':
            count = get_cellcount(result)
            with open(count_file, 'w') as f:
                f.write(im + str(count) +'\n')
'''

count_file = 'C:/Users/ebjam/Downloads/20211031_D5-20211031T201456Z-001/20211031_D5/buch_qc_count.txt'
input_path = 'C:/Users/ebjam/Downloads/20211031_D5-20211031T201456Z-001/20211031_D5'
#"C:\Users\ebjam\Downloads\test_image_by_age-20210802T182613Z-001\test_image_by_age"

for f in os.listdir(input_path):
    if f.startswith('.'):
        print(f)
    else:
        age_folder = join(input_path, f)
        for im in os.listdir(age_folder):
            if im.startswith('.'):
                print(im)
            elif im[-1] == 'g':
                print(im)
                result_file = join(age_folder, 'Prediction/qc' + im[0:-3] + 'result') # find result here (maybe from "age_foler" & "im").
                output_file = join(age_folder, 'Prediction/qc' + im) # change this
                get_segimg(join(age_folder, im), result_file, output_file)
                count = get_cellcount(result_file)
                avg_buc_area = get_average_area(result_file)
                #total_buc_area = get_total_area(result_file)
                #avg_buc_area_std = get_std_area(result_file)
                individual_buchnera = individual_buchnera_areas(result_file)
                area_no_overlap = get_total_area_without_buc_overlaps(result_file)
                with open(count_file, 'a+') as f:
                    print(count)
                    f.write('qc' + im + 'count: ' + str(count) + 'total buc area: ' + str(area_no_overlap) + '\n')
                    #f.write('qc' + im + 'individual Buchnera:' + str(individual_buchnera))
