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
count_file = '/Users/ed/Documents/UM/NN_outs/qc_check.txt'
input_path = '/Users/ed/Documents/UM/NN_outs/qc_check'
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
                with open(count_file, 'a+') as f:
                    print(count)
                    f.write('qc' + im + ': ' + str(count) +'\n')