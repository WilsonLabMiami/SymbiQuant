"""
Created on Thu Nov 19 13:16:17 2020

@author: ed, Xu
"""

import json
import PIL
from PIL import Image
import copy
import base64
import io
import os
from os import listdir
from os.path import isfile, join
import numpy as np

#https://github.com/wkentaro/labelme/blob/1d6ea6951c025a7db0540c7eac77577bc1507efa/labelme/utils/image.py#L10
def img_b64_to_arr(img_b64):
    f = io.BytesIO()
    f.write(base64.b64decode(img_b64))
    img_arr = np.array(PIL.Image.open(f))
    return img_arr
def img_arr_to_b64(img_arr):
    img_pil = PIL.Image.fromarray(img_arr)
    f = io.BytesIO()
    img_pil.save(f, format='PNG')
    img_bin = f.getvalue()
    if hasattr(base64, 'encodebytes'):
        img_b64 = base64.encodebytes(img_bin)
    else:
        img_b64 = base64.encodestring(img_bin)
    return img_b64

#%% Crop tool - taken from https://stackoverflow.com/questions/5953373/how-to-split-image-into-multiple-pieces-in-python
def imgcrop(file, output_path, square_size=512):
    filename, file_extension = os.path.splitext(file)
    [base_name, _] = os.path.splitext(os.path.basename(file))
    im = Image.open(file)
    imgwidth, imgheight = im.size
    square_size = imgwidth // 4 # resolution is very different, but FOV is similar, so I cut each into 4x4.
    yPieces = imgheight // square_size
    xPieces = imgwidth // square_size
    for i in range(0, yPieces):
        for j in range(0, xPieces):
            box = (j * square_size, i * square_size, (j + 1) * square_size, (i + 1) * square_size)
            a = im.crop(box)
            try:
                a.save(output_path + base_name + "_" + str(i) + "_" + str(j) + file_extension) #This gives a different name to the json
            except:
                pass


def split_json(file, output_path, square_size=512):
    filename, file_extension = os.path.splitext(file)
    [base_name, _] = os.path.splitext(os.path.basename(file))
    with open(file) as data_file:
        data = json.load(data_file)
        imgwidth = data['imageWidth']
        imgheight = data['imageHeight']
        square_size = imgwidth // 4 # resolution is very different, but FOV is similar, so I cut each into 4x4.
        xPieces = imgwidth // square_size
        yPieces = imgheight // square_size
        no_jsons = xPieces * yPieces
        shape_lists = [[] for i in range(no_jsons )] # [[]] * no_jsons doesn't work
        # Split the json up into 512 x 512 tile - output is a list of the shapes that fit into each segment
        for x in range(len(data['shapes'])):
            xlist = []
            ylist = []
            for point in data['shapes'][x]['points']:
                xlist.append(point[0])
                ylist.append(point[1])
            for i in range(xPieces):
                if (i * square_size) <= min(xlist) and max(xlist) < ((i + 1) * square_size):
                    for j in range(yPieces):
                        if (j * square_size) <= min(ylist) and max(ylist) < ((j + 1) * square_size):
                            current_shape = copy.deepcopy(data['shapes'][x])
                            current_shape['points'] = [[x-i*square_size,y-j*square_size] for [x,y] in data['shapes'][x]['points']]
                            list_index = (i * xPieces) + j
                            shape_lists[list_index].append(current_shape)

        # Prepare imageData, imagePath, jsonPath
        imageData = [None] * no_jsons
        imagePath = [None] * no_jsons
        jsonPath = [None] * no_jsons
        imageData_arr = img_b64_to_arr(data['imageData'])
        [path, extention] = os.path.splitext(data['imagePath'])
        for i in range(xPieces):
            for j in range(yPieces):
                list_index = (j * xPieces) + i
                imageData[list_index] = str(img_arr_to_b64(imageData_arr[i*square_size:(i+1)*square_size, j*square_size:(j+1)*square_size]), 'utf-8') # x y is reversed here.
                imagePath[list_index] = path+'_'+str(i)+'_'+str(j)+extention
                jsonPath[list_index] = output_path + base_name +'_'+str(i)+'_'+str(j)+'.json'

        # Save into json files
        for i in range(no_jsons):
            the_json = {
            'version' : '3.16.7',
            'flags' : {},
            'shapes' :[],
            'lineColor' : [0, 255, 0, 128],
            'fillColor' : [225, 0, 0, 128],
            'imagePath' : 'image path here as string',
            'imageData' : 'big mess look this up',
            'imageHeight' : 0,
            'imageWidth' : 0
            }
            the_json['shapes'] = shape_lists[i]
            the_json['imagePath'] = '../'+imagePath[i]
            the_json['imageData'] = imageData[i]
            with open(jsonPath[i], 'w') as final_file:
                    json.dump(the_json, final_file)


if __name__ == "__main__":
    input_path = './whole_cell_train'
    output_path = './whole_cell_train_split/'
    file_list = [f for f in listdir(input_path) if isfile(join(input_path, f))]
    for f in file_list:
        print(f)
        _, file_extension = os.path.splitext(f)
        if file_extension == '.json':
            split_json(join(input_path, f), output_path)
        else:
            imgcrop(join(input_path, f), output_path)