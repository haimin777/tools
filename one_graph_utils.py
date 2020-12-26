
# utils for creating custom model and pipeline

import zipfile
import datetime
import string
import glob
import math
import os, sys
import shutil as sh

import typing
import tqdm
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import tensorflow as tf
import sklearn.model_selection

from tensorflow import keras
import keras_ocr
import numpy as np
import pandas as pd

from xml.etree import ElementTree
from xml.etree.ElementTree import XMLParser
from numpy import array

import cv2
from PIL import Image

from tensorflow_addons import image as image_tfa



from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import ops

from tensorflow.python.ops import array_ops
from keras import layers
from tensorflow_addons.utils.resource_loader import LazySO
from tensorflow_addons.image import connected_components
_image_so = LazySO("custom_ops/image/_image_ops.so")



from tensorflow.python.keras.engine.base_preprocessing_layer import PreprocessingLayer


@tf.function
def decode_bbox(bboxes, bbox_num, res):

    "decodes from [y, x,  h, w] to [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]" 
    for i in tf.range(bbox_num):
        x1, x2, y1, y2 = bboxes[i][1] *2 , (bboxes[i][1]  + bboxes[i][3]) *2, bboxes[i][0] *2, (bboxes[i][0] + bboxes[i][2]) *2 
        # rescale
        res = res.write(i, [[x1, y1],                         
                            [x2, y1],
                            [x2, y2],
                            [x1, y2]])
   

    return res.stack()


@tf.function
def resize_crop(image, 
                box,
                target_height=31,
                target_width=200 ):
    
    ## for recognition ##
    
    crop = tf.image.crop_to_bounding_box(image, box[0], box[1],  box[2], box[3] )
    scale = tf.math.minimum(target_width / box[3], target_height / box[2])

    scaled_shape = [tf.cast(box[2], tf.float64)* scale, tf.cast(box[3], tf.float64)*scale]
    scaled_shape = tf.cast(scaled_shape, tf.int32)
 
    pad_h = target_height - tf.cast(scaled_shape[0], tf.int32)
    pad_w = target_width - tf.cast(scaled_shape[1], tf.int32)
    #paddings = tf.constant([[pad_h,0], [0, pad_w]])

    result_img = tf.image.resize(crop, scaled_shape)[0]
    result_img = tf.pad(result_img, [[pad_h,0], [0, pad_w], [0,0]], "CONSTANT", constant_values=0)

    return result_img 

   


@tf.function
def get_bboxes(res_img, group_num, shape, margin=0.2,):  # from connected components image

    mask = tf.where(res_img==group_num, True, False)

    coords_tensor = tf.where(mask) #get_bboxes(mask)
    coords_tensor = tf.cast(coords_tensor, tf.int32)

    y1 = tf.reduce_max(coords_tensor[:,0]) #+ margin
    #tf.cast(x1, tf.int32)

    y2 = tf.reduce_min(coords_tensor[:,0]) #- margin
    #tf.cast(x2, tf.int32)

    x1 = tf.reduce_max(coords_tensor[:,1]) #+ margin
    #tf.cast(y1, tf.int32)
    x2 = tf.reduce_min(coords_tensor[:,1]) #- margin
    #tf.cast(y2, tf.int32)
    w = tf.cast(x1 - x2, tf.int32)
    h = tf.cast(y1 - y2, tf.int32)

    return [y2, x2,   h,  w]


@tf.function
def apply_bbox(res_img, res, elem_num, shape, textmap,  margin=0.3): 
    num = 0 # item number to compensate passed zero elements
    for i in tf.range(elem_num):
        bbox = get_bboxes(res_img, i, shape, margin)
        if bbox[2] > 4 and bbox[3]  > 4 and tf.reduce_max(textmap[res_img==i]) > 0.7: # size and detector treshold 

            res = res.write(num, get_bboxes(res_img, i, shape, margin))
            num += 1


    return res.stack()     
  

@tf.function
def apply_resize_crop(box_number, boxes, crop_results, image):
    for i in tf.range(box_number): # inputs_shape[0]
       
            box = boxes[i] 

            crop_results = crop_results.write(i, resize_crop(image, box))

    return    crop_results.stack()   


class ComputeInputLayer(tf.keras.layers.Layer):


   def call(self, input):
      
        mean = tf.constant([123.6, 116.3, 103.5])
        variance = tf.constant([58.3, 57.12, 57.38])

        input -= mean  #* 255
        input /= variance  #* 255
        return input


class BboxLayer(tf.keras.layers.Layer):
    def __init__(self):

        super(BboxLayer, self).__init__()
       

    def build(self, input_shape):
       
        input_shape = tensor_shape.TensorShape(input_shape).as_list()


    def call(self, input):
 
        input_shape = array_ops.shape(input)    

        textmap = tf.identity(input[0,:,:, 0])
        linkmap = tf.identity(input[0,:,:, 1])
      
        textmap = tf.where(textmap > 0.4, 1.0, 0)
        linkmap = tf.where(linkmap > 0.4, 1.0, 0)
        res_img = tf.image.convert_image_dtype(tf.clip_by_value((textmap + linkmap), 0,1), tf.float32)
       
        filters = tf.ones([3,3,1], dtype=tf.dtypes.float32)
        strides = [1., 1., 1., 1.]
        padding = "SAME"
        dilations = [1., 1., 1., 1.]
        res_img = tf.expand_dims(res_img,0)
        res_img = tf.expand_dims(res_img,-1)

        res_img = tf.nn.dilation2d(res_img, filters, strides,
                                padding,
                                "NHWC",
                                dilations)[0,:,:,0]
        #####


        res_img = image_tfa.connected_components(res_img)
        elem_num = tf.reduce_max(res_img)
        self.res = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=False)#size=elem_num, element_shape=[4,])

     
        return  apply_bbox(res_img, self.res, elem_num, input_shape, textmap)# text_img)#self.res.stack()  #tf.convert_to_tensor(self.res)


class GrayScaleLayer(tf.keras.layers.Layer):

    def call(self, input):
        input_shape = array_ops.shape(input)
        
        img_hd = input_shape[1]/2
        img_wd = input_shape[2]/2
        input = tf.image.resize(input, [img_hd,img_wd]) 
 
        return tf.cast(tf.image.rgb_to_grayscale(input), tf.uint8)/255 


class CropBboxesLayer(tf.keras.layers.Layer): #(PreprocessingLayer):

  
    def call(self, inputs):        
                                     
        inputs_shape = array_ops.shape(inputs[1])

        res = []
        row_lengths = []
        crop = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)

        return apply_resize_crop(box_number=inputs_shape[0],
                                 boxes=inputs[1],
                                 crop_results=crop,
                                 image=inputs[0])#crop.stack()


class DecodeCharLayer(tf.keras.layers.Layer): 

    def __init__(self, alphabet):
         super(DecodeCharLayer, self).__init__()

         self.alphabet = alphabet

    def call(self, input):
        return decode_prediction(input, self.alphabet)


class DecodeBoxLayer(tf.keras.layers.Layer): 
 
    def call(self, input):
        
        box_num = array_ops.shape(input)[0]
        res = tf.TensorArray(tf.int32, size=0, dynamic_size=True, clear_after_read=False)


        return decode_bbox(input, box_num, res)



def get_recognition_part(weights, recognizer_alphabet):
    backbone, model, training_model, prediction_model = keras_ocr.recognition.build_model(recognizer_alphabet,
                                  height=31,
                                  width=200,
                                  color=False,
                                  filters=(64, 128, 256, 256, 512, 512, 512),
                                  rnn_units=(128, 128),
                                  dropout=0.25,
                                  rnn_steps_to_discard=2,
                                  pool_size=2,
                                  stn=True,)

    prediction_model.load_weights(weights)

    return prediction_model


def create_one_grap_model(detector_weights, recognizer_weights, recognizer_alphabet, debug=False):

    # if debug - output bbox images, not rectangles

    recognizer_predict_model = get_recognition_part(recognizer_weights, recognizer_alphabet)
    detector = keras_ocr.detection.Detector(weights='clovaai_general')

    detector.model.load_weights(detector_weights)

    rec_inp = tf.keras.Input([None, None, 3])
    normilized_inp = ComputeInputLayer()(rec_inp)
 
    bbox_model = detector.model(normilized_inp)

    bbox_model = BboxLayer()(bbox_model)

    grayscale_model = GrayScaleLayer()(rec_inp)   

    rec_model_func = tf.keras.models.Model(inputs=rec_inp, outputs=[bbox_model, grayscale_model])

    bboxes_model = CropBboxesLayer()([grayscale_model, bbox_model])
    bboxes_model = keras.models.Model(inputs=rec_inp, outputs=bboxes_model)
    final_model = recognizer_predict_model(bboxes_model.output)
    if debug:

        return keras.models.Model(inputs=rec_inp, outputs=[bboxes_model.output, final_model, bbox_model] )        
    
    else:
        decoded_bboxes = DecodeBoxLayer()(bbox_model)
        return keras.models.Model(inputs=rec_inp, outputs=[decoded_bboxes, final_model, ] ) # result for prod [[4,2]]



# create pipeline from one graph model

class OneGraphPipeline():
    def __init__(self, model, alphabet):

        self.model = model
        self.blank_label_idx = len(alphabet)
        self.alphabet = alphabet

    def decode_prediction(self, raw_predict): # from numerical to char groups
        
        predictions = [
            ''.join([self.alphabet[idx] for idx in row if idx not in [self.blank_label_idx, -1]])
            for row in raw_predict
        ]

        return predictions
    
            
    def get_prediction_groups(self, bboxes, char_groups):
            return [
                list(zip(predictions, boxes))
                for predictions, boxes in zip([char_groups], [bboxes])
            ]
    def compute_input(self, image):
        # should be RGB order
        #image = image.astype('float32')
        mean = np.array([0.485, 0.456, 0.406])
        variance = np.array([0.229, 0.224, 0.225])

        image -= mean
        image /= variance
        return image


    def recognize(self, images):

        #images = self.compute_input(images)

        raw_predict = self.model.predict([images])
        char_groups = self.decode_prediction(raw_predict[1])

        return self.get_prediction_groups(raw_predict[0], char_groups)


def initialize_image_ops():

    try:
        inp = tf.keras.Input([None, None, 2])
        init_model = BboxLayer()(inp)

        print('custom operation initialized...')

        return True

    except Exception as e:

        return e        
