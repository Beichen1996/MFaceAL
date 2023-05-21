from utils.api import *
import os
import sys
from sys import argv
import numpy as np
from sampling import AL_Greedy 


def samplingforever(infilepath):
    infoarray, featurearray = feature_ex(infilepath)

def feature_independent(filepath):
    init_mask = FACE_DETECT|FACERECOGNITION|LANDMARKER5
    seetaFace = SeetaFace(init_mask)
    seetaFace.SetProperty(DetectProperty.PROPERTY_MIN_FACE_SIZE,35)
    seetaFace.SetProperty(DetectProperty.PROPERTY_THRESHOLD,0.9)
    name = os.listdir(filepath)
    featurearray = []
    infoarray = []
    for i in name:
        imagepath = filepath + i
        image = cv2.imread(imagepath)
        detect_result1 = seetaFace.Detect(image)
        num = detect_result1.size
        infoarray.append([imagepath, num])
        local_feature = []
        for j in range(num):
            infoarray.append([imagepath, num])
            face = detect_result1.data[j].pos
            points1 = seetaFace.mark5(image,face)
            feature1 = seetaFace.Extract(image,points1)
            local_feature.append(feature1)
        featurearray.append(local_feature)
        #face1 = detect_result1.data[0].pos
        #points1 = seetaFace.mark5(image,face1)
        #feature1 = seetaFace.Extract(image,points1)
        #print(imagepath, num)
        
    return infoarray, featurearray


def sampling2(infilepath):
    output = []
    infoarray, featurearray, idarray= feature_ex2(infilepath)
    #print(len(infoarray), len(featurearray), len(idarray))
    featurearray = np.stack(featurearray)
    #print(featurearray.shape) 
    model = AL_Greedy(featurearray)
    sample_size = int(0.03 * len(idarray))
    results = model.select_batch_(None, [], sample_size)

    #print(results)
    image_results = []
    for i in results:
        image_results.append(idarray[i])
    #print(image_results)
    image_results.sort()
    #print(image_results)
    image_results = list(set(image_results))
    #print(image_results)
    for i in image_results:
        #print(infoarray[i], i)
        output.append(infoarray[i][0])
    return output


def feature_ex2(filepath):
    init_mask = FACE_DETECT|FACERECOGNITION|LANDMARKER5
    seetaFace = SeetaFace(init_mask)
    seetaFace.SetProperty(DetectProperty.PROPERTY_MIN_FACE_SIZE,35)
    seetaFace.SetProperty(DetectProperty.PROPERTY_THRESHOLD,0.9)
    name = os.listdir(filepath)
    featurearray = []
    infoarray = []
    idarray = []
    for i in name:
        imagepath = filepath + i
        image = cv2.imread(imagepath)
        detect_result1 = seetaFace.Detect(image)
        num = detect_result1.size
        infoarray.append([imagepath, num])
        ids = len(infoarray) - 1
        for j in range(num):
            face = detect_result1.data[j].pos
            points1 = seetaFace.mark5(image,face)
            feature1 = seetaFace.Extract(image,points1)
            feature1 = seetaFace.get_feature_numpy(feature1)
            #print(feature1.shape)
            featurearray.append(feature1)
            idarray.append(ids)

        #face1 = detect_result1.data[0].pos
        #points1 = seetaFace.mark5(image,face1)
        #feature1 = seetaFace.Extract(image,points1)
        #print(imagepath, num)
        
    return infoarray, featurearray, idarray

def copyfiles():
    pathsss = ['tt1699513_jd_1_347_356_00025.jpg','tt1699513_jd_1_347_356_00073.jpg','tt1699513_jd_1_347_356_00241.jpg', 'tt1699513_jd_1_347_356_00277.jpg', 'tt1699513_jd_1_347_356_00397.jpg', 'tt1699513_jd_1_347_356_00475.jpg', 'tt1699513_jd_1_347_356_00493.jpg', 'tt1699513_jd_1_347_356_00619.jpg', 'tt1699513_jd_1_347_356_00667.jpg' ]
    return pathsss