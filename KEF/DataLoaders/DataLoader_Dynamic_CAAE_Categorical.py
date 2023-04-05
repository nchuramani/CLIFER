# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 2018
@author: nc528
"""
import numpy as np
from keras.utils import Sequence
import cv2
import numpy
from scipy.misc import imread, imresize
import dlib
from PIL import Image
def preProcess(dataLocation, imageSize, grayScale, face=None, dset="Affect"):

    try:

        frame = cv2.imread(dataLocation)
        if face is not None:
            face = list(map(int,face))
            try:
                data = frame[face[0]:face[0] + face[2], face[1]:face[1] + face[3]]

            except Exception as e:
                print(dataLocation)
                print(e)
                input("ISSUE Face")
                data = frame
        else:
            data = frame


    except Exception as e:
        print(dataLocation)
        print(e)
        input("Image Loading Failed.")

    try:
        data = imresize(data, imageSize)
    except Exception as e:
        print(data.shape)
        print(e)
        print(dataLocation)
        input("Image Resizing failed.")

    if  grayScale:
        data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        data = numpy.expand_dims(data, axis=-1)

    data = data.astype(numpy.float32)


    data = data * 2. / 255. - 1.

    return data

class DataGenerator(Sequence):

    def __init__(self, image_filenames, labels, batch_size, preProcessingProperties, faces, dset="Affect"):


        self.batch_size = batch_size
        self.imageSize = preProcessingProperties[0]
        self.grayScale = preProcessingProperties[2]
        if dset is not None:
            self.dset = dset
        else:
            self.dset = "Affect"
        self.image_filenames, self.labels = np.array(image_filenames), \
                                            np.array(labels).reshape((len(labels),labels.shape[-1]))
        if faces is not None:
            self.faces = np.array(faces)
        else:
            self.faces = None

    def __len__(self):

        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):

        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        # try:
        if self.faces is not None:
            batch_faces = self.faces[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch = np.array([
                preProcess(file_name, self.imageSize, self.grayScale, face, dset=self.dset)
                for file_name, face in zip(batch_x, batch_faces)]), batch_y
        else:
            batch = np.array([
                preProcess(file_name, self.imageSize, self.grayScale, dset=self.dset)
                for file_name in batch_x]), batch_y

        return batch

    def remaining(self, idx, remaining):
        remaining = (self.batch_size-remaining)
        batch_x = self.image_filenames[idx * self.batch_size:]
        batch_y = self.labels[idx * self.batch_size:]
        batch_x_add = np.repeat(batch_x[-1],repeats=remaining)
        batch_y_add = np.repeat(numpy.array(batch_y[-1]).reshape((1,batch_y.shape[1])),repeats=remaining,axis=0)
        batch_x = numpy.hstack([batch_x, batch_x_add])
        batch_y = numpy.vstack([batch_y, batch_y_add])
        if self.faces is not None:
            batch_faces = self.faces[idx * self.batch_size:]
            batch_faces_add = np.repeat(batch_faces[-1],repeats=remaining)
            batch_faces = numpy.hstack([batch_faces, batch_faces_add])
            batch = np.array([
                preProcess(file_name, self.imageSize, self.grayScale, face, dset=self.dset)
                for file_name, face in zip(batch_x, batch_faces)]), batch_y
        else:

            batch = np.array([
                preProcess(file_name, self.imageSize, self.grayScale, dset=self.dset)
                for file_name in batch_x]), batch_y

        return batch