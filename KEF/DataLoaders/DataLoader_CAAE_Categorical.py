# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 2018
@author: nc528
"""

import datetime
import os
import cv2
import numpy
import pandas
import dlib
from KEF.Models import Data
from keras.utils import np_utils

from KEF.DataLoaders import IDataLoader

class DataLoader_CAAE_Categorical(IDataLoader.IDataLoader):
    @property
    def logManager(self):
        return self._logManager

    @property
    def dataTrain(self):
        return self._dataTrain

    @property
    def dataValidation(self):
        return self._dataValidation

    @property
    def dataTest(self):
        return self._dataTest

    @property
    def preProcessingProperties(self):
        return self._preProcessingProperties

    def __init__(self, logManager, preProcessingProperties=None):

        assert (not logManager == None), "No Log Manager was sent!"

        self._preProcessingProperties = preProcessingProperties
        self._dataTrain = None
        self._dataTest = None
        self._dataValidation = None
        self._logManager = logManager


    def orderDataFolder(self, folder, append=None):

        import re
        def sort_nicely(l):
            """ Sort the given list in the way that humans expect.
            """

            convert = lambda text: int(text) if text.isdigit() else text
            alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
            l.sort(key=alphanum_key)
            return l

        dataList = sort_nicely(os.listdir(folder))
        if append is not None:
            dataList = [str(append) + "/" + im for im in dataList]
        for d in dataList:
            if d.startswith('.'):
                dataList.remove(d)

        return dataList


    def loadFileNames(self, dataFolder, dataset, personID=None, emo=None, order=None):

        assert (not dataFolder == None or not dataFolder == ""), "Empty Data Folder!"
        if order is None:

            labelDictionary = ['Angry', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        else:
            labelDictionary = order
        self.logManager.write("Label Dictionary: ")
        self.logManager.write(labelDictionary)

        dataPacketImages = []
        dataX = []
        dataPacket = []
        dataFace = []
        dataLabels = []
        classesDictionary=[]

        if dataset.startswith("RAV"):
            classes = labelDictionary
            self.logManager.write("--- Classes reading order: " + str(classes))

            classNumber = 0
            lastImage = None
            time = datetime.datetime.now()

            for c in classes:
                perClass = 0
                classesDictionary.append("'" + str(classNumber) + "':'" + str(c) + "',")
                if emo is not None:
                    if classNumber + 1 in emo:
                        print("Loading Class: ", classes[classNumber])
                        dataPointPerClass = self.orderDataFolder(dataFolder + "/" + c + "/Frames")
                    else:
                        classNumber = classNumber + 1
                        continue
                else:
                    dataPointPerClass = self.orderDataFolder(dataFolder + "/" + c + "/Frames")

                time_1 = datetime.datetime.now()

                for d in dataPointPerClass:

                    id, _ = d.split(".")
                    id = id[-2:]

                    if personID is not None:
                        if id == personID:
                            dataPointImages = self.orderDataFolder(dataFolder + "/" + c + "/Frames/" + d)
                        else:
                            continue
                    else:
                        dataPointImages = self.orderDataFolder(dataFolder + "/" + c + "/Frames/" + d)

                    numberOfImages = 0
                    for dataPointLocation in dataPointImages:

                        numberOfImages = numberOfImages + 1

                        try:
                            dataPoint = dataFolder + "/" + c + "/Frames/" + d + "/" + dataPointLocation
                            lastImage = dataPoint
                        except:
                            dataPoint = lastImage
                        dataPacketImages.append(dataPoint)

                samples = len(dataPacketImages) // self.preProcessingProperties[4]
                # Compute samples by dividing the total number for frames by half of the frame-rate
                n = 1
                perClass += samples - 2 * n

                for i in range(n, samples - 1):
                    dataImages = dataPacketImages[
                                 i * self.preProcessingProperties[4]: (i + 1) *
                                                                      self.preProcessingProperties[4]]
                    dataImage = dataImages[-1]
                    dataPacket.append(dataImage)
                    dataLabels.append(classNumber)
                    dataImage = dataImages[-2]
                    dataPacket.append(dataImage)
                    dataLabels.append(classNumber)

                dataPacketImages = []

                self.logManager.write(
                    "--- Class: " + str(c) + "(Label: " + str(classNumber)  + "  " + str(len(dataPointPerClass)) + " Videos  - " + "(" + str(
                        perClass) + " Data Points - " +
                    str((datetime.datetime.now() - time_1).total_seconds()) + " seconds" + "))")
                classNumber = classNumber + 1

            dataX = numpy.array(dataPacket)
            dataLabels = np_utils.to_categorical(dataLabels, len(labelDictionary))

        elif dataset.startswith("MMI"):

            subjects = self.orderDataFolder(dataFolder)
            subjects_with_emotions = []
            for subject in subjects:
                if personID is not None:
                    if subject == personID:
                        subjects_with_emotions.append(subject)
                    else:
                        continue
                else:
                    subjects_with_emotions = subjects

            classes = labelDictionary
            self.logManager.write("--- Classes reading order: " + str(classes))
            classNumber = 0
            lastImage = None
            time = datetime.datetime.now()

            for subject in subjects_with_emotions:
                for c in classes:
                    perClass = 0
                    classesDictionary.append("'" + str(classNumber) + "':'" + str(c) + "',")
                    if emo is not None:
                        if classNumber + 1 in emo:
                            print("Loading Class: ", classes[classNumber])
                            dataPointPerClass = self.orderDataFolder(dataFolder + "/" + subject + "/" + c)
                        else:
                            classNumber = classNumber + 1
                            continue
                    else:
                        dataPointPerClass = self.orderDataFolder(dataFolder + "/" +  subject + "/" + c)

                    time_1 = datetime.datetime.now()
                    for d in dataPointPerClass:
                        dataPointImages = self.orderDataFolder(dataFolder + "/" + subject  + "/" + c + "/" + d)

                        numberOfImages = 0
                        for dataPointLocation in dataPointImages:

                            numberOfImages = numberOfImages + 1

                            try:
                                dataPoint = dataFolder + "/" + subject  + "/" + c + "/" + d + "/" + dataPointLocation
                                lastImage = dataPoint
                            except:
                                dataPoint = lastImage

                            dataPacketImages.append(dataPoint)
                    samples = len(dataPacketImages) // self.preProcessingProperties[4]
                    if samples == 0:
                        samples = 1
                    # Compute samples by dividing the total number for frames by half of the frame-rate
                    n = 0
                    perClass += samples - 2 * n
                    for i in range(n, samples):
                        if len(dataPacketImages) < self.preProcessingProperties[4]:
                            for dataImage in dataPacketImages:
                                dataPacket.append(dataImage)
                                dataLabels.append(classNumber)

                        else:
                            dataImages = dataPacketImages[
                                         i * self.preProcessingProperties[4]: (i + 1) *
                                                                              self.preProcessingProperties[4]]
                            if len(dataImages) > 1:

                                for i in range(len(dataImages)//2-5,len(dataImages)//2+5):
                                    dataPacket.append(dataImages[i])
                                    dataLabels.append(classNumber)

                            else:
                                dataImage = dataImages[0]
                                dataPacket.append(dataImage)
                                dataLabels.append(classNumber)
                    dataPacketImages = []
                    self.logManager.write(
                        "--- Class: " + str(c) + "(Label: " + str(classNumber) + "  " + str(
                            len(dataPointPerClass)) + " Videos  - " + "(" + str(
                            perClass) + " Data Points - " +
                        str((datetime.datetime.now() - time_1).total_seconds()) + " seconds" + "))")
                    classNumber = classNumber + 1

            while(len(dataPacket) < self.preProcessingProperties[-1]):
                dataPacket.append(dataPacket[len(dataPacket)//2])
                dataLabels.append(dataLabels[len(dataLabels)//2])
            dataX = numpy.array(dataPacket)
            dataLabels = np_utils.to_categorical(dataLabels, len(labelDictionary))

        elif dataset.startswith("BAUM"):

            subjects = self.orderDataFolder(dataFolder)
            subjects_with_emotions = []
            for subject in subjects:
                if personID is not None:
                    if subject == personID:
                        subjects_with_emotions.append(subject)
                    else:
                        continue
                else:
                    subjects_with_emotions = subjects

            classes = labelDictionary
            self.logManager.write("--- Classes reading order: " + str(classes))
            classNumber = 0
            lastImage = None
            time = datetime.datetime.now()

            for subject in subjects_with_emotions:
                for c in classes:
                    perClass = 0
                    classesDictionary.append("'" + str(classNumber) + "':'" + str(c) + "',")
                    if emo is not None:
                        if classNumber + 1 in emo:
                            print("Loading Class: ", classes[classNumber])
                            dataPointPerClass = self.orderDataFolder(dataFolder + "/" + subject + "/" + c)
                        else:
                            classNumber = classNumber + 1
                            continue
                    else:
                        dataPointPerClass = self.orderDataFolder(dataFolder + "/" +  subject + "/" + c)

                    time_1 = datetime.datetime.now()
                    for d in dataPointPerClass:
                        dataPointImages = self.orderDataFolder(dataFolder + "/" + subject  + "/" + c + "/" + d)

                        numberOfImages = 0
                        for dataPointLocation in dataPointImages:

                            numberOfImages = numberOfImages + 1

                            try:
                                dataPoint = dataFolder + "/" + subject  + "/" + c + "/" + d + "/" + dataPointLocation
                                lastImage = dataPoint
                            except:
                                dataPoint = lastImage
                            dataPacketImages.append(dataPoint)

                    samples = len(dataPacketImages) // self.preProcessingProperties[4]

                    # Compute samples by dividing the total number for frames by half of the frame-rate
                    n = 0
                    perClass += samples - 2 * n

                    for i in range(n, samples):
                        if len(dataPacketImages) < self.preProcessingProperties[4]:
                            dataImages = dataPacketImages
                        else:
                            dataImages = dataPacketImages[
                                         i * self.preProcessingProperties[4]: (i + 1) *
                                                                              self.preProcessingProperties[4]]
                        dataImage = dataImages[-1]
                        dataPacket.append(dataImage)
                        dataLabels.append(classNumber)
                        dataImage = dataImages[-2]
                        dataPacket.append(dataImage)
                        dataLabels.append(classNumber)

                    dataPacketImages = []
                    self.logManager.write(
                        "--- Class: " + str(c) + "(Label: " + str(classNumber) + "  " + str(
                            len(dataPointPerClass)) + " Videos  - " + "(" + str(
                            perClass) + " Data Points - " +
                        str((datetime.datetime.now() - time_1).total_seconds()) + " seconds" + "))")
                    classNumber = classNumber + 1


            while(len(dataPacket) < self.preProcessingProperties[-1]):
                dataPacket.append(dataPacket[-1])
                dataLabels.append(dataLabels[-1])
            dataX = numpy.array(dataPacket)
            dataLabels = np_utils.to_categorical(dataLabels, len(labelDictionary))

        self.logManager.write("----- Images: " + str(len(dataX)) + " Data points - " +
        str((datetime.datetime.now() - time).total_seconds()) + " seconds" + ")")

        dataX = numpy.array(dataX)
        dataY = numpy.array(dataLabels)

        return Data.Data(dataX,dataY,labelDictionary)

    def loadTrainData(self, dataFolder, dataset=None, personID=None,emo=None, order=None):
        self.logManager.newLogSession("Training Data")
        self.logManager.write("Loading From: " + dataFolder)
        self.logManager.write("Preprocessing Properties: " + str(self.preProcessingProperties))
        self._dataTrain = self.loadFileNames(dataFolder, dataset=dataset, personID=personID, emo=emo, order=order)
        self.logManager.write("Total data points: " + str(len(self.dataTrain.dataX)))
        self.logManager.write("Data points shape: " + str(numpy.array(self.dataTrain.dataX).shape))
        self.logManager.write("Label dictionary: " + str(self.dataTrain.labelDictionary))

        self.logManager.endLogSession()

    def loadTestData(self, dataFolder, dataset=None, personID=None,emo=None, order=None):
        self.logManager.newLogSession("Testing Data")
        self.logManager.write("Loading From: " + dataFolder)
        self.logManager.write("Preprocessing Properties: " + str(self.preProcessingProperties))
        self._dataTest = self.loadFileNames(dataFolder, dataset=dataset, personID=personID, emo=emo, order=order)

        self.logManager.write("Total data points: " + str(len(self.dataTest.dataX)))
        self.logManager.write("Data points shape: " + str(numpy.array(self.dataTest.dataX).shape))
        self.logManager.write("Label dictionary: " + str(self.dataTest.labelDictionary))

        self.logManager.endLogSession()

    def saveData(self, folder):
        pass

    def shuffleData(self, folder):
        pass

    def loadTrainTestValidationData(self, folder, percentage):
        pass

    def loadNFoldValidationData(self, folder, NFold):
        pass