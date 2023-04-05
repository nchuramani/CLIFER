# -*- coding: utf-8 -*-
import numpy
class Data:
    
    dataX = None
    dataY = None
    labelDictionary = None
    
    def __init__(self, dataX = None, dataY=None, labelDictionary=None, faces=None):

        self.dataX = dataX
        self.dataY = dataY
        self.labelDictionary = labelDictionary
        self.faces = faces
