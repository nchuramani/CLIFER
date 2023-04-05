
import numpy
from keras import backend as K

class TrainingMertics():

    """
    ###################################################################################
       Implementing F1 Metric
    ###################################################################################
    """

    def f1_score(self, y_true, y_pred):
        # Count positive samples.
        c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
        c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

        # If there are no true samples, fix the F1 score at 0.
        if c3 == 0:
            return 0

        # How many selected items are relevant?
        precision = c1 / c2

        # How many relevant items are selected?
        recall = c1 / c3

        # Calculate f1_score
        f1_score = 2 * (precision * recall) / (precision + recall)

        return f1_score


    """
    ###################################################################################
        Implementing PRECISION Metric
    ###################################################################################

    """


    def precision(self, y_true, y_pred):
        # Count positive samples.
        c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))

        # How many selected items are relevant?
        precision = c1 / c2

        return precision


    """
    ###################################################################################
        Implementing RECALL Metric
    ###################################################################################
    """


    def recall(self, y_true, y_pred):
        # Count positive samples.
        c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

        # If there are no true samples, fix the F1 score at 0.
        if c3 == 0:
            return 0

        # How many relevant items are selected?
        recall = c1 / c3

        return recall