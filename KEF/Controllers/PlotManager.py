# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy
from keras.utils import plot_model, np_utils
import pylab as P
import os
class PlotManager():
    


    @property
    def plotsDirectory(self):
        return self._plotsDirectory
        
        
    def __init__(self, plotDirectory):
        
        self._plotsDirectory = plotDirectory

    def  creatModelPlot(self, model, modelName=""):
        print ("Creating Plot at: " + str(self.plotsDirectory) + "/" + str(modelName) + "_plot.png")
        plot_model(model,to_file=self.plotsDirectory+"/"+modelName + "_plot.png",show_layer_names=True, show_shapes=True)

        
    def createTrainingPlot(self, trainingHistory, modelName=""):
        
        metrics = list(trainingHistory.history)

        for m in metrics:
            if m.startswith('lr'):
                metrics.remove(m)

        for i in range(len(metrics)):            
            
            if not "val_" in metrics[i]:

                plt.plot(trainingHistory.history[metrics[i]])
                plt.plot(trainingHistory.history["val_"+metrics[i]])
                
                
                plt.title("Model's " + metrics[i])
                
                plt.ylabel(metrics[i])
                plt.xlabel('epoch')
                plt.legend(['train', 'validation'], loc='upper left')
                plt.savefig(self.plotsDirectory+"/"+modelName+metrics[i]+".png")
                plt.clf()

    def plotLoss(self,loss_history):
        plt.gcf().clear()
        plt.figure(1)
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.plot(loss_history, c='b')
        plt.savefig("loss.png")
        plt.close()


    def plotGenloss(self,losses):
        for key, loss in losses.items():
            plt.gcf().clear()
            plt.figure()
            plt.ylabel('Avg. Loss per Epoch')
            plt.xlabel('Epochs')
            plt.plot(loss, label=key)
            plt.savefig(self.plotsDirectory+"/"+key+"_Loss.png")
            plt.close()
    def plotGenlossBatch(self,losses, batch=True):
        # number = len(losses)
        for key, loss in losses.items():
            plt.gcf().clear()
            plt.figure()
            if batch:
                plt.ylabel('Avg. Loss per Batch')
                plt.xlabel('Batch')
            else:
                plt.ylabel('Avg. Loss per Epoch')
                plt.xlabel('Epoch')
            plt.plot(loss, label=key)
            plt.savefig(self.plotsDirectory+"/"+key+"_Loss.png")
            plt.close()
    def plotMultiAcc(self,acc, title=None, label=None, location=None):
        for key, ac in acc.items():
            plt.gcf().clear()
            plt.figure()
            plt.ylabel('Accuracy')
            plt.xlabel('Classes Seen')
            plt.plot(ac, label=key)
            if title is not None:
                plt.title(title)
            plt.legend(loc="best")
            if location is None:
                plt.savefig(self.plotsDirectory+"/"+key+"_acc.png",bbox_inches='tight')
            else:
                if not os.path.exists(location + "/Individual_Plots/"):
                    os.makedirs(location + "/Individual_Plots/")
                plt.savefig(location + "/Individual_Plots/" + key + "_acc.png", bbox_inches='tight')
            plt.close()
    def plotMultiF1(self,f1, title=None, label=None, location=None):
        for key, ac in f1.items():
            plt.gcf().clear()
            plt.figure()
            plt.ylabel('F1 Score')
            plt.xlabel('Classes Seen')
            plt.plot(ac, label=key)
            if title is not None:
                plt.title(title)
            plt.legend(loc="best")
            if location is None:
                plt.savefig(self.plotsDirectory+"/"+key+"_f1.png",bbox_inches='tight')
            else:
                if not os.path.exists(location + "/Individual_Plots/"):
                    os.makedirs(location + "/Individual_Plots/")
                plt.savefig(location + "/Individual_Plots/" + key + "_f1.png", bbox_inches='tight')
            plt.close()


    def plotAcc(self,acc_history):
        plt.gcf().clear()
        plt.figure(1)
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.plot(acc_history, c='b')
        plt.savefig("acc.png")
        plt.close()

    def plotOutput(self,faceX, faceY):
        plt.gcf().clear()
        plt.figure(1)
        plt.ylabel('Y')
        plt.xlabel('X')
        plt.plot(faceX, c='b')
        plt.plot(numpy.tranpose(faceY), c='r')
        plt.savefig("face_plot.png")
        plt.close()
        
        
